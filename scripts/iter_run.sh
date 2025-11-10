MAINTAIN_STRATEGY=all
# SAMPLING_NUMBER=3
SAMPLING_NUMBER=16

if [ -z "$model_base" ]; then 
    model_base="llama38b"
fi

BASE_PATH=directory/of/training_set

# this is to save the synthesized training data
TASK_PATH=${BASE_PATH}/datasets/medical_train

model_base=llama318b
MODEL_PATH=/path/to/your/Meta-Llama-3.1-8B-Instruct
CKPT=llama3.1-8b-r16a32-1epoch

# TRAIN_DATA_NAME=PubmedQA

# this will create 8 cards to synthesize data
NUM_CHUNKS=8
BATCH_SIZE=1

CONFIG_NAME=trial5

TRAIN_DATA_NAME="$3"


################################## ITER 1 ##################################
iter=1
SELFGEN_LORA_PATH=None
INPUT_DATA=/directory/of/seed_data/mix16_500_data.json

SAVE_PATH=${model_base}_${METHOD}_${TRAIN_DATA_NAME}


OUTPUT_PATH=${TASK_PATH}/${SAVE_PATH}/sft_${iter}.jsonl

bash scripts/chunk_iter_self_gen_mcts.sh $OUTPUT_PATH $MODEL_PATH $INPUT_DATA $LOG_PATH $SAMPLING_NUMBER $SELFGEN_LORA_PATH  $iter $NUM_CHUNKS 


TRAINING_DATA=${SAVE_PATH}/sft_${iter}

SFT_CKPT=${CKPT}-SFT-ITER${iter}

SFT_MODEL_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${SFT_CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${SFT_CKPT}

if [ ! -f "${SFT_MODEL_PATH}/tokenizer_config.json" ]; then
    
    sbatch -o ${LOGS_BASE_PATH}/${SFT_CKPT}/train.log  scripts/slurm/sft_train.sh $TASK_PATH $TRAINING_DATA $MODEL_PATH $SFT_MODEL_PATH None "--learning_rate 1e-6 --num_train_epochs 1"
fi 



# first iter over, turn to next iter 
############################## END OF ITER 1 ##############################

################################## ITER 2 ##################################
SELFGEN_LORA_PATH=$SFT_MODEL_PATH
iter=2
INPUT_DATA=/directory/of/seed_data/mix16_500_data_filter.json

SAVE_PATH=${model_base}_${METHOD}_${TRAIN_DATA_NAME}

SHOTS=1 # at first gen iteration, we prompt with one-shot example
OUTPUT_PATH=${TASK_PATH}/${SAVE_PATH}/sft_${iter}.jsonl

bash scripts/chunk_iter_self_gen_mcts.sh $OUTPUT_PATH $MODEL_PATH $INPUT_DATA $LOG_PATH $SAMPLING_NUMBER $SELFGEN_LORA_PATH  $iter $NUM_CHUNKS 

# combine first iter/second iter sft data
srun -p medai_llm_p --debug python Evol_Instruct/utils/combine_json.py \
    --data_path ${TASK_PATH}/${SAVE_PATH} \
    --save_path ${TASK_PATH}/${SAVE_PATH}/sft_combined_${iter}.json \
    --num ${iter} \
    --prefix sft \
    --only_iter
fi


TRAINING_DATA=${SAVE_PATH}/sft_combined_${iter}

SFT_CKPT=${CKPT}-SFT-ITER${iter}
SFT_MODEL_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${SFT_CKPT}


if [ ! -f "${SFT_MODEL_PATH}/tokenizer_config.json" ]; then
    sbatch -o ${LOGS_BASE_PATH}/${SFT_CKPT}/train.log  scripts/slurm/sft_train.sh $TASK_PATH $TRAINING_DATA $MODEL_PATH $SFT_MODEL_PATH $SELFGEN_LORA_PATH "--learning_rate 1e-6 --num_train_epochs 1"
fi 

# waiting for the training over
while [ ! -f "${SFT_MODEL_PATH}/tokenizer_config.json" ]; do
    echo "Waiting for ${SFT_MODEL_PATH}/tokenizer_config.json to be generated..."
    sleep 60    
done

DPO_TRAINING_DATA=${SAVE_PATH}/sft_${iter}
# combine over, begin sft training
DPO_CKPT=${CKPT}-DPO-full-ITER${iter}-ls
# DPO_CKPT=${CKPT}-DPO-full-ITER${iter}-ls-only2
DPO_MODEL_PATH=${BASE_PATH}/checkpoints/${DPO_TRAINING_DATA}-${DPO_CKPT}
# model_base=
DPO_LOGS_BASE=./logs/${DPO_TRAINING_DATA}/${DPO_CKPT}
mkdir -p ${DPO_LOGS_BASE}
LOGS_FILE=${DPO_LOGS_BASE}/train.log


if [ ! -f "${DPO_MODEL_PATH}/tokenizer_config.json" ]; then
    sbatch -o $LOGS_FILE scripts/slurm/dpo_train.sh $oss_path/${TRAINING_DATA}.json $SFT_MODEL_PATH ${DPO_MODEL_PATH} "None" "--use_peft --learning_rate 1e-6 --num_train_epochs 1 --test_split_ratio 0.01 --data_process_method ls" 
fi 

# waiting for the dpo training over
while [ ! -f "${DPO_MODEL_PATH}/tokenizer.json" ]; do
    echo "Waiting for ${DPO_MODEL_PATH}/tokenizer.json to be generated..."
    sleep 60 
done
VALUE_TRAINING_DATA=${SAVE_PATH}/sft_${iter}
VALUE_CKPT=${CKPT}-prm_hardtrain_baseDPO-ITER${iter}-bceall-unify-lb

VALUE_MODEL_PATH=${BASE_PATH}/checkpoints/${VALUE_TRAINING_DATA}-${VALUE_CKPT}


LOGS_BASE_PATH=./logs/${VALUE_TRAINING_DATA}/${VALUE_CKPT}
mkdir -p ${LOGS_BASE_PATH}
LOGS_FILE=${LOGS_BASE_PATH}/train.log
echo $VALUE_MODEL_PATH

if [ ! -f "${VALUE_MODEL_PATH}/tokenizer.json" ]; then

    
    sbatch -o $LOGS_FILE scripts/slurm/prm_train_lora_r32a64.sh ${oss_path}/${VALUE_TRAINING_DATA}.json $DPO_MODEL_PATH ${VALUE_MODEL_PATH} None "--use_peft --learning_rate 5e-5 --num_train_epochs 2 --test_split_ratio 0.01 --train_pair_per_instance 0  --filter_invalid True --use_soft_training False --positive_thr 0.0 --loss_type bce --look_back_factor 1.0 "
fi

# the following is optional for test on the selective benchmarks 
bash scripts/eval/eval_models_per_dataset_med.sh $DPO_MODEL_PATH $DPO_TRAINING_DATA ${DPO_CKPT} sc 32 1 
bash scripts/rescore.sh $DPO_TRAINING_DATA ${DPO_CKPT} sc 32 1 "$DPO_MODEL_PATH" "$VALUE_MODEL_PATH" hard-dpo-unifybce-lb



############################## END OF ITER 2 ##############################

INPUT_DATA=~/datasets/medical_train/${TRAIN_DATA_NAME}.json
CONFIG_PATH=Evol_Instruct/config/${CONFIG_NAME}.json

SELFGEN_LORA_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-SFT-ITER1
VALUE_FUNCTION=None
for iter in {2..2}; do
    echo "Begin Iteration: ${iter}"
    ##################################CREATE FILES##################################
    SAVE_PATH=${model_base}_${METHOD}_vllm_${TRAIN_DATA_NAME}_${MAINTAIN_STRATEGY}_${CONFIG_NAME}
    # SAVE_PATH=${model_base}_${METHOD}_vllm_${TRAIN_DATA_NAME}_${MAINTAIN_STRATEGY}_${CONFIG_NAME}_${SAMPLE_NUM}
    # if [[ $SAMPLE_NUM == *"-"* ]]; then
    #     # SAMPLE_NUM=${SAMPLE_NUM//"-"/_}
    #     SAVE_PATH=${model_base}_${METHOD}_vllm_${TRAIN_DATA_NAME}_${MAINTAIN_STRATEGY}_${CONFIG_NAME}_${SAMPLE_NUM}
    # fi
    OUTPUT_PATH=${oss_path}/${SAVE_PATH}/sft_${iter}.jsonl
    # echo $OUTPUT_PATH
    mkdir -p ${oss_path}/${SAVE_PATH}
    
    mkdir -p logs/${SAVE_PATH}
    LOG_PATH=logs/${SAVE_PATH}/gen_${iter}.log
    

    ###################################RUN SELFGEN###################################
    if [ $iter -eq 1 ]; then
        SHOTS=1
    else 
        SHOTS=0
    fi
    # sbatch scripts/slurm/iter_self_gen_sc_llama.sh $OUTPUT_PATH $MODEL_PATH $SAMPLE_NUM $INPUT_DATA $LOG_PATH $METHOD $BATCH_SIZE $SAMPLING_NUMBER $MAINTAIN_STRATEGY $SELFGEN_LORA_PATH $SHOTS
    JSON_OUTPUT_PATH=${TASK_PATH}/${SAVE_PATH}/sft_${iter}.json
    # if ! aws s3 ls "$JSON_OUTPUT_PATH" > /dev/null 2>&1; then
    # # if [ ! -f ${JSON_OUTPUT_PATH} ]; then
    #     bash scripts/chunk_iter_self_gen_mcts.sh $OUTPUT_PATH $MODEL_PATH $SAMPLE_NUM $INPUT_DATA $LOG_PATH $METHOD $BATCH_SIZE $SAMPLING_NUMBER $MAINTAIN_STRATEGY $SELFGEN_LORA_PATH $SHOTS $CONFIG_PATH $iter $NUM_CHUNKS $VALUE_FUNCTION 
    # fi
    # waiting for combine
    # while ! aws s3 ls "$JSON_OUTPUT_PATH" > /dev/null 2>&1; do
    #     echo "Waiting for ${JSON_OUTPUT_PATH} to be generated..."
    #     sleep 60
    # while [ ! -f ${JSON_OUTPUT_PATH} ]; do
    #     echo "Waiting for ${JSON_OUTPUT_PATH} to be generated..."
    #     sleep 60
    # done
    # echo "Generating data over"
    # sleep 2
    #####################################COMBINE DATA################################
    # if [ ! -f "${TASK_PATH}/${SAVE_PATH}/sft_combined_${iter}.json" ]; then
    # if ! aws s3 ls "${TASK_PATH}/${SAVE_PATH}/sft_combined_${iter}.json" > /dev/null 2>&1; then
    #     srun -p medai_llm_p --debug python Evol_Instruct/utils/combine_json.py \
    #         --data_path ${oss_path}/${SAVE_PATH} \
    #         --save_path ${oss_path}/${SAVE_PATH}/sft_combined_${iter}.json \
    #         --num ${iter} \
    #         --prefix sft \
    #         --only_iter
    # fi

    # if [ ! -f "${TASK_PATH}/${SAVE_PATH}/value_combined_${iter}.json" ]; then
    # if ! aws s3 ls "${TASK_PATH}/${SAVE_PATH}/value_combined_${iter}.json" > /dev/null 2>&1; then
    #     srun -p medai_llm_p --debug python Evol_Instruct/utils/combine_json.py \
    #         --data_path ${TASK_PATH}/${SAVE_PATH} \
    #         --save_path ${TASK_PATH}/${SAVE_PATH}/value_combined_${iter}.json \
    #         --num ${iter} \
    #         --prefix value_sft \
    #         --only_iter
    # fi
    echo "Combining data over"
    # break
    #######################################SFT TRAINING AND DPO TRAINING################################
    # if [ $iter -eq 1 ]; then
    #     model_base=$MODEL_PATH
    # else
    #     model_base="${BASE_PATH}/checkpoints/${SAVE_PATH}/sft_combined_1-${CKPT}-SFT-ITER1-merged"
    #     if [ ! -f "$model_base/config.json" ]; then
    #         srun -p medai_llm_p --debug python Evol_Instruct/utils/merge_lora_weights.py \
    #             --model_base $MODEL_PATH \
    #             --model_path "${BASE_PATH}/checkpoints/${SAVE_PATH}/sft_combined_1-${CKPT}-SFT-ITER1" \
    #             --save_path $model_base
    #         echo "Merge Iter1's Lora weights to make $model_base new base model"
    #     fi
    # fi 
    model_base="${BASE_PATH}/checkpoints/${SAVE_PATH}/sft_1-${CKPT}-SFT-ITER1-merged"
    # if [ ! -f "$model_base/config.json" ]; then
    #     srun -p medai_llm_p --debug python Evol_Instruct/utils/merge_lora_weights.py \
    #         --model_base $MODEL_PATH \
    #         --model_path "/mnt/petrelfs/jiangshuyang/checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-SFT-ITER1" \
    #         --save_path $model_base
    #     echo "Merge Iter1's Lora weights to make $model_base new base model"
    # fi
    # break
    model_base=$MODEL_PATH
    TRAINING_DATA=${SAVE_PATH}/sft_${iter}
    # TRAINING_DATA=${SAVE_PATH}/sft_combined_${iter}
    # combine over, begin sft training
    SFT_CKPT=${CKPT}-SFT-ITER${iter}
    # SFT_CKPT=${CKPT}-SFT-ITER${iter}-fromiter1
    # SFT_CKPT=${CKPT}-SFT-full-ITER${iter}
    SFT_MODEL_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${SFT_CKPT}
    SFT_CONFIG_S3_PATH=s3://syj_test/checkpoints/${TRAINING_DATA}-${SFT_CKPT}/adapter_config.json

    # if [ ! -f "${SFT_MODEL_PATH}/tokenizer_config.json" ]; then
    #     bash scripts/train/sft_lora_r16_iter.sh ${TRAINING_DATA} $model_base ${SFT_CKPT} None \
    #     "--learning_rate 2e-4 --use_peft --lora_r 16 --lora_alpha 32 --num_train_epochs 1"
    # fi 
    # if [ ! -f "${SFT_MODEL_PATH}/tokenizer_config.json" ]; then
    #     bash scripts/train/sft_lora_r16_iter.sh ${TRAINING_DATA} $model_base ${SFT_CKPT} "/mnt/petrelfs/jiangshuyang//checkpoints/llama318b_mcts_vllm_mix16_500_data_all_trial5/sft_1-llama3.1-8b-r16a32-1epoch-SFT-ITER1" \
    #     "--learning_rate 2e-4 --use_peft --lora_r 16 --lora_alpha 32 --num_train_epochs 1"
    # fi 

    # break
    # combine over, begin Value Function training
    # VALUE_TRAINING_DATA=${SAVE_PATH}/sft_combined_${iter}
    VALUE_TRAINING_DATA=${SAVE_PATH}/sft_${iter}
    # VALUE_CKPT=${CKPT}-VALUE-prm_train4_r64-ITER${iter}
    # VALUE_CKPT=${CKPT}-VALUE-prm_trainall3_r64-ITER${iter}
    # VALUE_CKPT=${CKPT}-VALUE-prm_train5_full-ITER${iter}
    # VALUE_CKPT=${CKPT}-VALUE-prm_train5_r64_select_basepolicy-ITER${iter}
    # VALUE_CKPT=${CKPT}-VALUE-prm_trainall_r64_select-ITER${iter}
    # VALUE_CKPT=${CKPT}-VALUE-prm_train5_r64_softtrain-ITER${iter}
    VALUE_CKPT=${CKPT}-VALUE-prm_trainall_r64_softtrain_basepolicy-ITER${iter}
    # VALUE_CKPT=${CKPT}-VALUE-prm_trainall_r64_softtrain_basepolicy-ITER${iter}_mix

    # VALUE_CKPT=${CKPT}-VALUE-prm_train4_lenprior2_r64-ITER${iter}
    VALUE_MODEL_PATH=${BASE_PATH}/checkpoints/${VALUE_TRAINING_DATA}-${VALUE_CKPT}

    # VALUE_MODEL_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/llama38b_mcts_vllm_mmed_en_train_all_trial5/sft_combined_1-llama3-8b-r16a32-1epoch-VALUE-new-ITER1
    VALUE_CONFIG_S3_PATH=s3://syj_test/checkpoints/${VALUE_TRAINING_DATA}-${VALUE_CKPT}/adapter_config.json
    # if ! aws s3 ls "$VALUE_CONFIG_S3_PATH" > /dev/null 2>&1; then
    #     http_proxy=$proxy_url
    #     HTTP_PROXY=$proxy_url
    #     https_proxy=$proxy_url
    #     HTTPS_PROXY=$proxy_url
        # bash scripts/train/dpo_lora_iter.sh ${VALUE_TRAINING_DATA} $MODEL_PATH $CKPT-DPO-ITER${iter}
    #     unset http_proxy
    #     unset HTTP_PROXY
    # fi


    LOGS_BASE_PATH=./logs/${VALUE_TRAINING_DATA}/${VALUE_CKPT}
    mkdir -p ${LOGS_BASE_PATH}
    LOGS_FILE=${LOGS_BASE_PATH}/train.log

    # if [ ! -f "${VALUE_MODEL_PATH}/adapter_config.json" ]; then
    #     sbatch -o $LOGS_FILE scripts/slurm/prm_train_lora_r16a32.sh ${oss_path}/${VALUE_TRAINING_DATA}.json $model_base ${VALUE_MODEL_PATH} "${SFT_MODEL_PATH}" "--learning_rate 2e-4 --num_train_epochs 1 --test_split_ratio 0.01 --train_pair_per_instance 5 --use_peft --filter_invalid True"
    # fi
    while [ ! -f "${SFT_MODEL_PATH}/adapter_config.json" ]; do 
        echo "Waiting for ${SFT_MODEL_PATH}/adapter_config.json to be generated..."
        sleep 60
    done
    if [ ! -f "${VALUE_MODEL_PATH}/adapter_config.json" ]; then
        sbatch -o $LOGS_FILE scripts/slurm/prm_train_lora_r16a32.sh ${oss_path}/${VALUE_TRAINING_DATA}.json $model_base ${VALUE_MODEL_PATH} "${SFT_MODEL_PATH}" "--learning_rate 2e-4 --num_train_epochs 1 --test_split_ratio 0.01 --train_pair_per_instance 10 --use_peft --filter_invalid True --use_soft_training True"
    fi
    # break
    echo "Policy and Value training over"
    # break
    #######################################UPDATE SELFGEN MODEL#########################################
    bash /mnt/petrelfs/jiangshuyang/add_oss.sh
    while [ ! -f "${SFT_MODEL_PATH}/tokenizer_config.json" ]; do
    # while ! aws s3 ls "$SFT_CONFIG_S3_PATH" > /dev/null 2>&1; do
        echo "Waiting for ${SFT_MODEL_PATH}/tokenizer_config.json to be generated..."
        sleep 60
    done
    
    SELFGEN_LORA_PATH=${SFT_MODEL_PATH}
    VALUE_FUNCTION=${VALUE_MODEL_PATH}
    echo "Inference with Greedy"
    
    # bash scripts/eval/eval_models_per_dataset_med.sh $model_base $TRAINING_DATA ${SFT_CKPT} greedy 1 1 
    # break
    echo "Inference with SC"
    # bash scripts/eval/eval_models_per_dataset_med.sh $model_base $TRAINING_DATA ${SFT_CKPT} sc 16 1
    # bash scripts/eval/eval_models_per_dataset_med.sh $MODEL_PATH $TRAINING_DATA ${CKPT}-SFT-ITER${iter} dpo_judge 16 4 $DPO_MODEL_PATH base vote
    # while [ ! -f "${VALUE_MODEL_PATH}/adapter_config.json" ]; do
    # # while ! aws s3 ls "$VALUE_CONFIG_S3_PATH" > /dev/null 2>&1; do
    #     echo "Waiting for ${VALUE_MODEL_PATH}/adapter_config.json to be generated..."
    #     sleep 60
    # done
    bash scripts/eval/eval_models_per_dataset_med.sh $model_base $TRAINING_DATA ${SFT_CKPT} scvm 16 1 "None" "None" "None" "$model_base" "$SFT_MODEL_PATH $VALUE_FUNCTION" prm-min-max prm_softtrain_basepolicy

    echo "Inference with MCTS"
    # bash scripts/eval/eval_models_per_dataset_med.sh $model_base $TRAINING_DATA ${CKPT}-SFT-ITER${iter} scvm 16 1 "None" "None" "None" "$VALUE_FUNCTION/checkpoint-500" prm-prod-vote-sum prm-500

    # bash scripts/eval/eval_models_per_dataset_med_mcts.sh $SFT_MODEL_PATH $TRAINING_DATA $SFT_CKPT 1 $CONFIG_PATH "$model_base" $VALUE_FUNCTION tot-prm-gmean-vote-sum 8 bfs "--tot_expand_way bfs"
    
    # LoRA training evaluation
    # bash scripts/eval/eval_models_per_dataset_med_mcts.sh $model_base $TRAINING_DATA $SFT_CKPT 4 $CONFIG_PATH "$model_base" $VALUE_FUNCTION prm-gmean-vote-sum 1 allprm
    # bash scripts/eval/eval_models_per_dataset_med_mcts.sh $model_base $TRAINING_DATA $SFT_CKPT 2 Evol_Instruct/config/trial5_2.json "$model_base" $VALUE_FUNCTION prm-gmean-vote-sum 1 allprm_32
    # bash scripts/eval/eval_models_per_dataset_med_mcts.sh $model_base $TRAINING_DATA $SFT_CKPT 4 $CONFIG_PATH "$model_base" $VALUE_FUNCTION tot-prm-gmean-vote-sum 4 bfs-2 "--tot_expand_way bfs"
    # bash scripts/eval/eval_models_per_dataset_med_mcts.sh $model_base $TRAINING_DATA $SFT_CKPT 2 $CONFIG_PATH "$model_base" $VALUE_FUNCTION tot-prm-gmean-vote-sum 3 dfs "--tot_expand_way dfs"


    # full fine-tuning evaluation
    # bash scripts/eval/eval_models_per_dataset_med_mcts.sh $SFT_MODEL_PATH $TRAINING_DATA $SFT_CKPT 4 $CONFIG_PATH "$model_base" $VALUE_FUNCTION prm-gmean-vote-sum 1 allprm
    # bash scripts/eval/eval_models_per_dataset_med_mcts.sh $SFT_MODEL_PATH $TRAINING_DATA $SFT_CKPT 2 $CONFIG_PATH "$model_base" $VALUE_FUNCTION tot-prm-gmean-vote-sum 3 dfs "--tot_expand_way dfs"
    
    

done