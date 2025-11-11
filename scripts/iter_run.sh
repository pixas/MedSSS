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

# after inference, open show_results.ipynb, and place log_dir and save_name with the proper actual value
# if no things have been changed, the log_dir should be logs/$DPO_TRAINING_DATA/$DPO_CKPT
# the save_name should be sc-32/hard-dpo-unifybce-lb.jsonl
# the show_results.ipynb can compare models with different selection strategy, including best-of-n, P-VS and SC

############################## END OF ITER 2 ##############################
