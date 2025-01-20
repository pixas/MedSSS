MAINTAIN_STRATEGY=all
# SAMPLING_NUMBER=3
SAMPLING_NUMBER=16
model_base="$1"
if [ -z "$model_base" ]; then 
    model_base="llama38b"
fi

BASE_PATH=<checkpoint_parent_dir>
# TASK_PATH=s3://syj_test/datasets/medical_train
# oss_path=${BASE_PATH}/oss/datasets/medical_train
TASK_PATH=${BASE_PATH}/oss/datasets/medical_train

MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
CKPT=llama3.1-8b-r16a32-1epoch


# TRAIN_DATA_NAME=PubmedQA


NUM_CHUNKS=4
METHOD=mcts
BATCH_SIZE=1

CONFIG_NAME="$2"
if [ -z "$CONFIG_NAME" ]; then 
    CONFIG_NAME="trail1"
fi

TRAIN_DATA_NAME="$3"
SAMPLE_NUM=1000

if [ -z "$TRAIN_DATA_NAME" ]; then 
    TRAIN_DATA_NAME="mmed_en_train"
fi

INPUT_DATA=~/datasets/medical_train/${TRAIN_DATA_NAME}.json
CONFIG_PATH=Evol_Instruct/config/${CONFIG_NAME}.json

SELFGEN_LORA_PATH=None 
VALUE_FUNCTION=None
for iter in {1..1}; do
    echo "Begin Iteration: ${iter}"
    ##################################CREATE FILES##################################
    SAVE_PATH=${model_base}_${METHOD}_vllm_${TRAIN_DATA_NAME}_${MAINTAIN_STRATEGY}_${CONFIG_NAME}
    # SAVE_PATH=${model_base}_${METHOD}_vllm_${TRAIN_DATA_NAME}_${MAINTAIN_STRATEGY}_${CONFIG_NAME}_${SAMPLE_NUM}
    # if [[ $SAMPLE_NUM == *"-"* ]]; then
    #     # SAMPLE_NUM=${SAMPLE_NUM//"-"/_}
    #     SAVE_PATH=${model_base}_${METHOD}_vllm_${TRAIN_DATA_NAME}_${MAINTAIN_STRATEGY}_${CONFIG_NAME}_${SAMPLE_NUM}
    # fi
    OUTPUT_PATH=${TASK_PATH}/${SAVE_PATH}/sft_${iter}.jsonl
    # echo $OUTPUT_PATH
    mkdir -p ${TASK_PATH}/${SAVE_PATH}
    
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
    if [ ! -f ${JSON_OUTPUT_PATH} ]; then
        bash scripts/chunk_iter_self_gen_mcts.sh $OUTPUT_PATH $MODEL_PATH $SAMPLE_NUM $INPUT_DATA $LOG_PATH $METHOD $BATCH_SIZE $SAMPLING_NUMBER $MAINTAIN_STRATEGY $SELFGEN_LORA_PATH $SHOTS $CONFIG_PATH $iter $NUM_CHUNKS $VALUE_FUNCTION 
    fi
    # waiting for combine

    while [ ! -f ${JSON_OUTPUT_PATH} ]; do
        echo "Waiting for ${JSON_OUTPUT_PATH} to be generated..."
        sleep 60
    done
    echo "Generating data over"
    sleep 2
    #####################################COMBINE DATA################################
    if [ ! -f "${TASK_PATH}/${SAVE_PATH}/sft_combined_${iter}.json" ]; then
    # if ! aws s3 ls "${TASK_PATH}/${SAVE_PATH}/sft_combined_${iter}.json" > /dev/null 2>&1; then
        srun -p medai_llm --debug python Evol_Instruct/utils/combine_json.py \
            --data_path ${TASK_PATH}/${SAVE_PATH} \
            --save_path ${TASK_PATH}/${SAVE_PATH}/sft_combined_${iter}.json \
            --num ${iter} \
            --prefix sft \
            --only_iter
    fi


    echo "Combining data over"
    #######################################SFT TRAINING AND DPO TRAINING################################
    if [ $iter -eq 1 ]; then
        model_base=$MODEL_PATH
    else
        model_base="${BASE_PATH}/checkpoints/${SAVE_PATH}/sft_combined_1-${CKPT}-SFT-ITER1-merged"
        if [ ! -f "$model_base/config.json" ]; then
            srun -p medai_llm --debug python Evol_Instruct/utils/merge_lora_weights.py \
                --model_base $MODEL_PATH \
                --model_path "${BASE_PATH}/checkpoints/${SAVE_PATH}/sft_combined_1-${CKPT}-SFT-ITER1" \
                --save_path $model_base
            echo "Merge Iter1's Lora weights to make $model_base new base model"
        fi
    fi 
    TRAINING_DATA=${SAVE_PATH}/sft_${iter}
    # combine over, begin sft training
    SFT_CKPT=${CKPT}-SFT-ITER${iter}
    # SFT_CKPT=${CKPT}-SFT-full-ITER${iter}
    SFT_MODEL_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${SFT_CKPT}
    echo "${SFT_MODEL_PATH}"


    VALUE_TRAINING_DATA=${SAVE_PATH}/sft_${iter}
    

    VALUE_CKPT=${CKPT}-VALUE-prm_trainall_r64_softtrain_basepolicy-ITER${iter}

    VALUE_MODEL_PATH=${BASE_PATH}/checkpoints/${VALUE_TRAINING_DATA}-${VALUE_CKPT}

    LOGS_BASE_PATH=./logs/${VALUE_TRAINING_DATA}/${VALUE_CKPT}
    mkdir -p ${LOGS_BASE_PATH}
    LOGS_FILE=${LOGS_BASE_PATH}/train.log


    if [ ! -f "${VALUE_MODEL_PATH}/adapter_config.json" ]; then
        while [ ! -f "${SFT_MODEL_PATH}/tokenizer_config.json" ]; do
            echo "Waiting for ${SFT_MODEL_PATH}/tokenizer_config.json to be generated..."
            sleep 60 
        done
        sbatch -o $LOGS_FILE scripts/slurm/prm_train_lora_r16a32.sh ${TASK_PATH}/${VALUE_TRAINING_DATA}.json $model_base ${VALUE_MODEL_PATH} "${SFT_MODEL_PATH}" "--learning_rate 2e-4 --num_train_epochs 1 --test_split_ratio 0.01 --train_pair_per_instance -1 --use_peft --filter_invalid True --use_soft_training True --positive_thr 0.5"
    fi
    # break
    echo "Policy and Value training over"
    # break
    #######################################UPDATE SELFGEN MODEL#########################################
    bash /mnt/petrelfs/jiangshuyang.p/add_oss.sh
    while [ ! -f "${SFT_MODEL_PATH}/tokenizer_config.json" ]; do
    # while ! aws s3 ls "$SFT_CONFIG_S3_PATH" > /dev/null 2>&1; do
        echo "Waiting for ${SFT_MODEL_PATH}/tokenizer_config.json to be generated..."
        sleep 60
    done
    
    SELFGEN_LORA_PATH=${SFT_MODEL_PATH}
    VALUE_FUNCTION=${VALUE_MODEL_PATH}
    echo "Inference with Greedy"
    bash scripts/eval/eval_models_per_dataset_med.sh $model_base $TRAINING_DATA ${SFT_CKPT} greedy 1 1 
    # break
    echo "Inference with SC"
    bash scripts/eval/eval_models_per_dataset_med.sh $model_base $TRAINING_DATA ${SFT_CKPT} sc 16 1
    
    while [ ! -f "${VALUE_MODEL_PATH}/adapter_config.json" ]; do
    
        echo "Waiting for ${VALUE_MODEL_PATH}/adapter_config.json to be generated..."
        sleep 60
    done
    echo "Inference with PRM"

    bash scripts/eval/eval_models_per_dataset_med.sh $model_base $TRAINING_DATA ${CKPT}-SFT-ITER${iter} scvm 16 1 "None" "None" "None" "$model_base" "$SFT_MODEL_PATH $VALUE_FUNCTION" prm-min-vote-sum prm_softtrain_basepolicy


done