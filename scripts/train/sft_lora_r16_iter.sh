BASE_PATH=<parent_dir_of_checkpoints>
TASK_PATH=<training_sets_path>


TRAINING_DATA="$1"

MODEL_BASE="$2"

CKPT="$3"

previous_lora_path="$4"
other_params="$5"
SAVE_PATH=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}

MORA_TYPE=6
LORAPLUS_LR_RATIO=2

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}


sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/slurm/sft_lora_r16a32_iter.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $CKPT "$previous_lora_path" "$other_params" & sleep 1
# fi
