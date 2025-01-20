#!/bin/bash
#SBATCH -J eval_mcts
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=96:00:00
###SBATCH --kill-on-bad-exit=1


TASK_PATH="$1"
MODEL_BASE="$2"
MODEL_PATH="$3"
CKPT="$4" # 使用实际的检查点名称替换CHECKPOINT_NAME
LOGS_BASE_PATH="$5"
DATASET="$6"
num_chunks=${7}
chunk_idx=${8}
CONFIG_PATH=${9}
VALUE_MODEL_BASE=${10}
VALUE_FUNCTION=${11}
infer_rule=${12}
SAMPLING_NUMBERS=${13}
custom_name=${14}
other_params=${15}
DATA_PATH=${TASK_PATH}/medical_test



if [[ $DATASET == "tydiqa_cot" ]]; then
    bs=2
elif [[ $DATASET == *"CBLUE"* ]]; then
    bs=4
else
    bs=16
fi
bash ~/add_oss.sh
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

dir_path=${LOGS_BASE_PATH}/${CKPT}/${DATASET}
mkdir -p ${dir_path}


config_name=$(basename $CONFIG_PATH)
config="${config_name%.*}"
dir_path=${dir_path}/mcts-${config}-${infer_rule}-${SAMPLING_NUMBERS}

if [ ! -z "$custom_name" ]; then
    dir_path=${dir_path}-${custom_name}
fi

mkdir -p ${dir_path}


argv=()
read -ra argv <<< "$other_params"

if [[ "${MODEL_BASE}" != "None" ]]; then 
    argv+=("--model-base" "${MODEL_BASE}")
fi


temperature=1


echo "Processing ${DATASET}"
srun --output=${dir_path}/infer-${num_chunks}-${chunk_idx}.log  python -m Evol_Instruct.evaluation.model_diverse_gen_batch \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${dir_path}/infer-${num_chunks}-${chunk_idx}.jsonl \
    --temperature $temperature \
    --use-logit-bias \
    --batch-size 1 \
    --sampling_strategy mcts \
    --sampling_numbers $SAMPLING_NUMBERS \
    --num-chunks $num_chunks \
    --chunk-idx $chunk_idx \
    --mcts_config $CONFIG_PATH \
    --value_model_base $VALUE_MODEL_BASE \
    --value_function $VALUE_FUNCTION \
    --infer_rule $infer_rule \
    --resume \
    ${argv[@]}



