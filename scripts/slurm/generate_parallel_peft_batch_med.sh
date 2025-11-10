#!/bin/bash
#SBATCH -J eval_med_chunk
#SBATCH --partition=partition
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1    
#SBATCH --mem=64G  
#SBATCH --time=5-00:00:00
###SBATCH --kill-on-bad-exit=1



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)


NNODES=$SLURM_NNODES

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

TASK_PATH="$1"
MODEL_BASE="$2"
MODEL_PATH="$3"
CKPT="$4" # 使用实际的检查点名称替换CHECKPOINT_NAME
LOGS_BASE_PATH="$5"
DATASET="$6"
SAMPLING_STRATEGY="$7"
SAMPLING_NUMBER="$8"
num_chunks=${9}
chunk_idx=${10}

custom_name=${11}
DATA_PATH=${TASK_PATH}/medical_test


CUDA_LAUNCH_BLOCKING=1



bs=8



dir_path=${LOGS_BASE_PATH}/${CKPT}/${DATASET}
mkdir -p ${dir_path}

if [[ $SAMPLING_STRATEGY == "sc" ]]; then 
    dir_path=${dir_path}/sc-${SAMPLING_NUMBER}
else
    dir_path=${dir_path}/greedy
fi

if [ ! -z "$custom_name" ]; then
    dir_path=${dir_path}-${custom_name}
fi

mkdir -p ${dir_path}



if [[ $SAMPLING_STRATEGY == "greedy" ]]; then
    temperature=0
else 
    temperature=1
fi

argv=()

if [[ "${MODEL_BASE}" != "None" ]]; then 
    argv+=("--model-base" "${MODEL_BASE}")
fi

echo "Processing ${DATASET}"
srun --output=${dir_path}/infer-${num_chunks}-${chunk_idx}.log  python -m Evol_Instruct.evaluation.model_diverse_gen_batch \
    --model-path ${MODEL_PATH} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${dir_path}/infer-${num_chunks}-${chunk_idx}.jsonl \
    --temperature $temperature \
    --batch-size $bs \
    --sampling_numbers $SAMPLING_NUMBER \
    --sampling_strategy $SAMPLING_STRATEGY \
    --num-chunks $num_chunks \
    --chunk-idx $chunk_idx \
    --resume \
    ${argv[@]}



