#!/bin/bash


#SBATCH -J evol
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G  
#SBATCH --time=72:00:00
#SBATCH -x SH-IDC1-10-140-0-221
###SBATCH --kill-on-bad-exit=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)


NNODES=$SLURM_NNODES

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号


OUTPUT_PATH="$1"
MODEL_PATH="$2"
SAMPLE_NUM="$3"
INPUT_DATA="$4"
LOG_PATH="$5"
METHOD="$6"
BATCH_SIZE="$7"
SAMPLING_NUMBER="$8"
MAINTAIN_STRATEGY="$9"
SELFGEN_LORA_PATH=${10}
SHOTS=${11}
CONFIG_PATH=${12}
iter=${13}
NUM_CHUNKS=${14:-1}
CHUNK_IDX=${15:-0}
VALUE_FUNCTION=${16:-""}


echo "iter_self_gen_mcts_llama.sh ${CONFIG_PATH}"
srun -o ${LOG_PATH} python Evol_Instruct/self_gen_mcts.py \
    --model_path $MODEL_PATH \
    --data_path $INPUT_DATA \
    --output_path $OUTPUT_PATH \
    --sample_num $SAMPLE_NUM  \
    --lora_path $SELFGEN_LORA_PATH \
    --num_chunks $NUM_CHUNKS \
    --chunk_idx $CHUNK_IDX \
    --intermediate_select all \
    --value_function $VALUE_FUNCTION \
    --iter $iter \
    --config $CONFIG_PATH \
    --intermediate_select $MAINTAIN_STRATEGY \
    --resume