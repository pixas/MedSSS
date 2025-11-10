#!/bin/bash


#SBATCH -J vmscore
#SBATCH --partition=partition
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=6G  
#SBATCH --time=5-00:00:00
###SBATCH --kill-on-bad-exit=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=1
NNODES=$SLURM_NNODES
echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号


DATA_PATH="$1"
MODEL_BASE="$2"
MODEL_PATH="$3"

srun python Evol_Instruct/utils/reevaluate.py \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --model_base "$MODEL_BASE" \
    ${@:4}