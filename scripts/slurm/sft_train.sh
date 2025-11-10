#!/bin/bash

#SBATCH -J sft_llama
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1  
#SBATCH --mem-per-cpu=4G  


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

export LOGLEVEL=INFO
# export NCCL_SOCKET_IFNAME="eth0"
MASTER_PORT=$((RANDOM % 1001 + 20000))
export NCCL_DEBUG=ERROR


TASK_PATH="$1"
TRAINING_DATA="$2"
MODEL_BASE="$3"
SAVE_PATH="$4"
previous_lora_path="$5"
other_params="$6"
# --learn_advantage True/False 
if [ -z "$previous_lora_path" ]; then
    previous_lora_path="None"
fi


DATA_PATH=${TASK_PATH}/${TRAINING_DATA}.json
argv=()
read -ra argv <<< "$other_params"

if [[ "$other_params" != *"--num_train_epochs"* ]]; then 
    argv+=("--num_train_epochs" "1")
fi

if [[ "$other_params" != *"--learning_rate"* ]]; then 
    argv+=("--learning_rate" "2e-4")
fi

echo "argv: ${argv[@]}"
echo "${other_params}"
echo ""



TORCH_USE_CUDA_DSA=1
CUDA_LAUNCH_BLOCKING=1
srun --jobid $SLURM_JOBID python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_backend c10d \
    --rdzv_id $MASTER_PORT --standalone \
    --node_rank $SLURM_PROCID \
    Evol_Instruct/training/sft_train.py \
    --lora_target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $MODEL_BASE \
    --data_path $DATA_PATH \
    --torch_dtype bfloat16 \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --tuned_lora_path $previous_lora_path \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_strategy "no" \
    --eval_steps 100 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --max_seq_length 8192 \
    --gradient_checkpointing True \
    --dataset_num_proc 8 \
    --report_to wandb \
    ${argv[@]}

