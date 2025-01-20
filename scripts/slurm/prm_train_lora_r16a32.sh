#!/bin/bash


#SBATCH -J prm
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=6G  
#SBATCH --time=5-00:00:00
#SBATCH -x SH-IDC1-10-140-0-221
###SBATCH --kill-on-bad-exit=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES


echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

# export NCCL_SOCKET_IFNAME="eth0"
MASTER_PORT=$((RANDOM % 1001 + 20000))
# MASTER_PORT=$((20000 + SLURM_PROCID))

export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME=eth0        


DATA_PATH="$1"
MODEL_PATH="$2"
OUTPUT_PATH="$3"
previous_lora_path="$4"
other_params="$5"
# --learning_rate --score_lr --learn_by_stage 


if [ -z "$previous_lora_path" ]; then
    previous_lora_path="None"
    deep_speed_path=scripts/zero3.json
else
    deep_speed_path=scripts/zero2.json
fi

argv=()
read -ra argv <<< "$other_params"

if [[ "$other_params" != *"--num_train_epochs"* ]]; then 
    argv+=("--num_train_epochs" "1")
fi

if [[ "$other_params" != *"--learning_rate"* ]]; then 
    argv+=("--learning_rate" "2e-4")
fi

if [[ "$other_params" != *"--use_peft"* ]]; then 
    batch_size=2
else
    batch_size=4
fi

bash /mnt/petrelfs/jiangshuyang.p/add_oss.sh

srun --jobid $SLURM_JOBID python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_backend c10d \
    --rdzv_id $MASTER_PORT --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    Evol_Instruct/training/prm_train.py \
    --data_path $DATA_PATH \
    --deepspeed $deep_speed_path \
    --lora_task_type "TOKEN_CLS" \
    --model_name_or_path $MODEL_PATH \
    --tuned_lora_path $previous_lora_path \
    --weight_decay 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --output_dir $OUTPUT_PATH \
    --lr_scheduler_type "cosine" \
    --max_length 8192 \
    --warmup_ratio 0.03 \
    --bf16 True \
    --gradient_checkpointing True \
    --dataset_num_proc 16 \
    --report_to wandb \
    --test_split_ratio 0.1 \
    --lora_r 64 --lora_alpha 128 \
    ${argv[@]}