#!/bin/bash


#SBATCH -J dpo_llama
#SBATCH --partition=partition
#SBATCH -N1
#SBATCH --quotatype=reserved
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=16G  
#SBATCH --time=5-00:00:00
###SBATCH --kill-on-bad-exit=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES


echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号
export LOGLEVEL=INFO
# export NCCL_SOCKET_IFNAME="eth0"
MASTER_PORT=$((RANDOM % 1001 + 20000))
export NCCL_DEBUG=ERROR


DATA_PATH="$1"
MODEL_PATH="$2"
OUTPUT_PATH="$3"
previous_lora_path="$4"
other_params="$5"
# --learn_advantage True/False 
if [ -z "$previous_lora_path" ]; then
    previous_lora_path="None"
fi

echo $previous_lora_path

if [ $previous_lora_path != "None" ]; then 
    deep_speed_path=scripts/zero2.json
else
    deep_speed_path=scripts/zero3.json
fi



argv=()
read -ra argv <<< "$other_params"

if [[ "$other_params" != *"--num_train_epochs"* ]]; then 
    argv+=("--num_train_epochs" "1")
fi

if [[ "$other_params" != *"--learning_rate"* ]]; then 
    argv+=("--learning_rate" "5e-6")
fi

echo "argv: ${argv[@]}"
echo "${other_params}"
echo ""


srun --jobid $SLURM_JOBID python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_backend c10d \
    --rdzv_id $MASTER_PORT  \
    --node_rank $SLURM_PROCID \
     Evol_Instruct/training/dpo_train.py \
    --data_path $DATA_PATH \
    --model_name_or_path $MODEL_PATH \
    --deepspeed $deep_speed_path \
    --tuned_lora_path $previous_lora_path \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --save_steps 1024 \
    --save_total_limit 1 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --output_dir $OUTPUT_PATH \
    --no_remove_unused_columns \
    --torch_dtype bfloat16 \
    --bf16 True \
    --max_length 8192 \
    --max_prompt_length 3072 \
    --gradient_checkpointing True \
    --lora_r 16 \
    --lora_alpha 32 \
    --dataset_num_proc 16 \
    --report_to wandb \
    --lora_target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    ${argv[@]}