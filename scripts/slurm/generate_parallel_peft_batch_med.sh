#!/bin/bash
#SBATCH -J eval_med_chunk
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=5-00:00:00
###SBATCH --kill-on-bad-exit=1


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
dpo_model_path=${11:-"None"}
dpo_from=${12:-"sft"}
dpo_select_method=${13:-"max"}
value_model_base=${14:-"None"}
value_function=${15:-"None"}
infer_rule=${16:-"None"}
custom_name=${17}
DATA_PATH=${TASK_PATH}/medical_test

srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号
CUDA_LAUNCH_BLOCKING=1
if [[ $DATASET == "tydiqa_cot" ]]; then
    bs=2
elif [[ $DATASET == *"CBLUE"* ]]; then
    bs=4
else
    bs=16
fi
if [[ $SAMPLING_NUMBER == 32 ]]; then
    
    bs=8
fi
if [[  $DATASET == *"medsins"* ]]; then 
    bs=1
fi

if [[ $SAMPLING_STRATEGY == 'scvm' ]]; then 
    bs=2
fi

bash ~/add_oss.sh

dir_path=${LOGS_BASE_PATH}/${CKPT}/${DATASET}
mkdir -p ${dir_path}

if [[ $SAMPLING_STRATEGY == "sc" ]]; then 
    dir_path=${dir_path}/sc-${SAMPLING_NUMBER}
elif [[ $SAMPLING_STRATEGY == "scvm" ]]; then 
    dir_path=${dir_path}/scvm-${infer_rule}-${SAMPLING_NUMBER}
elif [[ $SAMPLING_STRATEGY == "dpo_judge" ]]; then 
    dir_path=${dir_path}/dpo_judge-${dpo_from}-${dpo_select_method}-${SAMPLING_NUMBER}
elif [[ $SAMPLING_STRATEGY == "dpo_greedy" ]]; then 
    dir_path=${dir_path}/dpo-greedy
elif [[ $SAMPLING_STRATEGY == "dpo_sc" ]]; then 
    dir_path=${dir_path}/dpo-sc-${SAMPLING_NUMBER}
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
    --use-logit-bias \
    --batch-size $bs \
    --max-new-tokens 8192 \
    --value_model_base "${value_model_base}" \
    --sampling_numbers $SAMPLING_NUMBER \
    --sampling_strategy $SAMPLING_STRATEGY \
    --dpo_model_path $dpo_model_path \
    --dpo_select_method $dpo_select_method \
    --num-chunks $num_chunks \
    --chunk-idx $chunk_idx \
    --value_function $value_function \
    --infer_rule $infer_rule \
    --resume \
    ${argv[@]}



