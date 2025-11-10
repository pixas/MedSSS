#!/bin/bash
#SBATCH -J eval_med_chunk
#SBATCH --partition=partition
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1


TASK_PATH="$1"
MODEL_BASE="$2"
MODEL_PATH="$3"
LOGS_BASE_PATH="$4"
DATASET="$5"
SAMPLING_STRATEGY="$6"
SAMPLING_NUMBER="$7"
num_chunks=${8}
chunk_idx=${9}
value_function=${10:-"None"}
infer_rule=${11:-"None"}

DATA_PATH=${TASK_PATH}/medical_test


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
if [[ $SAMPLING_STRATEGY == 'scvm' ]]; then 
    bs=2
fi

bash ~/add_oss.sh

dir_path=${LOGS_BASE_PATH}/${DATASET}
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
mkdir -p ${dir_path}



if [[ $SAMPLING_STRATEGY == "greedy" ]]; then
    temperature=0
else 
    temperature=0.7
fi

echo "Processing ${DATASET}"

srun  --output=${dir_path}/infer-${num_chunks}-${chunk_idx}.log  python -m Evol_Instruct.evaluation.model_diverse_gen_batch \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${dir_path}/infer-${num_chunks}-${chunk_idx}.jsonl \
    --temperature $temperature \
    --use-logit-bias \
    --batch-size $bs \
    --sampling_numbers $SAMPLING_NUMBER \
    --sampling_strategy $SAMPLING_STRATEGY \
    --num-chunks $num_chunks \
    --chunk-idx $chunk_idx \
    --value_function $value_function \
    --infer_rule $infer_rule \
    --resume



