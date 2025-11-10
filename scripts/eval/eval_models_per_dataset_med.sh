BASE_PATH=/directory/of/
TASK_PATH=/directory/of/test/set

domains=(MedQA drugdose ddxplus seer pubmedqa MedMCQA "bioasq" "med_mmlu" "biomrc" "pubhealth" "healthfact")


MODEL_BASE="$1"
# MODEL_BASE=/mnt/hwfile/medai/jiangshuyang/checkpoints/ming-moe-clinical-v2-qwen1.5-1.8b-molora-r16a32_share_expert_2_mergelora

# TRAINING_DATA=ming-moe-clinical-v2
TRAINING_DATA="$2"
LOGS_BASE_PATH=./logs/${TRAINING_DATA}

# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_2_fix
# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_4_fix
CKPT="$3"
SAMPLING_STRATEGY="$4"
SAMPLING_NUMBER="$5"
NUM_CHUNKS="$6"
# OTHER_MODEL_PATH="$7"
custom_name="${7}"

if [[ ${CKPT} == *"full"* ]]; then 
    cur_model_path=${MODEL_BASE}
    MODEL_BASE=None
else 
    cur_model_path=${BASE_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
fi
MODEL_PATH="${cur_model_path}"





for domain in "${domains[@]}"; do
    DATASET=$domain
    dir_path=${LOGS_BASE_PATH}/${CKPT}/${DATASET}
    if [[ $SAMPLING_STRATEGY == "sc" ]]; then 
        dir_path=${dir_path}/sc-${SAMPLING_NUMBER}
        dir_path=${dir_path}/greedy
    fi
    if [ ! -z "$custom_name" ]; then
        dir_path=${dir_path}-${custom_name}
    fi
    OUTPUT_PATH=${dir_path}

    MERGED_FILE="$OUTPUT_PATH/merge.jsonl"

    echo "$MERGED_FILE"
    if [[ $domain == "drugdose" || $domain == "pubmedqa" || $domain == "bioasq" ]]; then 
        NUM_CHUNKS=1
    fi
    if [ ! -f "$MERGED_FILE" ]; then
        job_ids=""
        echo "Begin to chunk evaluation"
        for idx in $(seq 0 $((NUM_CHUNKS-1))); do
        # for idx in $(seq 0 0); do
            job_id=$(sbatch scripts/slurm/generate_parallel_peft_batch_med.sh $TASK_PATH $MODEL_BASE "$MODEL_PATH" $CKPT $LOGS_BASE_PATH $domain $SAMPLING_STRATEGY $SAMPLING_NUMBER $NUM_CHUNKS $idx $custom_name ${@:8} | awk '{print $4}')
            echo "Submitted batch job ${job_id}"
            job_ids="${job_ids}:${job_id}"
            sleep 2
        done
        job_ids=${job_ids#:}
        sbatch --dependency=afterok:${job_ids} scripts/slurm/eval_parallel_peft_batch_med.sh $dir_path $NUM_CHUNKS
    else
        if [  -z $value_function ]; then 
            
            sbatch scripts/slurm/eval_parallel_peft_batch_med.sh $dir_path $NUM_CHUNKS
        else 
            bash scripts/rescore.sh $MERGED_FILE "${value_model_base}" "$value_function" $infer_rule $custom_name
        fi
        # if [ ! -f "$OUTPUT_PATH/eval.log" ]; then
        #     sbatch scripts/slurm/eval_parallel_peft_batch_med.sh $dir_path $NUM_CHUNKS
        # fi
    fi
    sleep 5

done
