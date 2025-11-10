
TASK_PATH=/directory/of/test/set

domains=(MedQA drugdose ddxplus seer pubmedqa MedMCQA "bioasq" "med_mmlu" "biomrc" "pubhealth" "healthfact")

 
MODEL_BASE="$1"



MODEL_NAME="$2"
LOGS_BASE_PATH=./logs/base/${MODEL_NAME}


SAMPLING_STRATEGY="$3"
SAMPLING_NUMBER="$4"
NUM_CHUNKS="$5"
MODEL_PATH="$6"



for domain in "${domains[@]}"; do
    DATASET=$domain
    dir_path=${LOGS_BASE_PATH}/${DATASET}
    if [[ $SAMPLING_STRATEGY == "sc" ]]; then 
        dir_path=${dir_path}/sc-${SAMPLING_NUMBER}
    else
        dir_path=${dir_path}/greedy
    fi

    OUTPUT_PATH=${dir_path}

    MERGED_FILE="$OUTPUT_PATH/merge.jsonl"
    
    echo "$MERGED_FILE"
    if [ ! -f "$MERGED_FILE" ]; then
        job_ids=""
        echo "Begin to chunk evaluation"
        # echo $MODEL_BASE
        for idx in $(seq 0 $((NUM_CHUNKS-1))); do
            job_id=$(sbatch scripts/slurm/generate_parallel_peft_batch_med_base.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $LOGS_BASE_PATH $domain $SAMPLING_STRATEGY $SAMPLING_NUMBER $NUM_CHUNKS $idx $value_function $infer_rule | awk '{print $4}')
            echo "Submitted batch job ${job_id}"
            job_ids="${job_ids}:${job_id}"
            sleep 2
        done
        job_ids=${job_ids#:}
        sbatch --dependency=afterok:${job_ids} scripts/slurm/eval_parallel_peft_batch_med.sh $dir_path $NUM_CHUNKS
    else
        if [ ! -f "$OUTPUT_PATH/eval.log" ]; then
            sbatch scripts/slurm/eval_parallel_peft_batch_med.sh $dir_path $NUM_CHUNKS
        fi
    fi
    

done
