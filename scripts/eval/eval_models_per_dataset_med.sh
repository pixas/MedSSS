BASE_PATH=<parent_dir_of_checkpoints>
TASK_PATH=<evaluation_sets_path>

domains=("MedQA_cot" "medsins_task16" "medsins_task29" "medsins_task130" "medsins_task131" "MedMCQA_cot" "med_mmlu_cot" "pubmedqa_c_cot" "bioasq" "pubhealth" "biomrc")

 
MODEL_BASE="$1"



TRAINING_DATA="$2"
LOGS_BASE_PATH=./logs/${TRAINING_DATA}

CKPT="$3"
SAMPLING_STRATEGY="$4"
SAMPLING_NUMBER="$5"
NUM_CHUNKS="$6"
# OTHER_MODEL_PATH="$7"
dpo_model_path="$7"
dpo_from="$8"
dpo_select_method="${9}"
value_model_base="${10}"
value_function="${11}"
infer_rule="${12}"
custom_name="${13}"

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
    OUTPUT_PATH=${dir_path}

    MERGED_FILE="$OUTPUT_PATH/merge.jsonl"

    echo "$MERGED_FILE"
    if [ ! -f "$MERGED_FILE" ]; then
        job_ids=""
        echo "Begin to chunk evaluation"
        for idx in $(seq 0 $((NUM_CHUNKS-1))); do
        # for idx in $(seq 0 0); do
            job_id=$(sbatch scripts/slurm/generate_parallel_peft_batch_med.sh $TASK_PATH $MODEL_BASE "$MODEL_PATH" $CKPT $LOGS_BASE_PATH $domain $SAMPLING_STRATEGY $SAMPLING_NUMBER $NUM_CHUNKS $idx $dpo_model_path $dpo_from $dpo_select_method "${value_model_base}" "$value_function" $infer_rule $custom_name | awk '{print $4}')
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
