BASE_PATH=<parent_dir_of_checkpoints>
TASK_PATH=evaluation_data

domains=("MedQA_cot" "medsins_task16" "medsins_task29" "medsins_task130" "medsins_task131" "MedMCQA_cot" "med_mmlu_cot" "pubmedqa_c_cot" "bioasq" "pubhealth" "biomrc")

 
MODEL_BASE="$1"


TRAINING_DATA="$2"
LOGS_BASE_PATH=./logs/${TRAINING_DATA}


CKPT="$3"
NUM_CHUNKS="$4"
CONFIG_PATH=$5
value_model_base="$6"
VALUE_FUNCTION=$7
infer_rule=$8
SAMPLING_NUMBERS=$9
custom_name=${10}
other_params=${11}

if [ -z "$SAMPLING_NUMBERS" ]; then
    SAMPLING_NUMBERS=1
fi

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

    config_name=$(basename $CONFIG_PATH)
    config="${config_name%.*}"
    dir_path=${dir_path}/mcts-${config}-${infer_rule}-${SAMPLING_NUMBERS}
    # fi
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
            job_id=$(sbatch scripts/slurm/generate_parallel_peft_batch_med_mcts.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain $NUM_CHUNKS $idx $CONFIG_PATH "${value_model_base}" "$VALUE_FUNCTION" $infer_rule $SAMPLING_NUMBERS $custom_name "${other_params}" | awk '{print $4}')
            echo "Submitted batch job ${job_id}"
            job_ids="${job_ids}:${job_id}"
            sleep 2
            # break
        done
        job_ids=${job_ids#:}
        sbatch --dependency=afterok:${job_ids} scripts/slurm/eval_parallel_peft_batch_med.sh $dir_path $NUM_CHUNKS
    else
        if [ ! -f "$OUTPUT_PATH/eval.log" ]; then
            sbatch scripts/slurm/eval_parallel_peft_batch_med.sh $dir_path $NUM_CHUNKS
        fi
    fi
    sleep 10

done



