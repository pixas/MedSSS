
model_type=prm

# the dirname of data_path

TASK_PATH=/directory/of/test/set

domains=(MedQA drugdose ddxplus seer pubmedqa MedMCQA "bioasq" "med_mmlu" "biomrc" "pubhealth" "healthfact")



# MODEL_BASE=/mnt/hwfile/medai/jiangshuyang/checkpoints/ming-moe-clinical-v2-qwen1.5-1.8b-molora-r16a32_share_expert_2_mergelora

# TRAINING_DATA=ming-moe-clinical-v2
TRAINING_DATA="$1"
LOGS_BASE_PATH=./logs/${TRAINING_DATA}

# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_2_fix
# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_4_fix
CKPT="$2"
SAMPLING_STRATEGY="$3"
SAMPLING_NUMBER="$4"
NUM_CHUNKS="$5"

value_model_base="${6}"
value_function="${7}"
custom_name="${8}"
input_custom_name="${9}"
infer_rule=prm-max




if [ -z "$input_custom_name" ]; then
    input_custom_name="None"
fi


for domain in "${domains[@]}"; do
    DATASET=$domain
    dir_path=${LOGS_BASE_PATH}/${CKPT}/${DATASET}
    
    dir_path=${dir_path}/sc-${SAMPLING_NUMBER}
    if [[ "$input_custom_name" != "None" ]]; then 
        dir_path=${dir_path}-$input_custom_name
    fi
    
    OUTPUT_PATH=${dir_path}

    MERGED_FILE="$OUTPUT_PATH/merge.jsonl"

    echo "$MERGED_FILE"
    save_path=${dir_path}/${custom_name}.jsonl
    log_file=${dir_path}/${custom_name}.log
    if  [ ! -f "$save_path" ]; then 
        job_ids=""
        if [[ $NUM_CHUNKS -eq 1 ]]; then
            sbatch -o $log_file scripts/slurm/vm_rescore.sh $MERGED_FILE $value_model_base "$value_function" "--model_type ${model_type} --infer_rule ${infer_rule} --save_path ${save_path}"
        else
            echo "begin to chunk value function"
            for idx in $(seq 0 $((NUM_CHUNKS-1))); do
                cur_save_path=${dir_path}/${custom_name}-${NUM_CHUNKS}-${idx}.jsonl
                cur_log_file=${dir_path}/${custom_name}-${NUM_CHUNKS}-${idx}.log
                job_id=$(sbatch -o $cur_log_file scripts/slurm/vm_rescore.sh $MERGED_FILE $value_model_base "$value_function" "--model_type ${model_type} --infer_rule ${infer_rule} --save_path ${cur_save_path} --num_chunks ${NUM_CHUNKS} --chunk_idx ${idx}" | awk '{print $4}')
                echo "Submitted batch job ${job_id}"
                job_ids="${job_ids}:${job_id}"
                sleep 2 
            done 
            job_ids=${job_ids#:}
            sbatch --dependency=afterok:${job_ids} scripts/slurm/merge_prm_files.sh $dir_path $NUM_CHUNKS $custom_name
        fi
    fi

    sleep 5

done

