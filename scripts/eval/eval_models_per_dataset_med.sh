BASE_PATH=/mnt/petrelfs/jiangshuyang/
TASK_PATH=/mnt/petrelfs/jiangshuyang/datasets
# domains=(MedQA_cot pubmedqa_c_cot "MedMCQA_cot_500" "med_mmlu_cot" "medsins_task16" "biomrc_500" pubhealth medsins_task130_500 medsins_task131_500 MedMCQA_cot_500 bioasq)
# domains=("MedMCQA_cot_500" "medsins_task131_500" "medsins_task130_500" "medsins_task29")
# domains=("medsins_task130" "medsins_task131" "MedMCQA_cot" "biomrc")
# domains=("MedQA_cot" "medsins_task16" "medsins_task29" "medsins_task130" "medsins_task131" "MedMCQA_cot" "med_mmlu_cot" "pubmedqa_c_cot" "bioasq" "pubhealth" "biomrc")
domains=("medsins_task130" "medsins_task131" "biomrc" "med_mmlu_cot" "medsins_task16" "pubhealth" "MedQA_cot" "pubmedqa_c_cot" "bioasq" "medsins_task29" "MedMCQA_cot")
# domains=("medsins_task130_500" "medsins_task131_500" "MedMCQA_cot_500")
# domains=("MedMCQA_cot_500" "med_mmlu_cot" "medsins_task16" "biomrc_500" "pubhealth" "medsins_task130_500" "medsins_task131_500" "bioasq")
# domains=("MedMCQA_cot_500" "medsins_task131_500" "medsins_task130_500")
# domains=("medsins_task130" "medsins_task131" "MedMCQA_cot" "biomrc" "medsins_task16" "pubhealth")
# domains=("MedQA_cot" "pubmedqa_c_cot" "bioasq" "med_mmlu_cot")
# domains=(bioasq)
domains=(rds)
# domains=(medsins_task29 pubhealth medsins_task16 biomrc medsins_task130 medsins_task131)
# domains=("MedQA_cot" "MedMCQA_cot" "med_mmlu_cot" "pubmedqa_c_cot" "bioasq" "biomrc" "pubhealth")
# domains=("participant_extraction" "drug_dose_extraction" "intervention_extraction" "outcome_extraction")
# domains=(medsins_task16 med_mmlu_cot medsins_task29 MedQA_cot pubmedqa_c_cot bioasq biomrc_500)

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
# if [ "${OTHER_MODEL_PATH}" == "None" ]; then
# else 
#     MODEL_PATH="${OTHER_MODEL_PATH} ${cur_model_path}"
# fi
# echo "${MODEL_PATH}"

# bash ~/add_oss.sh

# while [ ! -f "${MODEL_PATH}/adapter_config.json" ]; do
#     echo "Waiting for ${MODEL_PATH}/adapter_config.json to appear..."
#     sleep 60
# done





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
    if [[ $domain == "medsins_task29" || $domain == "pubmedqa_c_cot" || $domain == "bioasq" ]]; then 
        NUM_CHUNKS=1
    fi
    if [ ! -f "$MERGED_FILE" ]; then
        job_ids=""
        echo "Begin to chunk evaluation"
        for idx in $(seq 0 $((NUM_CHUNKS-1))); do
        # for idx in $(seq 0 0); do
            job_id=$(sbatch scripts/slurm/generate_parallel_peft_batch_med.sh $TASK_PATH $MODEL_BASE "$MODEL_PATH" $CKPT $LOGS_BASE_PATH $domain $SAMPLING_STRATEGY $SAMPLING_NUMBER $NUM_CHUNKS $idx $dpo_model_path $dpo_from $dpo_select_method "${value_model_base}" "$value_function" $infer_rule $custom_name ${@:14} | awk '{print $4}')
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
