OUTPUT_PATH="$1"
MODEL_PATH="$2"
SAMPLE_NUM="$3"
INPUT_DATA="$4"
LOG_PATH="$5"
METHOD="$6"
BATCH_SIZE="$7"
SAMPLING_NUMBER="$8"
MAINTAIN_STRATEGY="$9"
SELFGEN_LORA_PATH=${10}
SHOTS=${11}
CONFIG_PATH=${12}
iter=${13}
NUM_CHUNKS=${14:-1}
VALUE_FUNCTION=${15:-None}

# echo "All parameters: $1, $2, $3, $4, $5, $6, $7, $8, $9, ${10}, ${11}, ${12}, ${13}, ${14}"
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
filename=$(basename "$OUTPUT_PATH")
FILE_NAME="${filename%.*}"

LOG_DIR=$(dirname "$LOG_PATH")
filename=$(basename "$LOG_PATH")
LOG_NAME="${filename%.*}"

if [ $METHOD == "sc" ]; then
    script_path=scripts/slurm/iter_self_gen_sc_llama.sh
    critic_prefix=dpo
elif [ $METHOD == "mcts" ]; then
    script_path=scripts/slurm/iter_self_gen_mcts_llama.sh
    critic_prefix=value
else
    echo "Invalid method: $METHOD"
    exit 1
fi



job_ids=""

for IDX in  $(seq 0 $((NUM_CHUNKS-1))); do
    CHUNK_OUTPUT_PATH=${OUTPUT_DIR}/${FILE_NAME}_${NUM_CHUNKS}_${IDX}.jsonl
    CHUNK_LOG_FILE=$LOG_DIR/${LOG_NAME}_${NUM_CHUNKS}_${IDX}.log

    job_id=$(sbatch $script_path $CHUNK_OUTPUT_PATH $MODEL_PATH $SAMPLE_NUM $INPUT_DATA $CHUNK_LOG_FILE $METHOD $BATCH_SIZE $SAMPLING_NUMBER $MAINTAIN_STRATEGY $SELFGEN_LORA_PATH $SHOTS $CONFIG_PATH $iter $NUM_CHUNKS $IDX  "$VALUE_FUNCTION"  | awk '{print $4}')
    echo "Submitted batch job ${job_id}"
    job_ids="${job_ids}:${job_id}"
    sleep 2

done 

job_ids=${job_ids#:}
echo ${job_ids}
srun -p medai_llm --debug --dependency=afterok:${job_ids} python Evol_Instruct/utils/combine_json.py \
    --data_path $OUTPUT_DIR \
    --save_path ${OUTPUT_DIR}/${FILE_NAME}.json \
    --num $NUM_CHUNKS \
    --prefix ${FILE_NAME}_${NUM_CHUNKS} 



if [ -f "${OUTPUT_DIR}/${FILE_NAME}.json" ]; then
    for IDX in  $(seq 0 $((NUM_CHUNKS-1))); do
        CHUNK_OUTPUT_PATH=${OUTPUT_DIR}/${FILE_NAME}_${NUM_CHUNKS}_${IDX}.json
        rm $CHUNK_OUTPUT_PATH
    done
fi 
