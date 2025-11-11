OUTPUT_PATH="$1"
MODEL_PATH="$2"

INPUT_DATA="$3"
LOG_PATH="$4"
SAMPLING_NUMBER="$5"
SELFGEN_LORA_PATH=${6}


iter=${7}
NUM_CHUNKS=${8:-1}
VALUE_FUNCTION=${9:-None}

OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
filename=$(basename "$OUTPUT_PATH")
FILE_NAME="${filename%.*}"

LOG_DIR=$(dirname "$LOG_PATH")
filename=$(basename "$LOG_PATH")
LOG_NAME="${filename%.*}"

script_path=scripts/slurm/iter_self_gen_mcts_llama.sh





job_ids=""

for IDX in  $(seq 0 $((NUM_CHUNKS-1))); do
    CHUNK_OUTPUT_PATH=${OUTPUT_DIR}/${FILE_NAME}_${NUM_CHUNKS}_${IDX}.jsonl
    CHUNK_LOG_FILE=$LOG_DIR/${LOG_NAME}_${NUM_CHUNKS}_${IDX}.log

    job_id=$(sbatch $script_path $CHUNK_OUTPUT_PATH $MODEL_PATH $INPUT_DATA $CHUNK_LOG_FILE $SAMPLING_NUMBER  $SELFGEN_LORA_PATH  $iter $NUM_CHUNKS $IDX  "$VALUE_FUNCTION"  | awk '{print $4}')
    echo "Submitted batch job ${job_id}"
    job_ids="${job_ids}:${job_id}"
    sleep 2

done 

job_ids=${job_ids#:}
echo ${job_ids}
srun -p partition --dependency=afterok:${job_ids} python Evol_Instruct/utils/combine_json.py \
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
