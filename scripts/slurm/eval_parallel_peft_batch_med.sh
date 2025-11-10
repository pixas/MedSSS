#!/bin/bash
#SBATCH -J eval_med_chunk
#SBATCH --partition=partition
#SBATCH -N1
#SBATCH --quotatype=reserved
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=4G  
#SBATCH --time=24:00:00
###SBATCH --kill-on-bad-exit=1


dir_path="$1"
NUM_CHUNKS="$2"
# wait
other_params="$3"

first_eval=0
output_file=${dir_path}/merge.jsonl
for IDX in $(seq 0 $((NUM_CHUNKS-1))); do
    CHUNK_OUTPUT_PATH=${dir_path}/infer-${NUM_CHUNKS}-${IDX}.jsonl
    if [ -f $CHUNK_OUTPUT_PATH ]; then
        echo "first evaluate; merge"
        first_eval=1
        > "$output_file"
        break
    fi
done


if [[ $first_eval -eq 1 ]]; then
    for IDX in $(seq 0 $((NUM_CHUNKS-1))); do
        cat ${dir_path}/infer-${NUM_CHUNKS}-${IDX}.jsonl >> "$output_file"
    done

    # clear chunked file
    for IDX in $(seq 0 $((NUM_CHUNKS-1))); do
        CHUNK_OUTPUT_PATH=${dir_path}/infer-${NUM_CHUNKS}-${IDX}.jsonl
        rm $CHUNK_OUTPUT_PATH
    done
fi


argv=()
read -ra argv <<< "$other_params"

# echo "Evaluating ${DATASET}"
srun -p medai_llm_p --output=${dir_path}/eval.log python -m Evol_Instruct.evaluation.eval_bench \
    --input_file ${dir_path}/merge.jsonl \
    --output_file ${dir_path}/wrong.json \
    ${argv[@]} 