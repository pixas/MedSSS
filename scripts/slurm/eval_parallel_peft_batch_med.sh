#!/bin/bash
#SBATCH -J eval_med_chunk
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=reserved
#SBATCH --debug
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=24:00:00
###SBATCH --kill-on-bad-exit=1


dir_path="$1"
NUM_CHUNKS="$2"
# wait
other_params="$3"

output_file=${dir_path}/merge.jsonl
> "$output_file"


for IDX in $(seq 0 $((NUM_CHUNKS-1))); do
    cat ${dir_path}/infer-${NUM_CHUNKS}-${IDX}.jsonl >> "$output_file"
done

argv=()
read -ra argv <<< "$other_params"

# echo "Evaluating ${DATASET}"
srun -p medai_llm --output=${dir_path}/eval.log python -m Evol_Instruct.evaluation.eval_bench \
    --input_file ${dir_path}/merge.jsonl \
    --output_file ${dir_path}/wrong.json \
    ${argv[@]} 