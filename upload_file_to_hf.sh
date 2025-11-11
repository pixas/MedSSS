#!/bin/bash
#SBATCH --partition=medai_llm_p
#SBATCH --quotatype=reserved
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G 
#SBATCH --time=1-00:00:00
#SBATCH --output=upload.log
###SBATCH --kill-on-bad-exit=1



nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node IP: $head_node_ip"


if [[ "$SLURM_NODELIST" == "SH-IDCA1404-10-140-54-16" ]]; then
    quotatype="reserved"
else
    quotatype="spot"
fi
# print current directory
# echo "Current directory: $(pwd)"
job_id=$(sbatch -w $head_node --quotatype=${quotatype} -o upload.log /mnt/petrelfs/jiangshuyang/s3mount.sh | awk '{print $4}')
# bash /mnt/petrelfs/jiangshuyang/s3mount.sh &
# srun bash /mnt/petrelfs/jiangshuyang/s3mount.sh
# echo "Submitted job ID: $job_id"
sleep 10

# ls /nvme/jiangshuyang/s3_mount
# ls /mnt/petrelfs/jiangshuyang/s3_mount
# srun python test.py
source_dir="$1"
target_dir="$2"
srun python -m Evol_Instruct.utils.upload_to_hf

scancel $job_id
