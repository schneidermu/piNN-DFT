#!/bin/bash
#SBATCH --job-name="T19_NR8_RSUM"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr8_sum_repair_%j.out

export NO_REPAIR_PRESET=repair_sum_merge
export NO_REPAIR_TAG=nr8_sum_repair
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
