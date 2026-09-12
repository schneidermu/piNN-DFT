#!/bin/bash
#SBATCH --job-name="T19_NR5_MERGE"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr5_merged_anchor_%j.out

export NO_REPAIR_PRESET=no_repair_merged_anchor
export NO_REPAIR_TAG=nr5_merged_anchor
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
