#!/bin/bash
#SBATCH --job-name="T19_NR6_LOW"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr6_anchor_direct_low_%j.out

export NO_REPAIR_PRESET=no_repair_anchor_direct_low
export NO_REPAIR_TAG=nr6_anchor_direct_low
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
