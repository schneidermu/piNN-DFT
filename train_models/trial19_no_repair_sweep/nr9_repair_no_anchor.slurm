#!/bin/bash
#SBATCH --job-name="T19_NR9_RNOA"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr9_repair_no_anchor_%j.out

export NO_REPAIR_PRESET=repair_no_anchor
export NO_REPAIR_TAG=nr9_repair_no_anchor
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
