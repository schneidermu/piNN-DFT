#!/bin/bash
#SBATCH --job-name="T19_NR3_S220"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr3_switch220_%j.out

export NO_REPAIR_PRESET=no_repair_switch220
export NO_REPAIR_TAG=nr3_switch220
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
