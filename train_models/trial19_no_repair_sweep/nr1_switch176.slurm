#!/bin/bash
#SBATCH --job-name="T19_NR1_S176"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr1_switch176_%j.out

export NO_REPAIR_PRESET=no_repair_switch176
export NO_REPAIR_TAG=nr1_switch176
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
