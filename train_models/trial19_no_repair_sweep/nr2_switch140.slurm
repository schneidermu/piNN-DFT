#!/bin/bash
#SBATCH --job-name="T19_NR2_S140"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr2_switch140_%j.out

export NO_REPAIR_PRESET=no_repair_switch140
export NO_REPAIR_TAG=nr2_switch140
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
