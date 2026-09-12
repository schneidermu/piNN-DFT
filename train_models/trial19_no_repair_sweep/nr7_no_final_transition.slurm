#!/bin/bash
#SBATCH --job-name="T19_NR7_NOFIN"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr7_no_final_transition_%j.out

export NO_REPAIR_PRESET=no_repair_no_final_transition
export NO_REPAIR_TAG=nr7_no_final_transition
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
