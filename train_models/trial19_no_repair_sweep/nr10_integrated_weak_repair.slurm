#!/bin/bash
#SBATCH --job-name="T19_NR10_WEAK"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/t19_nr10_integrated_weak_repair_%j.out

export NO_REPAIR_PRESET=repair_integrated_weak
export NO_REPAIR_TAG=nr10_integrated_weak_repair
bash "${SLURM_SUBMIT_DIR}/trial19_no_repair_sweep/run_no_repair_job.sh"
