#!/bin/bash
#SBATCH --job-name="T19_S4_8_R430"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-T19_S4_8_R430_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

SIMPLE4_PRESET=simple4_linear_repair430
SIMPLE4_TAG=s8_linear_repair430
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_simple4_sota_sweep/run_simple4_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_simple4_sota_sweep/run_simple4_job.sh"
source "$COMMON_JOB"
