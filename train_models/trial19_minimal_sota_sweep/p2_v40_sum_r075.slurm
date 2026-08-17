#!/bin/bash
#SBATCH --job-name="T19_P2_40S075"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-T19_P2_40S075_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

MINIMAL_SOTA_PRESET=minimal_v40_sum_r075
MINIMAL_SOTA_TAG=p2_v40_sum_r075
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_minimal_sota_sweep/run_minimal_sota_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_minimal_sota_sweep/run_minimal_sota_job.sh"
source "$COMMON_JOB"
