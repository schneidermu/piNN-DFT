#!/bin/bash
#SBATCH --job-name="T19_P7_80C1"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-T19_P7_80C1_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

MINIMAL_SOTA_PRESET=minimal_v80_clip_r1
MINIMAL_SOTA_TAG=p7_v80_clip_r1
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_minimal_sota_sweep/run_minimal_sota_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_minimal_sota_sweep/run_minimal_sota_job.sh"
source "$COMMON_JOB"
