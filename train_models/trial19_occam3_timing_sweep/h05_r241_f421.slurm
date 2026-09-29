#!/bin/bash
#SBATCH --job-name="T19_O3_05"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-OCCAM3_05_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

OCCAM3_PRESET=occam3_r241_f421
OCCAM3_TAG=r241_f421
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_occam3_timing_sweep/run_occam3_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_occam3_timing_sweep/run_occam3_job.sh"
source "$COMMON_JOB"
