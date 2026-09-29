#!/bin/bash
#SBATCH --job-name="T19_S5_RF_04"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-S5_RF_04_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

S5_TIMING_PRESET=s5_timing_r281_f441
S5_TIMING_TAG=r281_f441
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_occam3_timing_sweep/run_occam3_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_occam3_timing_sweep/run_occam3_job.sh"
source "$COMMON_JOB"
