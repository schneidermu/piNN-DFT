#!/bin/bash
#SBATCH --job-name="T19_O2_O3"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-OCCAM2_O3_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

OCCAM2_PRESET=occam2_o3_three_stage_pulse
OCCAM2_TAG=o3_three_stage_pulse
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_occam2_sota_sweep/run_occam2_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_occam2_sota_sweep/run_occam2_job.sh"
source "$COMMON_JOB"
