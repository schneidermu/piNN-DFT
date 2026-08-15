#!/bin/bash
#SBATCH --job-name="T19_O4_QStep3"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-T19_O4_QStep3_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

OCCAM_PRESET=occam_quarter_step_triple
OCCAM_TAG=o4_quarter_step_triple
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_occam_schedule_sweep/run_occam_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_occam_schedule_sweep/run_occam_job.sh"
source "$COMMON_JOB"
