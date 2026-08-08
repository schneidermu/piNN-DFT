#!/bin/bash
#SBATCH --job-name="T19_G9_RawPrimary"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-T19_G9_RawPrimary_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

GEOMETRY_MODE=raw_primary_pcgrad
GEOMETRY_TAG=g9_raw_primary_pcgrad
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_gradient_geometry_sweep/run_geometry_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_gradient_geometry_sweep/run_geometry_job.sh"
source "$COMMON_JOB"
