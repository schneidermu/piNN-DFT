#!/bin/bash
#SBATCH --job-name="T19_G2_AuxMean"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-T19_G2_AuxMean_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

GEOMETRY_MODE=primary_pcgrad_auxmean
GEOMETRY_TAG=g2_primary_pcgrad_auxmean
COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/train_models/trial19_gradient_geometry_sweep/run_geometry_job.sh"
[[ -f "$COMMON_JOB" ]] || COMMON_JOB="${SLURM_SUBMIT_DIR:-$PWD}/trial19_gradient_geometry_sweep/run_geometry_job.sh"
source "$COMMON_JOB"
