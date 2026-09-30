#!/bin/bash
#SBATCH --job-name="DIETCLEAN_PREP_V1"
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --constraint="type_d"
#SBATCH --output="dietclean_noval_prep_%j.out"

# HSE rocky advertises RealMemory=1 MB; use its unlimited default memory.
# Explicit --mem/--mem-per-cpu requests cannot match that scheduler metadata.
set -euo pipefail
export PYTHONNOUSERSITE=1
source /home/mmedvedev/anaconda3/etc/profile.d/conda.sh
conda activate ML_param

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT=$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"
REQUIRED_SCIENTIFIC_FIX="29b36ac2faa29d31a758e0067499f10b41c3ab39"
git merge-base --is-ancestor "$REQUIRED_SCIENTIFIC_FIX" HEAD || {
    echo "Checkout predates the required protocol/manifest commit: $REQUIRED_SCIENTIFIC_FIX" >&2
    exit 1
}
hostname
git rev-parse HEAD
python --version
python -c 'import torch, numpy, h5py; print("torch:", torch.__version__, "numpy:", numpy.__version__, "h5py:", h5py.__version__)'
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""
cd train_models
[[ ! -e checkpoints_dietclean_noval_v1 ]] || {
    echo "Refusing to overwrite existing corpus: $PWD/checkpoints_dietclean_noval_v1" >&2
    exit 1
}
[[ -d data && -d h5_vrho_from_mrks ]] || {
    echo "Required raw H5 source directories are missing." >&2
    exit 1
}
du -sh data h5_vrho_from_mrks
python prepare_training_corpus.py \
    --mn-dir data \
    --mrks-dir h5_vrho_from_mrks \
    --output-dir checkpoints_dietclean_noval_v1
# Batch success gates all training jobs; verify independently after generation.
python prepare_training_corpus.py \
    --verify-only \
    --output-dir checkpoints_dietclean_noval_v1
