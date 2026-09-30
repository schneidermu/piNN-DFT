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
if [[ -f "$SUBMIT_DIR/train_models/launch_provenance.py" ]]; then
    REPO_ROOT="$SUBMIT_DIR"
else
    REPO_ROOT=$(cd "$SUBMIT_DIR/.." && pwd -P)
fi
cd "$REPO_ROOT"
hostname
VERIFIED_COMMIT=$(python train_models/launch_provenance.py)
echo "Git commit: $VERIFIED_COMMIT"
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
[[ -d data && -s checkpoints/data_vxc_train.pickle && -s checkpoints/data_vxc_val.pickle ]] || {
    echo "Minnesota raw data or existing mRKS train/validation pickles are missing." >&2
    exit 1
}
du -shL data
python -u prepare_training_corpus.py \
    --mn-dir data \
    --mrks-train-pickle checkpoints/data_vxc_train.pickle \
    --mrks-val-pickle checkpoints/data_vxc_val.pickle \
    --output-dir checkpoints_dietclean_noval_v1
# Batch success gates all training jobs; verify independently after generation.
python prepare_training_corpus.py \
    --verify-only \
    --output-dir checkpoints_dietclean_noval_v1
