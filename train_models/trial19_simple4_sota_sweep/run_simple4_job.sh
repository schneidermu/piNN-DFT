#!/bin/bash

set -euo pipefail

export PYTHONNOUSERSITE=1

source /home/mmedvedev/anaconda3/etc/profile.d/conda.sh
conda activate ML_param

: "${SIMPLE4_PRESET:?SIMPLE4_PRESET must be set by the SLURM file}"
: "${SIMPLE4_TAG:?SIMPLE4_TAG must be set by the SLURM file}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$SUBMIT_DIR/replay_trial_19_bridge.py" ]]; then
    WORK_DIR="$SUBMIT_DIR"
elif [[ -f "$SUBMIT_DIR/train_models/replay_trial_19_bridge.py" ]]; then
    WORK_DIR="$SUBMIT_DIR/train_models"
else
    echo "Could not locate replay_trial_19_bridge.py from: $SUBMIT_DIR" >&2
    exit 1
fi

cd "$WORK_DIR"

REQUIRED_SCIENTIFIC_FIX="29b36ac2faa29d31a758e0067499f10b41c3ab39"
if ! git merge-base --is-ancestor "$REQUIRED_SCIENTIFIC_FIX" HEAD; then
    echo "Checkout lacks the required scientific-fix commit $REQUIRED_SCIENTIFIC_FIX." >&2
    exit 1
fi
echo "Git commit: $(git rev-parse HEAD)"
python -c 'import pyscf; print("PySCF:", pyscf.__version__)'

# All planned jobs default to the same newly regenerated corpus; no old-data fallback.
RESOLVED_CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$WORK_DIR/checkpoints_dietclean_noval_v1}"
RESOLVED_CHECKPOINTS_DIR=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RESOLVED_CHECKPOINTS_DIR")
python prepare_training_corpus.py --verify-only --output-dir "$RESOLVED_CHECKPOINTS_DIR"

OUTPUT_DIR="optuna_joint_runs/replay_trial_19_simple4_${SIMPLE4_TAG}_500_gc_svelu_mirror_dietclean_noval_v1"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to reuse existing output directory: $OUTPUT_DIR" >&2
    exit 1
fi

MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

python -c 'import numpy, pandas, pyarrow, sklearn, torch; assert int(numpy.__version__.split(".")[0]) < 2, numpy.__version__; print("Environment:", numpy.__version__, pandas.__version__, pyarrow.__version__, sklearn.__version__, torch.__version__)'

CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    replay_trial_19_bridge.py \
    --output-dir "$OUTPUT_DIR" \
    --checkpoints-dir "$RESOLVED_CHECKPOINTS_DIR" \
    --force-preopt \
    --seed 41 \
    --name PBE-LGxGc_6_32 \
    --model-type gc_svelu_mirror \
    --n-predopt 2 \
    --n-train 500 \
    --e3-schedule-preset "$SIMPLE4_PRESET" \
    --batch-size 1 \
    --vxc-batch-size 1 \
    --lr-predopt 1e-2 \
    --dropout 0.0 \
    --weight-decay 1e-2 \
    --num-workers-train 4 \
    --num-workers-vxc 2 \
    --preopt-vxc-weight 0.0 \
    --preopt-vxc-steps 0 \
    --preopt-vxc-target pbe \
    --trial-number 19 \
    --training-state-every 10 \
    --snapshot-start-epoch 10 \
    --snapshot-every 10 \
    --include-mrks-dispersion
