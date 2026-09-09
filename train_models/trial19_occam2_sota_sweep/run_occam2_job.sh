#!/bin/bash

set -euo pipefail

export PYTHONNOUSERSITE=1

: "${OCCAM2_PRESET:?OCCAM2_PRESET must be set by the SLURM file}"
: "${OCCAM2_TAG:?OCCAM2_TAG must be set by the SLURM file}"

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

if [[ -n "${CHECKPOINTS_DIR:-}" ]]; then
    RESOLVED_CHECKPOINTS_DIR="$CHECKPOINTS_DIR"
elif [[ -f "$WORK_DIR/checkpoints/data_predopt.pickle" ]]; then
    RESOLVED_CHECKPOINTS_DIR="checkpoints"
elif [[ -f "$WORK_DIR/../checkpoints/data_predopt.pickle" ]]; then
    RESOLVED_CHECKPOINTS_DIR="../checkpoints"
else
    echo "Could not locate checkpoints/data_predopt.pickle from: $WORK_DIR" >&2
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
    --output-dir "optuna_joint_runs/replay_trial_19_occam2_${OCCAM2_TAG}_500_gc_svelu_mirror" \
    --checkpoints-dir "$RESOLVED_CHECKPOINTS_DIR" \
    --force-preopt \
    --seed 41 \
    --name PBE-LGxGc_6_32 \
    --model-type gc_svelu_mirror \
    --n-predopt 2 \
    --n-train 500 \
    --e3-schedule-preset "$OCCAM2_PRESET" \
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
    --train-fchem-target 40 \
    --val-vxc-target 1.1 \
    --val-fchem-soft-cap 90 \
    --training-state-every 10 \
    --snapshot-start-epoch 400 \
    --snapshot-every 10 \
    --include-mrks-dispersion
