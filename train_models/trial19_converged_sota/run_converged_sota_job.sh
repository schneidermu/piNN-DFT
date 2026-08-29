#!/bin/bash

set -euo pipefail

export PYTHONNOUSERSITE=1

: "${CONVERGED_SOTA_PRESET:?CONVERGED_SOTA_PRESET must be set by the SLURM file}"
: "${CONVERGED_SOTA_TAG:?CONVERGED_SOTA_TAG must be set by the SLURM file}"

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
    echo "Set CHECKPOINTS_DIR=/path/to/checkpoints when submitting if needed." >&2
    exit 1
fi

N_TRAIN="${N_TRAIN:-800}"
OUTPUT_DIR="optuna_joint_runs/replay_trial_19_converged_${CONVERGED_SOTA_TAG}_gc_svelu_mirror"
RESUME_ARGS=()
PREOPT_ARGS=(--force-preopt)
if [[ -n "${RESUME_TRAINING_STATE:-}" ]]; then
    RESUME_ARGS=(--resume-training-state "$RESUME_TRAINING_STATE")
    PREOPT_ARGS=()
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
    "${PREOPT_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --seed 41 \
    --name PBE-LGxGc_6_32 \
    --model-type gc_svelu_mirror \
    --n-predopt 2 \
    --n-train "$N_TRAIN" \
    --converged-sota-schedule-preset "$CONVERGED_SOTA_PRESET" \
    --convergence-base-epochs 500 \
    --convergence-tail-epochs 300 \
    --convergence-tail-start-lr 1e-5 \
    --convergence-tail-min-lr 1e-7 \
    --training-state-every 1 \
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
    --include-mrks-dispersion
