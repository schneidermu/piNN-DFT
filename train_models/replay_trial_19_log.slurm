#! /bin/bash
#SBATCH --job-name="trial19_log_PBE-LGxGc"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="trial19_log_PBE-LGxGc_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$SUBMIT_DIR/replay_trial_19_bridge.py" ]]; then
    WORK_DIR="$SUBMIT_DIR"
elif [[ -f "$SUBMIT_DIR/train_models/replay_trial_19_bridge.py" ]]; then
    WORK_DIR="$SUBMIT_DIR/train_models"
else
    WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

if [[ -n "${TRIAL19_RUN_ROOT:-}" ]]; then
    RUN_ROOT="$TRIAL19_RUN_ROOT"
elif [[ -n "${SCRATCH:-}" ]]; then
    RUN_ROOT="$SCRATCH/trial19_log_runs"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
    RUN_ROOT="$SLURM_TMPDIR/trial19_log_runs"
elif [[ -n "${TMPDIR:-}" ]]; then
    RUN_ROOT="$TMPDIR/trial19_log_runs"
else
    RUN_ROOT="$HOME/trial19_log_runs"
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/replay_trial_19_log_runs"
cd "$WORK_DIR"

if [[ -n "${CHECKPOINTS_DIR:-}" ]]; then
    RESOLVED_CHECKPOINTS_DIR="$CHECKPOINTS_DIR"
elif [[ -f "$WORK_DIR/checkpoints/data_predopt.pickle" ]]; then
    RESOLVED_CHECKPOINTS_DIR="$WORK_DIR/checkpoints"
elif [[ -f "$WORK_DIR/../checkpoints/data_predopt.pickle" ]]; then
    RESOLVED_CHECKPOINTS_DIR="$WORK_DIR/../checkpoints"
else
    echo "Could not locate checkpoints/data_predopt.pickle from work dir: $WORK_DIR" >&2
    echo "Set CHECKPOINTS_DIR=/path/to/checkpoints when submitting if needed." >&2
    exit 1
fi

MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun \
    --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    replay_trial_19_bridge.py \
    --output-dir "$RUN_ROOT/replay_trial_19_log_runs/PBE-LGxGc_6_64_log" \
    --checkpoints-dir "$RESOLVED_CHECKPOINTS_DIR" \
    --seed 41 \
    --name PBE-LGxGc_6_64 \
    --model-type log \
    --n-predopt 2 \
    --n-train 500 \
    --batch-size 1 \
    --vxc-batch-size 1 \
    --lr-predopt 1e-2 \
    --dropout 0.0 \
    --weight-decay 1e-2 \
    --num-workers-train 4 \
    --num-workers-vxc 2 \
    --trial-number 19 \
    --include-mrks-dispersion \
    --save-selected-checkpoints
