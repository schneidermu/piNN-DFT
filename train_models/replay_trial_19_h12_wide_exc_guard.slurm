#! /bin/bash
#SBATCH --job-name="t19_h12_6x48"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="trial19_h12_6x48_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$SUBMIT_DIR/replay_trial_19_hypothesis_sweep.py" ]]; then
    WORK_DIR="$SUBMIT_DIR"
elif [[ -f "$SUBMIT_DIR/train_models/replay_trial_19_hypothesis_sweep.py" ]]; then
    WORK_DIR="$SUBMIT_DIR/train_models"
else
    echo "Could not locate replay_trial_19_hypothesis_sweep.py from submit dir: $SUBMIT_DIR" >&2
    exit 1
fi
cd "$WORK_DIR"

RUN_ROOT="${TRIAL19_RUN_ROOT:-${SCRATCH:-$HOME}/trial19_hypothesis_sweep_3}"
mkdir -p "$RUN_ROOT"

if [[ -n "${CHECKPOINTS_DIR:-}" ]]; then
    RESOLVED_CHECKPOINTS_DIR="$CHECKPOINTS_DIR"
elif [[ -f "$WORK_DIR/checkpoints/data_predopt.pickle" ]]; then
    RESOLVED_CHECKPOINTS_DIR="$WORK_DIR/checkpoints"
elif [[ -f "$WORK_DIR/../checkpoints/data_predopt.pickle" ]]; then
    RESOLVED_CHECKPOINTS_DIR="$WORK_DIR/../checkpoints"
else
    echo "Could not locate checkpoints/data_predopt.pickle from work dir: $WORK_DIR" >&2
    exit 1
fi

MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    replay_trial_19_hypothesis_sweep.py \
    --hypothesis h12_wide_exc_guard \
    --output-dir "$RUN_ROOT/h12_wide_exc_guard_6_48" \
    --checkpoints-dir "$RESOLVED_CHECKPOINTS_DIR" \
    --force-preopt --seed 41 --name PBE-LGxGc_6_48 --n-predopt 2 --n-train 900 \
    --batch-size 1 --vxc-batch-size 1 --lr-predopt 1e-2 --dropout 0.0 --weight-decay 5e-3 \
    --num-workers-train 4 --num-workers-vxc 2 --preopt-vxc-weight 0.0 --preopt-vxc-steps 0 \
    --preopt-vxc-target pbe --trial-number 19 --include-mrks-dispersion --save-selected-checkpoints
