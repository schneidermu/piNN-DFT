#! /bin/bash
#SBATCH --job-name="T19_h9_X1_soft"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="PBE-Replay_T19_h9_X1_soft_%j.out"
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
    echo "Could not locate replay_trial_19_bridge.py from submit dir: $SUBMIT_DIR" >&2
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
    echo "Could not locate checkpoints/data_predopt.pickle from work dir: $WORK_DIR" >&2
    echo "Set CHECKPOINTS_DIR=/path/to/checkpoints when submitting if needed." >&2
    exit 1
fi

MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    replay_trial_19_bridge.py \
    --output-dir optuna_joint_runs/replay_trial_19_h9_exploit_x1_soft_exc2_vxc10_r085_gc_svelu_mirror \
    --checkpoints-dir "$RESOLVED_CHECKPOINTS_DIR" \
    --force-preopt \
    --seed 41 \
    --name PBE-LGxGc_6_32 \
    --model-type gc_svelu_mirror \
    --n-predopt 2 \
    --n-train 500 \
    --h9-schedule-preset h9_exploit_soft_exc2_vxc10_r085 \
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