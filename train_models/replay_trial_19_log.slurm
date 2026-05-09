#! /bin/bash
#SBATCH --job-name="trial19_log_PBE-LGxGc"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="logs/trial19_log_PBE-LGxGc_%j.out"
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=7-0
#SBATCH --exclude=cn-024

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs replay_trial_19_log_runs

MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun \
    --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    replay_trial_19_bridge.py \
    --output-dir replay_trial_19_log_runs/PBE-LGxGc_6_64_log \
    --checkpoints-dir checkpoints \
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
