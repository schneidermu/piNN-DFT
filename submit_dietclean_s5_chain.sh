#!/bin/bash
# Submit one CPU preparation job and eleven parallel afterok training siblings.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$REPO_ROOT"
REQUIRED_SCIENTIFIC_FIX="29b36ac2faa29d31a758e0067499f10b41c3ab39"
git merge-base --is-ancestor "$REQUIRED_SCIENTIFIC_FIX" HEAD || {
    echo "Checkout predates required protocol/manifest commit: $REQUIRED_SCIENTIFIC_FIX" >&2
    exit 1
}
export CHECKPOINTS_DIR="$REPO_ROOT/train_models/checkpoints_dietclean_noval_v1"
[[ ! -e "$CHECKPOINTS_DIR" ]] || {
    echo "Refusing submission: corpus target already exists: $CHECKPOINTS_DIR" >&2
    exit 1
}
for source in train_models/data train_models/h5_vrho_from_mrks; do
    [[ -d "$source" ]] && compgen -G "$source/*.h5" > /dev/null || {
        echo "Required raw H5 directory is missing or empty: $REPO_ROOT/$source" >&2
        exit 1
    }
done
jobs=(
    train_models/trial19_simple4_sota_sweep/s5_two_step_40_10.slurm
    train_models/trial19_occam3_timing_sweep/h01_r241_f441.slurm
    train_models/trial19_occam3_timing_sweep/h02_r261_f441.slurm
    train_models/trial19_occam3_timing_sweep/h03_r221_f441.slurm
    train_models/trial19_occam3_timing_sweep/h04_r281_f441.slurm
    train_models/trial19_occam3_timing_sweep/h05_r241_f421.slurm
    train_models/trial19_occam3_timing_sweep/h06_r261_f421.slurm
    train_models/trial19_occam3_timing_sweep/h07_r281_f421.slurm
    train_models/trial19_occam3_timing_sweep/h08_r221_f421.slurm
    train_models/trial19_occam3_timing_sweep/h09_r241_f461.slurm
    train_models/trial19_occam3_timing_sweep/h10_r261_f461.slurm
)
for script in train_models/prepare_dietclean_noval_v1.slurm "${jobs[@]}"; do
    [[ -f "$script" ]] || { echo "Missing launch script: $script" >&2; exit 1; }
done
# Accept JOBID or JOBID;CLUSTER, and reject ambiguous/non-machine-readable output.
submit_job() {
    local response
    response=$(sbatch --parsable "$@") || {
        echo "Submission failed; inspect already printed job IDs before retrying." >&2
        return 1
    }
    [[ "$response" =~ ^[0-9]+(\;[^[:space:]\;]+)?$ ]] || {
        echo "Invalid sbatch --parsable response: $response" >&2
        return 1
    }
    printf '%s\n' "${response%%;*}"
}
PREP_JOB_ID=$(submit_job --chdir="$REPO_ROOT" train_models/prepare_dietclean_noval_v1.slurm)
printf 'CPU preprocessing: %s (log: %s/dietclean_noval_prep_%s.out)\n' "$PREP_JOB_ID" "$REPO_ROOT" "$PREP_JOB_ID"
for script in "${jobs[@]}"; do
    job_id=$(submit_job --chdir="$REPO_ROOT" --export=ALL,CHECKPOINTS_DIR \
        --dependency="afterok:$PREP_JOB_ID" "$script")
    printf 'Training %s: %s (afterok:%s)\n' "$script" "$job_id" "$PREP_JOB_ID"
done
printf 'Inspect with: squeue -u "%s" -o "%%i %%j %%T %%r %%E"\n' "${USER:-user}"
