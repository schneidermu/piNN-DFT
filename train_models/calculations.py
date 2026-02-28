import subprocess

import numpy as np
from mpmath import chebyt, chop, taylor

train_sbatch_template = """#! /bin/bash
#SBATCH --job-name="NN_{functional}"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="logs/{functional}_0.2_{omega:.5f}_RMSE_"%j.out
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=3-0
#SBATCH --exclude=cn-024

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Executable
CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT predopt_train.py \
    --name {functional} \
    --n_predopt 2 \
    --n_train 300 \
    --batch_size {batch_size} \
    --dropout 0.0 \
    --omega {omega:.5f} \
    --lr_predopt 0.01 \
    --lr_train 0.0001 \
    --weight_decay 0.0 \
    --optimizer radamw \
    --vxc_lr_scale 1.0 \
    --grad_clip 0.1 \
    --probe_vxc_steps 20 \
    --preopt_vxc_weight 0.05 \
    --preopt_vxc_steps 128 \
    --preopt_vxc_target pbe
"""

functionals = [
    ("PBE-L_6_64", 1),
#    ("PBESTAR_6_32", 1),
#    ("PBESTARSTAR_6_32", 1),
#    ("XALPHA_6_128", 1),
]

n = 9
omegas = list(np.roots(chop(taylor(lambda x: chebyt(n, x), 0, n))[::-1]) / 2 + 0.5) + [
    0,
    1,
]

omegas = np.array(omegas)

calculation_job_ids = []


for functional, batch_size in functionals:
    for omega in omegas:
        print("-" * 80)
        print(f"Submitting for: {functional} with Omega = {omega:.5f}")

        train_script_content = train_sbatch_template.format(
            functional=functional, omega=omega, batch_size=batch_size
        )
        train_script_file = "temp_train_job.slurm"
        with open(train_script_file, "w") as file:
            file.write(train_script_content)

        try:
            train_job_id_bytes = subprocess.check_output(
                ["sbatch", "--parsable", train_script_file]
            )
            train_job_id = train_job_id_bytes.decode().strip()
            print(f"Training job submitted. ID: {train_job_id}")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to submit training job. Slurm error: {e}")
            continue
