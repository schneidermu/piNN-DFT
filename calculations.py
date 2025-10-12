import numpy as np
import subprocess

from mpmath import chebyt, chop, taylor

train_sbatch_template = """#! /bin/bash
#SBATCH --job-name="NN_{functional}"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="../log/NN_PBE_final/{functional}_0.2_{omega:.5f}_RMSE_"%j.out
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=3-0
#SBATCH --exclude=cn-003
# Executable
CUBLAS_WORKSPACE_CONFIG=:16:8 python predopt_train.py --Name {functional} --N_preopt 10 --N_train 200 --Batch_size {batch_size} --Dropout 0.2 --Omega {omega:.5f} --LR_predopt 0.01 --LR_train 0.00003"""

functionals = [
    ("PBE_6_32", 1),
    ("PBESTAR_6_32", 1),
    ("PBESTARSTAR_6_32", 1),
    ("XALPHA_6_128", 1),
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
            train_job_id_bytes = subprocess.check_output(["sbatch", "--parsable", train_script_file])
            train_job_id = train_job_id_bytes.decode().strip()
            print(f"Training job submitted. ID: {train_job_id}")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to submit training job. Slurm error: {e}")
            continue


