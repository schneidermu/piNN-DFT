from mpmath import chebyt, chop, taylor
import numpy as np
import os


sbatch_template = '''#! /bin/bash
#SBATCH --job-name="NN_{functional}"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="/home/mmedvedev/schnm/log/13_august/RELATIVE_AE17/{functional}_0.4_{omega:.5f}_"%j.out
#SBATCH --constraint="type_e"
#SBATCH --time=3-0
# Executable
python test.py --Name {functional} --N_preopt 3 --N_train 300 --Batch_size {batch_size} --Dropout 0.6 --Omega {omega:.5f} --LR_predopt 0.02'''


# Names and batch sizes
functionals = [
    ("PBE_8_32", 4),
    ("XALPHA_32_32", 6),
    ("XALPHA_16_32", 12),
    ("XALPHA_8_32", 12),
    ("XALPHA_4_512", 1)
]

n = 9
omegas = list(np.roots(chop(taylor(lambda x: chebyt(n, x), 0, n))[::-1])/2+0.5) + [0,]

for functional, batch_size in functionals:
    for omega in omegas:
        script = sbatch_template.format(
            functional=functional,
            omega=omega,
            batch_size=batch_size,
        )
        with open("run_calculations.slurm", "w") as file:
            file.write(script)
        os.system("sbatch run_calculations.slurm")
