from mpmath import chebyt, chop, taylor
import numpy as np
import os


sbatch_template = '''#! /bin/bash
#SBATCH --job-name="NN_{functional}"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="/home/mmedvedev/schnm/log/20_november/{functional}_0.2_{omega:.5f}_"%j.out
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH --time=3-0
# Executable
python predopt_train.py --Name {functional} --N_preopt 3 --N_train 300 --Batch_size {batch_size} --Dropout 0.2 --Omega {omega:.5f} --LR_predopt 0.02'''


# Names and batch sizes
functionals = [
#    ("PBE_8_32", 3),
#    ("PBE_16_32", 4),
#    ("XALPHA_32_32", 2),
#    ("XALPHA_16_32", 6),
#    ("XALPHA_8_32", 8),
#    ("XALPHA_4_512", 1)
#    ("PBE_4_32", 4),
#    ("PBE_6_32", 4),
#    ("PBE_4_16", 4),
#    ("PBE_6_16", 4),
#    ("PBE_8_16", 4),
#    ("PBE_14_8", 4),
    ("PBE_8_4", 4),
#    ("PBE_6_4", 4),
#    ("PBE_6_12", 4),
#    ("PBE_6_24", 4),
#    ("PBE_6_48", 4),
#    ("PBE_8_8", 4),
]

n = 9
omegas = list(np.roots(chop(taylor(lambda x: chebyt(n, x), 0, n))[::-1])/2+0.5) + [0,1]

omegas = np.array(omegas)

omegas = omegas[omegas<0.5]
#omegas = [0.06699,]
omegas = [0,]

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
