import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option(
    "--Functional", type=str, default="PBE0", help="Functional for calculation"
)
(Opts, args) = parser.parse_args()
functional = Opts.Functional


script_template = '''#! /bin/bash
#SBATCH --job-name="NN Functionals Benchmark"
#SBATCH --ntasks=4
#SBATCH --output="/home/xray/schneiderm/log_files/molden/{functional}_{molecule}_"%j.out
# Executable
python -m get_molden --Functional {functional} --Molecule {molecule_name}'''

molecules = ['BH3', 'CO', 'F2', 'H2', 'H2O', 'HF', 'Li2', 'LiF', 'LiH', 'N2']

for molecule in molecules:
    filename = f"./molden/{molecule}.xyz"
    slurm_file = f"get_molden_{molecule}.slurm"
    with open(slurm_file, "w") as file:
        file.write(script_template.format(functional=functional, molecule=molecule))
    os.system(f"sbatch {slurm_file}")