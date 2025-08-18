import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option(
    "--Functional", type=str, default="PBE0", help="Functional for calculation"
)
(Opts, args) = parser.parse_args()
functional = Opts.Functional


script_template = """#! /bin/bash
#SBATCH --job-name="Rho {molecule} {functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output="/home/xray/schneiderm/log_molden/{functional}_{molecule}_"%j.out
# Executable
python -m get_molden --Functional '{functional}' --Molecule {molecule}"""

script_atom_template = """#! /bin/bash
#SBATCH --job-name="Rho {atom} +{charge} {functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output="/home/xray/schneiderm/log_atoms/{functional}_{atom}_+{charge}_"%j.out
# Executable
python -m get_molden --Functional '{functional}' --Atom {atom} --Charge {charge}"""

molecules = ["BH3", "CO", "F2", "H2", "H2O", "HF", "Li2", "LiF", "LiH", "N2"]
atoms = [
    "Be 0",
    "B 1",
    "B 3",
    "C 2",
    "C 4",
    "N 3",
    "N 5",
    "O 4",
    "O 6",
    "F 5",
    "F 7",
    "Ne 0",
    "Ne 6",
    "Ne 8",
]

for molecule in molecules:
    slurm_file = f"get_molden.slurm"
    with open(slurm_file, "w") as file:
        file.write(script_template.format(functional=functional, molecule=molecule))
    os.system(f"sbatch {slurm_file}")

for atom_string in atoms:
    atom, charge = atom_string.split()
    charge = int(charge)

    slurm_file = f"get_molden_atom.slurm"
    with open(slurm_file, "w") as file:
        file.write(
            script_atom_template.format(functional=functional, atom=atom, charge=charge)
        )
    os.system(f"sbatch {slurm_file}")
