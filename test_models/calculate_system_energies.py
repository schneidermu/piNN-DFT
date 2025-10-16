import os
from optparse import OptionParser

from DFT.functional import omega_str_list

func_list = (
    [f"NN_PBE_{omega}" for omega in omega_str_list]
    + [f"NN_XALPHA_{omega}" for omega in omega_str_list]
    + ["Nagai"]
    + ["NN_PBE_star", "NN_PBE_star_star"]
    + [f"NN_PBE_star_star_{omega}" for omega in omega_str_list]
)

parser = OptionParser()
parser.add_option("--Mode", type="string", default="Analyse", help="Mode")
parser.add_option(
    "--Functional", type="string", default="NN_PBE_0", help="Functional to evaluate"
)

(Opts, args) = parser.parse_args()

Functional = Opts.Functional

script_template = """#! /bin/bash
#SBATCH --job-name="E {system_name} {Functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output="logs/log_energy/{Functional}_{system_name}_"%j.out
# Executable
python -m script --System {system_name} --NFinal 30 --Functional {Functional}"""
dispersion_script_template = """#! /bin/bash
#SBATCH --job-name="E {system_name} {Functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output="logs/log_energy//D3(BJ)_{system_name}_"%j.out
# Executable
python -m script --Dispersion True --System {system_name} --NFinal 30"""



Mode = Opts.Mode.upper()[:2]
if Mode == "GE":
    filenames = sorted(list(os.walk("GIF"))[0][2:][0])
    filenames = [name for name in filenames if name.endswith(".gif_")]
    for name in filenames:
        system_name = name[:-5]
        dir = f"GIF/{system_name}"
        old_dir = f"GIF/{system_name}.gif_"
        new_dir = f"{dir}/{system_name}.gif_"
        try:
            os.mkdir(dir)
            os.rename(old_dir, new_dir)
        except:
            print(f"{dir} already exists")
        for Functional in func_list:
            with open(
                f"GIF/{system_name}/calculate_system_energy_{Functional}.slurm", "w"
            ) as file:
                file.write(
                    script_template.format(
                        system_name=system_name, NFinal=30, Functional=Functional
                    )
                )
            print(f"GIF/{system_name}/calculate_system_energy_{Functional}.slurm")
        for non_nn in ["PBE", "XAlpha", "r2SCAN", "SCAN", "TPSS"]:
            with open(
                f"GIF/{system_name}/calculate_system_energy_{non_nn}.slurm", "w"
            ) as file:
                file.write(
                    script_template.format(
                        system_name=system_name, NFinal=30, Functional=non_nn
                    )
                )
elif Mode == "CE":
    filenames = list(os.walk("GIF"))[1:]
    for name in sorted(filenames):
        current_path = name[0].replace("\\", "/")
        slurm_path = f"{current_path}/calculate_system_energy_{Functional}.slurm"
        print(slurm_path)
        os.system(f"sbatch {slurm_path}")

elif Mode == "D3":
    filenames = list(os.walk("GIF"))[1:]
    for name in sorted(filenames):
        current_path = name[0].replace("\\", "/")
        slurm_path = f"{current_path}/calculate_system_dispersion.slurm"
        os.system(f"sbatch {slurm_path}")
