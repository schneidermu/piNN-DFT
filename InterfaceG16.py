import os
from optparse import OptionParser

import yaml

from DFT.functional import omega_str_list

func_list = (
    [f"NN_PBE_{omega}" for omega in omega_str_list]
    + [f"NN_XALPHA_{omega}" for omega in omega_str_list]
    + ["Nagai"]
    + ["NN_PBE_star", "NN_PBE_star_star"]
    + [f"NN_PBE_star_star_{omega}" for omega in omega_str_list]
)

parser = OptionParser()
parser.add_option("--NFinal", type=int, default=30, help="Number systems to select")
parser.add_option("--Mode", type="string", default="Analyse", help="Mode")
parser.add_option(
    "--Functional", type="string", default="NN_PBE_0", help="Functional to evaluate"
)

(Opts, args) = parser.parse_args()

Functional = Opts.Functional
NFinal = Opts.NFinal
script_template = """#! /bin/bash
#SBATCH --job-name="E {system_name} {Functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output="../log_files/log_files/{Functional}_{system_name}_"%j.out
# Executable
python -m script --System {system_name} --NFinal {NFinal} --Functional {Functional}"""
dispersion_script_template = """#! /bin/bash
#SBATCH --job-name="E {system_name} {Functional}"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output="../log_files/log_files/D3(BJ)_{system_name}_"%j.out
# Executable
python -m script --Dispersion True --System {system_name} --NFinal {NFinal}"""


def WriteSystems(NFinal=30, Suff=""):
    Y = yaml.safe_load(open("GoodSamples/AllElements_%03d%s.yaml" % (NFinal, Suff)))
    IDs = []
    FC = open("GIF/ComboList_%d.txt" % (NFinal), "w")
    for Set in Y:
        for l in sorted(Y[Set]):
            ID0 = Set + "-" + "%d" % (l)

            Comment = ID0 + " : energy = %.4f" % (Y[Set][l]["Energy"])

            FC.write("%s:\n" % (ID0))
            FC.write("  Energy: %.4f\n" % (Y[Set][l]["Energy"]))
            FC.write("  Weight: %.4f\n" % (Y[Set][l]["Weight"]))
            FC.write("  Species:\n")
            SS = Y[Set][l]["Species"]
            for P in SS:
                ID = ID0 + "-" + str(P)
                IDs += [ID]

                FC.write('    - { ID: "%s", Count: %d }\n' % (ID, SS[P]["Count"]))

                C = SS[P]["Charge"]
                D = SS[P]["UHF"] + 1

                F = open("GIF/%s.gif_" % (ID), "w")
                F.write("%s\n\n" % (ID))
                F.write("%d %d\n" % (C, D))
                for l in range(SS[P]["Number"]):
                    x, y, z = tuple(SS[P]["Positions"][l])
                    F.write(
                        "%-4s %12.5f %12.5f %12.5f\n" % (SS[P]["Elements"][l], x, y, z)
                    )
                F.close()
    FC.close()

    F = open("GIF/FullList_%d.txt" % (NFinal), "w")
    for ID in list(set(IDs)):
        F.write("%s.gif_\n" % (ID))
    F.close()


def G16ExtractEnergies(FileName):
    # Extract the energy from a Gaussian run
    # Var = grep "SCF Done" FILENAME
    # Energy in Ha is 5th element after split
    # This assume a file processed like in the
    # examples
    F = open(FileName)
    E = {}
    for L in F:
        if L[0] == "#":
            continue

        T = L.split()
        if len(T) > 1:
            ID = T[0][:-5]
            En = float(T[1])
            E[ID] = En * 627.509  # Ha -> kcal/mol
        else:
            print("Warning! %s not defined" % (T[0]))
    return E


def ReadSystems(NFinal=NFinal, InputFile=None):
    Y = yaml.safe_load(open("GIF/ComboList_%d.txt" % (NFinal)))

    if InputFile is None:
        InputFile = f"Results/EnergyList_{NFinal}_{Functional}.txt"

    G16Energy = G16ExtractEnergies(InputFile)

    WarningList = []

    MAE, MAEDen = 0.0, 0.0
    Errors = {}
    for System in Y:
        Energy = Y[System]["Energy"]
        Weight = Y[System]["Weight"]

        EnergyError = False
        EnergyApprox = 0.0
        for S in Y[System]["Species"]:
            if not S["ID"] in G16Energy:
                EnergyError = True
            else:
                EnergyApprox += G16Energy[S["ID"]] * float(S["Count"])

        EnergyDiff = EnergyApprox - Energy
        if EnergyError:
            print(
                "Skipping system due to absence: %s [Weight: %.2f]" % (System, Weight)
            )
            WarningList += [S["ID"] for S in Y[System]["Species"]]
        elif abs(EnergyDiff * Weight) > 1e3:
            ErrList = [S["ID"] for S in Y[System]["Species"]]
            print(
                "Major error: %s - %.2f vs %.2f [Weight: %.2f]:"
                % (System, EnergyApprox, Energy, Weight)
            )
            print("Species: " + ", ".join(ErrList))
            print("Energies:" + ", ".join(["%.2f" % (G16Energy[x]) for x in ErrList]))
            WarningList += ErrList
        elif not (EnergyError):
            print(
                "%-16s %8.2f %8.2f %8.2f"
                % (System, EnergyApprox, Energy, abs(EnergyDiff))
            )
            Errors[System] = EnergyDiff

            MAE += abs(EnergyDiff) * Weight
            #            print("Weight", Weight)
            #            print(System, abs(EnergyDiff) * Weight)
            MAEDen += 1.0  # Weight

    if len(WarningList) > 0:
        print("The following systems probably have errors:")
        print("- [ %s ]" % (" ".join("%s" % W for W in WarningList)))

    return MAE / MAEDen, Errors


Mode = Opts.Mode.upper()[:2]
if Mode == "GE":  # Generation mode - makes the .gif_ files
    WriteSystems(NFinal=NFinal)
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
                        system_name=system_name, NFinal=NFinal, Functional=Functional
                    )
                )
            print(f"GIF/{system_name}/calculate_system_energy_{Functional}.slurm")
        for non_nn in ["PBE", "XAlpha", "r2SCAN", "SCAN", "TPSS"]:
            with open(
                f"GIF/{system_name}/calculate_system_energy_{non_nn}.slurm", "w"
            ) as file:
                file.write(
                    script_template.format(
                        system_name=system_name, NFinal=NFinal, Functional=non_nn
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
else:
    MAE, Errors = ReadSystems(NFinal)
    print(
        "NFinal = %3d, NActual = %3d, WTMAD2 = %.3f" % (NFinal, len(list(Errors)), MAE)
    )
