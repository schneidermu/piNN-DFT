from optparse import OptionParser

parser = OptionParser()
parser.add_option("--NFinal", type=int, default=50, help="Number systems to select")

(Opts, args) = parser.parse_args()

NFinal = Opts.NFinal


paths = {
    "PBE": ("Results/EnergyList_{NFinal}_PBE.txt", "Results/DispersionList_PBE.txt"),
    "XAlpha": (
        "Results/EnergyList_{NFinal}_XAlpha.txt",
        "Results/DispersionList_PBE0.txt",
    ),
}

for functional in ["PBE", "XAlpha"]:
    energy_dict = dict()
    energy_path, dispersion_path = paths[functional]
    for path in energy_path, dispersion_path:
        with open(path, "r") as file:
            for line in file:
                system_name, energy = line.strip().split()
                system_name = system_name.split(".gif_")[0]
                energy = float(energy)
                if energy_dict.get(system_name) is None:
                    energy_dict[system_name] = [
                        energy,
                    ]
                else:
                    energy_dict[system_name].append(energy)
    with open(f"Results/EnergyList_{NFinal}_{functional}_D3BJ.txt", "w") as file:
        for system in energy_dict:
            file.write(f"{system}.gif_ {sum(energy_dict[system])}\n")
