from optparse import OptionParser

import dftd3.pyscf as disp
from DFT.functional import NN_FUNCTIONAL
from DFT.numint import RKS_with_Laplacian, UKS_with_Laplacian
from pcNN_mol.dft_pcnn import model as Nagai_model
from pyscf import dft, gto, lib
from pyscf.scf import addons, diis

PROBLEMATIC_SYSTEMS = [
    "G21EA-14-EA_14",
    "G21EA-14-EA_14n",
    "W4-11-57-b",
    "MB16-43-10-10",
    "BH76-5-hclhts",
    "MB16-43-10-BH3",
    "HEAVY28-16-sbh3",
    "W4-11-57-f",
    "W4-11-30-cl",
    "W4-11-132-cl",
]


def get_coords_charge_spin(system_name):
    with open(f"GIF/{system_name}/{system_name}.gif_", "r") as file:
        coords = ""

        lines = file.readlines()
        for line in lines[3:]:
            coords += " ".join(line.split()) + "\n"

        charge, spin = map(int, lines[2].split())

        coords = coords[:-1]
        spin = spin - 1
    return coords, charge, spin


def initialize_molecule(coords, charge, spin, lapl=False):
    ecp_atoms = []
    molecule = gto.Mole()
    molecule.atom = coords
    if "I" in coords:
        ecp_atoms.append("I")
    if "Sb" in coords:
        ecp_atoms.append("Sb")
    if "Bi" in coords:
        ecp_atoms.append("Bi")

    molecule.ecp = {atom: "def2-qzvp" for atom in ecp_atoms}
    molecule.basis = "def2-qzvp"
    molecule.verbose = 4
    molecule.spin = spin
    molecule.charge = charge
    molecule.symmetry = False
    molecule.build()

    if spin == 0:
        if not lapl:
            mf = dft.RKS(molecule)
        else:
            mf = RKS_with_Laplacian(molecule)
    else:
        if not lapl:
            mf = dft.UKS(molecule)
        else:
            mf = UKS_with_Laplacian(molecule)

    mf.max_cycle = 25

    return molecule, mf


def get_PBE0_density(mf):
    mf.xc = "PBE0"
    mf.run()
    dm0 = mf.make_rdm1()

    return mf, dm0


def calculate_functional_energy(mf, functional_name, dm0=None, system_name=None):
    print(functional_name)

    if functional_name == "Nagai":
        mf.define_xc_(Nagai_model.eval_xc, "MGGA")
    else:
        model = NN_FUNCTIONAL(functional_name)
        mf.define_xc_(model.eval_xc, "MGGA")
    mf.conv_tol = 1e-6
    mf.conv_tol_grad = 1e-3

    scf_data = {"latest_delta_e": None, "latest_g_norm": None}

    def log_convergence(env):
        scf_data["latest_delta_e"] = abs(env["e_tot"] - env["last_hf_e"])
        scf_data["latest_g_norm"] = env["norm_gorb"]

    mf.callback = log_convergence

    if system_name in PROBLEMATIC_SYSTEMS and "XALPHA" in functional_name:

        mf.level_shift = 1.0
        mf.damp = 0.7
        mf.diis = diis.EDIIS()
        mf.diis.space = 12
        mf.diis_start_cycle = 25
        mf.max_cycle = 100
        mf.conv_tol = 1e-5
        mf.conv_tol_grad = 1e-3

        energy = mf.kernel()

    else:
        energy = mf.kernel(dm0=dm0)

    if not mf.converged:
        latest_delta_e = scf_data["latest_delta_e"]
        latest_g_norm = scf_data["latest_g_norm"]

        if latest_delta_e is not None:
            with open("./non_converged_systems_gmtkn55.log", "a") as file:
                log_line = (
                    f"{functional}-{system_name}: Not converged. "
                    f"Last dE = {latest_delta_e:.2e}, |g| = {latest_g_norm:.2e}\n"
                )
                file.write(log_line)
                print(f"Logged: {log_line.strip()}")

    d3 = disp.DFTD3Dispersion(mf.mol, xc="PBE0", version="d3bj")
    d3_energy = d3.kernel()[0]

    if functional_name == "Nagai":
        d3_energy = 0  # D3 is not used in Nagai et al. paper

    return energy + d3_energy


def calculate_non_nn_functional_energy(mf, functional_name):
    mf.xc = functional_name
    mf.conv_tol = 1e-6
    mf.conv_tol_grad = 1e-3
    scf_data = {"latest_delta_e": None, "latest_g_norm": None}

    def log_convergence(env):
        scf_data["latest_delta_e"] = abs(env["e_tot"] - env["last_hf_e"])
        scf_data["latest_g_norm"] = env["norm_gorb"]

    mf.callback = log_convergence

    energy = mf.kernel()

    if not mf.converged:
        latest_delta_e = scf_data["latest_delta_e"]
        latest_g_norm = scf_data["latest_g_norm"]

        if latest_delta_e is not None:
            with open("./non_converged_systems_gmtkn55.log", "a") as file:
                log_line = (
                    f"{functional}-{system_name}: Not converged. "
                    f"Last dE = {latest_delta_e:.2e}, |g| = {latest_g_norm:.2e}\n"
                )
                file.write(log_line)
                print(f"Logged: {log_line.strip()}")

    d3 = disp.DFTD3Dispersion(mf.mol, xc=functional_name, version="d3bj")
    d3_energy = d3.kernel()[0]

    return energy + d3_energy


def main(system_name, functional, NFinal):

    lib.num_threads(4)
    print("\n\n", system_name, "\n\n")
    coords, charge, spin = get_coords_charge_spin(system_name)

    lapl = False
    if "PBE-L" in functional:
        lapl = True

    _, mf = initialize_molecule(coords, charge, spin, lapl=lapl)
    dm0 = None

    mf.chkfile = None

    print(f"\n\n{functional} calculation \n\n")
    try:
        corrected_energy = calculate_functional_energy(
            mf, functional, dm0=dm0, system_name=system_name
        )
    except Exception as E:
        print(E)
        corrected_energy = "ERROR"
    finally:
        with open(f"Results/EnergyList_{NFinal}_{functional}.txt", "a") as file:
            file.write(f"{system_name}.gif_ {corrected_energy}\n")


def test_non_nn_functional(system_name, non_nn_functional, NFinal):
    print("Number of threads:", lib.num_threads())
    print("\n\n", system_name, "\n\n")
    coords, charge, spin = get_coords_charge_spin(system_name)

    _, mf = initialize_molecule(coords, charge, spin)

    energy = calculate_non_nn_functional_energy(mf, non_nn_functional)

    with open(f"Results/EnergyList_{NFinal}_{functional}.txt", "a") as file:
        file.write(f"{system_name}.gif_ {energy}\n")


def calculate_dispersions():
    print("Number of threads:", lib.num_threads())
    print("\n\n", system_name, "\n\n")
    coords, charge, spin = get_coords_charge_spin(system_name)
    molecule, mf = initialize_molecule(coords, charge, spin)
    d3_PBE0 = disp.DFTD3Dispersion(molecule, xc="PBE0", version="d3bj")
    d3_PBE = disp.DFTD3Dispersion(molecule, xc="PBE", version="d3bj")
    d3_PBE0_energy = d3_PBE0.kernel()[0]
    d3_PBE_energy = d3_PBE.kernel()[0]
    with open(f"Results/DispersionList_PBE0.txt", "a") as file:
        file.write(f"{system_name}.gif_ {d3_PBE0_energy}\n")
    with open(f"Results/DispersionList_PBE.txt", "a") as file:
        file.write(f"{system_name}.gif_ {d3_PBE_energy}\n")


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--Functional", type=str, default="NN_PBE_0", help="Functional for calculation"
    )
    parser.add_option(
        "--Dispersion", type=str, default=False, help="D3 Dispersion calculation"
    )
    parser.add_option("--System", type=str, help="System to calculate")
    parser.add_option(
        "--NFinal", type=int, default=30, help="Number of systems to select"
    )

    (Opts, args) = parser.parse_args()

    system_name = Opts.System
    functional = Opts.Functional
    dispersion = Opts.Dispersion
    NFinal = Opts.NFinal

    if dispersion:
        calculate_dispersions()
    elif "NN" in functional or functional == "Nagai":
        main(system_name, functional, NFinal)
    else:
        test_non_nn_functional(system_name, functional, NFinal)
