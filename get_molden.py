import os
from optparse import OptionParser

from pyscf import gto, lib, scf
from pyscf.gto.basis import parse_gaussian
from pyscf.tools import wfn_format

from density_functional_approximation_dm21.functional import NN_FUNCTIONAL


def main():
    lib.num_threads(4)
    parser = OptionParser()
    parser.add_option("--Molecule", type=str, help="Molecule formula", default=None)
    parser.add_option("--Atom", type=str, help="Atom name", default=None)
    parser.add_option("--Charge", type=int, help="Charge of the system", default=0)
    parser.add_option(
        "--Functional",
        type="string",
        default="PBE0",
        help="Functional to calculate densities",
    )

    (Opts, args) = parser.parse_args()
    molecule_name = Opts.Molecule
    atom_name = Opts.Atom
    charge = Opts.Charge
    functional = Opts.Functional

    # Initialize molecule
    mol = gto.Mole()
    if atom_name and molecule_name:
        raise Exception("Choose either molecule or atom")
    elif not (atom_name or molecule_name):
        raise Exception("System not provided")

    if molecule_name:
        mol.atom = f"./molden/{molecule_name}.xyz"
    else:
        mol.atom = f"{atom_name} 0 0 0"

    mol.unit = "B"

    mol.basis = {
        "H": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "H"),
        "B": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "B"),
        "C": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "C"),
        "O": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "O"),
        "F": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "F"),
        "N": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "N"),
        "Li": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "Li"),
        "Be": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "Be"),
        "Ne": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "Ne"),
    }
    mol.symmetry = False
    mol.spin = 0
    mol.charge = charge
    mol.build()
    mol.verbose = 4
    print(mol._atom)

    # Configure solver
    mf = scf.RKS(mol)
    if "NN" not in functional:
        mf.xc = functional
        functional += "_pyscf"
    else:
        model = NN_FUNCTIONAL(functional)
        mf.define_xc_(model.eval_xc, "MGGA")

    scf_data = {"latest_delta_e": None, "latest_g_norm": None}

    def log_convergence(env):
        scf_data["latest_delta_e"] = abs(env["e_tot"] - env["last_hf_e"])
        scf_data["latest_g_norm"] = env["norm_gorb"]

    mf.callback = log_convergence

    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-6
    mf.chkfile = None
    if molecule_name:
        mf.grids.level = 5
    else:
        mf.grids.atom_grid = (155, 974)
    mf.max_cycle = 25
    mf.run()

    if not mf.converged:
        latest_delta_e = scf_data["latest_delta_e"]
        latest_g_norm = scf_data["latest_g_norm"]

        if latest_delta_e is not None:
            with open("./non_converged_systems_density.log", "a") as file:
                log_line = (
                    f"{functional}-{molecule_name if molecule_name else atom_name}: Not converged. "
                    f"Last dE = {latest_delta_e:.2e}, |g| = {latest_g_norm:.2e}\n"
                )
                file.write(log_line)
                print(f"Logged: {log_line.strip()}")

    if molecule_name:
        CALC_DIR = "/home/xray/schneiderm/den_mol_or/calc"
        GRID_DIR = "/home/xray/schneiderm/den_mol_or/grids"
        FUNCTIONAL_DIR = os.path.join(CALC_DIR, functional)
        MOLECULE_DIR = os.path.join(FUNCTIONAL_DIR, molecule_name)
        PBE0_DIR = os.path.join(GRID_DIR, f"grid_{molecule_name}")
        INPUT_DIR = os.path.join(MOLECULE_DIR, "gamess.wfn")
        OUTPUT_DIR = os.path.join(MOLECULE_DIR, "calc.out")
        if not os.path.exists(MOLECULE_DIR):
            os.mkdir(MOLECULE_DIR)
    else:
        CALC_DIR = "/home/xray/schneiderm/denrho/dtestin"
        FUNCTIONAL_DIR = os.path.join(CALC_DIR, functional.replace("_pyscf", ""))
        INPUT_DIR = os.path.join(
            FUNCTIONAL_DIR,
            f"{atom_name}_+{charge}_{functional.replace('_pyscf', '')}.wfn",
        )

    if not os.path.exists(FUNCTIONAL_DIR):
        os.mkdir(FUNCTIONAL_DIR)

    print(f"Saving to {INPUT_DIR}")
    with open(INPUT_DIR, "w") as file:
        wfn_format.write_mo(
            file, mol, mf.mo_coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ
        )

    if molecule_name:
        command = f'echo "5 1 100 {PBE0_DIR} {os.path.join(MOLECULE_DIR, "rho")} 5 2 100 {PBE0_DIR} {os.path.join(MOLECULE_DIR, "grad")} 5 3 100 {PBE0_DIR} {os.path.join   (MOLECULE_DIR, "lapl")} q" | tr " " "\n" | /home/xray/schneiderm/Multiwfn38/Multiwfn  {INPUT_DIR} &>> {OUTPUT_DIR}'
        os.system(command)


if __name__ == "__main__":
    main()
