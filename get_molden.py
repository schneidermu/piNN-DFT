from optparse import OptionParser
from pyscf import gto
import os
from pyscf import scf, lib, dft
from pyscf.tools import wfn_format
from pyscf.gto.basis import parse_gaussian

from density_functional_approximation_dm21.functional import NN_FUNCTIONAL


CALC_DIR = '/home/xray/schneiderm/den_mol_or/calc'
GRID_DIR = '/home/xray/schneiderm/den_mol_or/grids'


def main():
    lib.num_threads(4)
    parser = OptionParser()
    parser.add_option("--Molecule", type=str, help="Molecule formula")
    parser.add_option("--Functional", type="string", default="PBE0", help="Functional to calculate densities")

    (Opts, args) = parser.parse_args()
    molecule_name = Opts.Molecule
    functional = Opts.Functional


    # Initialize molecule
    mol = gto.Mole()
    mol.atom = f"./molden/{molecule_name}.xyz"
    mol.unit = 'B'

    mol.basis = {
        "H": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "H"),
        "B": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "B"),
        "C": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "C"),
        "O": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "O"),
        "F": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "F"),
        "N": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "N"),
        "Li": parse_gaussian.load("./aug-cc-pwcv5z.gbs", "Li")
    }
    mol.symmetry = False
    mol.spin = 0
    mol.charge = 0
    mol.build()
    mol.verbose = 4
    print(mol._atom)


    # Configure solver
    mf = scf.RKS(mol)
    if "NN" not in functional:
        mf.xc = functional
        functional += '_pyscf'
    else:
        model = NN_FUNCTIONAL(functional)
        mf.define_xc_(model.eval_xc, 'MGGA')

    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-6
    mf.chkfile = None
    mf.grids.level = 5
    mf.max_cycle = 25
    mf.run()

    if not mf.converged:
        with open("./non_converged_systems_density.log", "a") as file:
            file.write(f"{functional}-{molecule_name}\n")

    FUNCTIONAL_DIR = os.path.join(CALC_DIR, functional)
    if not os.path.exists(FUNCTIONAL_DIR):
        os.mkdir(FUNCTIONAL_DIR)
    MOLECULE_DIR = os.path.join(FUNCTIONAL_DIR, molecule_name)
    if not os.path.exists(MOLECULE_DIR):
        os.mkdir(MOLECULE_DIR)

    PBE0_DIR = os.path.join(GRID_DIR, f"grid_{molecule_name}")
    INPUT_DIR = os.path.join(MOLECULE_DIR, "gamess.wfn")
    OUTPUT_DIR = os.path.join(MOLECULE_DIR, "calc.out")

    print(f"Saving to {INPUT_DIR}")
    with open(INPUT_DIR, 'w') as file:
        wfn_format.write_mo(file, mol, mf.mo_coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ)

    command = f'echo "5 1 100 {PBE0_DIR} {os.path.join(MOLECULE_DIR, "rho")} 5 2 100 {PBE0_DIR} {os.path.join(MOLECULE_DIR, "grad")} 5 3 100 {PBE0_DIR} {os.path.join(MOLECULE_DIR, "lapl")} q" | tr " " "\n" | /home/xray/schneiderm/Multiwfn38/Multiwfn  {INPUT_DIR} &>> {OUTPUT_DIR}'
    os.system(command)

if __name__ == "__main__":
    main()


