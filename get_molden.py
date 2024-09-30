from optparse import OptionParser
from pyscf import gto
import density_functional_approximation_dm21 as dm21
import os
from pyscf import scf
from pyscf.tools import wfn_format


CALC_DIR = '/home/xray/schneiderm/den_mol_or/calc'
omega_str_list = ['0', '0076', '067', '18', '33', '50', '67', '82', '93', '99', '100']

func_dict = {f'NN_PBE_{omega}': dm21.NeuralNumInt(getattr(dm21.Functional, f'NN_PBE_{omega}')) for omega in omega_str_list}
func_dict.update({f'NN_XALPHA_{omega}': dm21.NeuralNumInt(getattr(dm21.Functional, f'NN_XALPHA_{omega}')) for omega in omega_str_list})



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
with open("./basis.nw", "r") as file:
    li_bas = file.read()
mol.basis = {
    "H": "aug-cc-pv5z",
    "B": "aug-cc-pwcv5z",
    "C": "aug-cc-pwcv5z",
    "O": "aug-cc-pwcv5z",
    "F": "aug-cc-pwcv5z",
    "N": "aug-cc-pwcv5z",
    "Li": gto.parse(li_bas)
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
    mf._numint = func_dict[functional]

mf.conv_tol = 1e-9
mf.conv_tol_grad = None
mf.run()


FUNCTIONAL_DIR = os.path.join(CALC_DIR, functional)
if not os.path.exists(FUNCTIONAL_DIR):
    os.mkdir(FUNCTIONAL_DIR)
MOLECULE_DIR = os.path.join(FUNCTIONAL_DIR, molecule_name)
if not os.path.exists(MOLECULE_DIR):
    os.mkdir(MOLECULE_DIR)

PBE0_DIR = os.path.join(CALC_DIR, "PBE0", molecule_name, "grid")
INPUT_DIR = os.path.join(MOLECULE_DIR, "gamess.wfn")
OUTPUT_DIR = os.path.join(MOLECULE_DIR, "calc.out")

print(f"Saving to {INPUT_DIR}")
with open(INPUT_DIR, 'w') as file:
    wfn_format.write_mo(file, mol, mf.mo_coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ)

command = f'echo "5 1 100 {PBE0_DIR} {os.path.join(MOLECULE_DIR, "rho")} 5 2 100 {PBE0_DIR} {os.path.join(MOLECULE_DIR, "grad")} 5 3 100 {PBE0_DIR} {os.path.join(MOLECULE_DIR, "lapl")} q" | tr " " "\n" | /home/xray/schneiderm/Multiwfn38/Multiwfn  {INPUT_DIR} &>> {OUTPUT_DIR}'
os.system(command)


