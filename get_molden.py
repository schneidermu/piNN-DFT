from optparse import OptionParser
from pyscf import dft, gto, lib
from pyscf.tools import molden
import density_functional_approximation_dm21 as dm21


func_dict = {
    "NN_PBE": dm21.NN_FUNCTIONAL("NN_PBE"),
#    "NN_XALPHA": dm21.NN_FUNCTIONAL("NN_XALPHA"),
}

parser = OptionParser()
parser.add_option("--Molecule", type=str, help="Molecule formula")
parser.add_option("--Functional", type="string", default="PBE0", help="Functional to calculate densities")

(Opts, args) = parser.parse_args()
molecule_name = Opts.Molecule
functional = Opts.Functional


mol = gto.Mole()
mol.atom = f"./molden/{molecule_name}.xyz"
mol.unit = 'B'
mol.spin = 0
mol.charge = 0
with open("./basis.nw", "r") as file:
    bas = file.read()
mol.basis = gto.basis.parse_nwchem.parse(bas)
mol.symmetry = False
mol.build()


mf = dft.RKS(mol)
if functional == 'PBE0':
    mf.xc = "PBE0"
else:
    mf._numint = func_dict[functional_name]


mf.conv_tol = 1e-9
mf.conv_tol_grad = None
mf.grids.level = 5
mf.kernel()


with open(f'./molden_results/{molecule_name}.molden', 'w') as f1:
    molden.header(mol, f1)
    molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)



