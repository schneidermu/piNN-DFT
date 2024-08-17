from optparse import OptionParser
from pyscf import dft, gto
from pyscf.tools import molden
import density_functional_approximation_dm21 as dm21
from pyscf import lo


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
with open("./basis.nw", "r") as file:
    bas = file.read()
mol.basis = gto.basis.parse_nwchem.parse(bas)
mol.symmetry = False
mol.spin = 0
mol.charge = 0
mol.build()
mol.verbose = 4
print(mol._atom)


mf = dft.RKS(mol)
if functional == 'PBE0':
    mf.xc = "PBE0"
else:
    mf._numint = func_dict[functional]


mf.conv_tol = 1e-9
mf.conv_tol_grad = None
mf.grids.level = 5
mf.kernel()


c_loc_orth = lo.orth.orth_ao(mol)
molden.from_mo(mol, './molden_results/{molecule_name}.molden', c_loc_orth)



