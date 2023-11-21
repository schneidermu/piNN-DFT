import density_functional_approximation_dm21 as dm21
from pyscf import gto
from pyscf import dft
import dftd3.pyscf as disp

system_name = 'AHB21-6-6B'
functional_name = 'NN_PBE'

numint_dict = {
    'NN_PBE': dm21.NeuralNumInt(dm21.Functional.NN_PBE),
    'NN_XALPHA': dm21.NeuralNumInt(dm21.Functional.NN_XALPHA)
}

with open(f'GIF/{system_name}/{system_name}.gif_', 'r') as file:
    coords = ''

    lines = file.readlines()
    for line in lines[3:]:
        coords += ' '.join(line.split())+'\n'

    charge, spin = map(int, lines[2].split())

    coords = coords[:-1]
    spin = spin-1



molecule = gto.Mole()
molecule.atom = coords
molecule.basis = 'def2-qzvp'
molecule.verbose = 4
molecule.spin = spin
molecule.charge = charge
molecule.build()


mf = dft.UKS(molecule)
mf._numint = numint_dict[functional_name]
mf.conv_tol = 1E-6
mf.conv_tol_grad = 1E-3


energy = mf.kernel()
d3 = disp.DFTD3Dispersion(molecule, xc="PBE0", version="d3bj")

corrected_energy = energy + d3.kernel()[0]
print(corrected_energy)

with open(f'Results/EnergyList_150.txt', 'a') as file:
    file.write(f'{system_name}.gif_ {corrected_energy}\n')