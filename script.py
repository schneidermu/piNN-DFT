import density_functional_approximation_dm21 as dm21
from pyscf import gto
from pyscf import dft
from pyscf import lib
import dftd3.pyscf as disp
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--System', type=str,
                  help="System to calculate")
parser.add_option('--Functional', type=str,
                  default='NN',
                  help="Non-NN functional")

(Opts, args) = parser.parse_args()


numint_dict = {
    'NN_PBE': dm21.NeuralNumInt(dm21.Functional.NN_PBE),
    'NN_XALPHA': dm21.NeuralNumInt(dm21.Functional.NN_XALPHA)
}
non_nn_functional_dict = {
    'PBE':'PBE',
    'XAlpha':'1.05*lda,'
}

system_name = Opts.System
functional = Opts.Functional

non_nn_functional = non_nn_functional_dict.get(functional)



def get_coords_charge_spin(system_name):
    with open(f'GIF/{system_name}/{system_name}.gif_', 'r') as file:
            coords = ''

            lines = file.readlines()
            for line in lines[3:]:
                coords += ' '.join(line.split())+'\n'

            charge, spin = map(int, lines[2].split())

            coords = coords[:-1]
            spin = spin-1
    return coords, charge, spin


def initialize_molecule(coords, charge, spin):
    molecule = gto.Mole()
    molecule.atom = coords
    if "I" in coords:
        molecule.ecp = {"I": 'def2-qzvp'}
    if "Sb" in coords:
        molecule.ecp = {"Sb": 'def2-qzvp'}
    molecule.basis = 'def2-qzvp'
    molecule.verbose = 4
    molecule.spin = spin
    molecule.charge = charge
    molecule.build()

    mf = dft.UKS(molecule)

    return molecule, mf


def get_PBE0_density(mf):
    mf.xc = 'PBE0'
    mf.run()
    dm0 = mf.make_rdm1()

    return mf, dm0


def calculate_functional_energy(mf, functional_name, d3_energy, dm0):
    mf._numint = numint_dict[functional_name]
    mf.conv_tol = 1E-6
    mf.conv_tol_grad = 1E-3

    energy = mf.kernel(dm0=dm0)

    return energy + d3_energy


def calculate_non_nn_functional_energy(mf, functional_name):
    mf.xc = functional_name
    energy = mf.kernel()
    return energy



def main():
    print('Number of threads:', lib.num_threads())
    print('\n\n', system_name, '\n\n')
    coords, charge, spin = get_coords_charge_spin(system_name)

    molecule, mf = initialize_molecule(coords, charge, spin)

    print('\n\nInitial PBE0 calculation\n\n')
    mf, dm0 = get_PBE0_density(mf)

    d3 = disp.DFTD3Dispersion(molecule, xc="PBE0", version="d3bj")
    d3_energy = d3.kernel()[0]

    for functional_name in numint_dict:
        print(f'\n\n{functional_name} calculation \n\n')
        try:
            corrected_energy = calculate_functional_energy(mf, functional_name, d3_energy, dm0)

        except Exception:
            corrected_energy = 'ERROR'

        finally:
            with open(f'Results/EnergyList_30_{functional_name}.txt', 'a') as file:
                file.write(f'{system_name}.gif_ {corrected_energy}\n')

def test_non_nn_functional():
    print('Number of threads:', lib.num_threads())
    print('\n\n', system_name, '\n\n')
    coords, charge, spin = get_coords_charge_spin(system_name)

    molecule, mf = initialize_molecule(coords, charge, spin)

    energy = calculate_non_nn_functional_energy(mf, non_nn_functional)

    with open(f'Results/EnergyList_30_{functional}.txt', 'a') as file:
        file.write(f'{system_name}.gif_ {energy}\n')


if __name__ == '__main__':
    if functional == 'NN':
        main()
    else:
        test_non_nn_functional()
