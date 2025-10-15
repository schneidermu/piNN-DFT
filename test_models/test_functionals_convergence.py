from optparse import OptionParser

from DFT.functional import NN_FUNCTIONAL
from script import get_coords_charge_spin, initialize_molecule


def calculate_functional_energy(mf, functional_name):

    print(functional_name)

    model = NN_FUNCTIONAL(functional_name)
    mf.define_xc_(model.eval_xc, "MGGA")
    mf.conv_tol = 1e-15
    mf.max_cycle = 50

    energy = mf.kernel()

    return energy


def main(functional_name):

    coords, charge, spin = get_coords_charge_spin("DC13-1-ISO_P36")
    _, mf = initialize_molecule(coords, charge, spin)
    calculate_functional_energy(mf, functional_name)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option(
        "--Functional", type="string", default="NN_PBE_0", help="Functional to evaluate"
    )
    (Opts, args) = parser.parse_args()

    Functional = Opts.Functional

    main(functional_name=Functional)
