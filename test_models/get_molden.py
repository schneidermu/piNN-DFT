import os
from optparse import OptionParser
import shutil
import subprocess
from pathlib import Path

from DFT.functional import NN_FUNCTIONAL
from DFT.numint import RKS_with_Laplacian
from pcNN_mol.dft_pcnn import model as Nagai_model
from pyscf import gto, lib, scf
from pyscf.gto.basis import parse_gaussian
from pyscf.scf import diis
from pyscf.tools import wfn_format

from common import GBS_PATH, MOLDEN_DIR, TEST_MODELS_ROOT

PROBLEMATIC_SYSTEMS = [
    "Li2",
    "LiH",
    "H2",
    "C +2",
    "C +4",
    "N +3",
    "Be +0",
    "B +1",
    "B +3",
    "O +4",
    "O +6",
    "Ne +6",
    "F +5",
]

DEFAULT_MULTIWFN_CMD = os.environ.get("MULTIWFN_CMD", "Multiwfn")


def resolve_multiwfn_cmd(cmd: str) -> str:
    path = Path(cmd).expanduser()
    if path.parent != Path("."):
        if path.exists():
            return str(path)
    elif shutil.which(cmd):
        return cmd
    raise FileNotFoundError(
        "Multiwfn executable was not found. Install Multiwfn on this node, add it to "
        "PATH, set MULTIWFN_CMD=/path/to/Multiwfn, or pass "
        "--MultiwfnCmd /path/to/Multiwfn when generating/running molden jobs."
    )

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
    parser.add_option("--ExperimentRoot", type=str, default="")
    parser.add_option("--CheckpointPath", type=str, default="")
    parser.add_option("--ModelKey", type=str, default="")
    parser.add_option("--MultiwfnCmd", type=str, default=DEFAULT_MULTIWFN_CMD)

    (Opts, args) = parser.parse_args()
    molecule_name = Opts.Molecule
    atom_name = Opts.Atom
    charge = Opts.Charge
    functional = Opts.Functional
    experiment_root = Path(Opts.ExperimentRoot).resolve() if Opts.ExperimentRoot else None
    checkpoint_path = Opts.CheckpointPath or None
    model_key = Opts.ModelKey or None
    multiwfn_cmd = Opts.MultiwfnCmd

    # Initialize molecule
    mol = gto.Mole()
    if atom_name and molecule_name:
        raise Exception("Choose either molecule or atom")
    elif not (atom_name or molecule_name):
        raise Exception("System not provided")

    if molecule_name:
        multiwfn_cmd = resolve_multiwfn_cmd(multiwfn_cmd)
        mol.atom = str(MOLDEN_DIR / f"{molecule_name}.xyz")
    else:
        mol.atom = f"{atom_name} 0 0 0"

    mol.unit = "B"

    mol.basis = {
        "H": parse_gaussian.load(str(GBS_PATH), "H"),
        "B": parse_gaussian.load(str(GBS_PATH), "B"),
        "C": parse_gaussian.load(str(GBS_PATH), "C"),
        "O": parse_gaussian.load(str(GBS_PATH), "O"),
        "F": parse_gaussian.load(str(GBS_PATH), "F"),
        "N": parse_gaussian.load(str(GBS_PATH), "N"),
        "Li": parse_gaussian.load(str(GBS_PATH), "Li"),
        "Be": parse_gaussian.load(str(GBS_PATH), "Be"),
        "Ne": parse_gaussian.load(str(GBS_PATH), "Ne"),
    }
    mol.symmetry = False
    mol.spin = 0
    mol.charge = charge
    mol.build()
    mol.verbose = 4
    print(mol._atom)

    if (model_key and "PBE-L" in model_key) or "PBE-L" in functional:
        mf = RKS_with_Laplacian(mol)
    else:
        mf = scf.RKS(mol)

    use_checkpoint_functional = bool(checkpoint_path or model_key)

    if functional == "Nagai":
        mf.define_xc_(Nagai_model.eval_xc, "MGGA")
    elif use_checkpoint_functional or "NN" in functional:
        model = NN_FUNCTIONAL(
            functional,
            checkpoint_path=checkpoint_path,
            model_key=model_key,
        )
        mf.define_xc_(model.eval_xc, "MGGA")
    else:
        mf.xc = functional
        functional += "_pyscf"

    scf_data = {"latest_delta_e": None, "latest_g_norm": None}

    def log_convergence(env):
        scf_data["latest_delta_e"] = abs(env["e_tot"] - env["last_hf_e"])
        scf_data["latest_g_norm"] = env["norm_gorb"]

    mf.callback = log_convergence

    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-6
    mf.max_cycle = 50

    mf.chkfile = None

    if molecule_name:
        mf.grids.level = 5
    else:
        mf.grids.atom_grid = (155, 974)

    if "XALPHA" in functional and (
        molecule_name in PROBLEMATIC_SYSTEMS
        or f"{atom_name} +{charge}" in PROBLEMATIC_SYSTEMS
    ):

        mf.conv_tol = 1e-6
        mf.conv_tol_grad = 1e-3

        mf.level_shift = 0.5

        mf.damp = 0.5

        mf.diis = diis.EDIIS()

        mf.diis.space = 12

        mf.diis_start_cycle = 10

        mf.max_cycle = 100

    mf.run()

    if not mf.converged:
        latest_delta_e = scf_data["latest_delta_e"]
        latest_g_norm = scf_data["latest_g_norm"]

        if latest_delta_e is not None:
            with open(TEST_MODELS_ROOT / "non_converged_systems_density.log", "a") as file:
                log_line = (
                    f"{functional}-{molecule_name if molecule_name else atom_name}: Not converged. "
                    f"Last dE = {latest_delta_e:.2e}, |g| = {latest_g_norm:.2e}\n"
                )
                file.write(log_line)
                print(f"Logged: {log_line.strip()}")

    if molecule_name:
        if experiment_root:
            calc_root = experiment_root / "outputs" / "avrane" / "den_mol_or" / "calc"
            grid_root = TEST_MODELS_ROOT.parent / "den_mol_or" / "grids"
        else:
            calc_root = TEST_MODELS_ROOT.parent / "den_mol_or" / "calc"
            grid_root = TEST_MODELS_ROOT.parent / "den_mol_or" / "grids"
        CALC_DIR = calc_root
        GRID_DIR = grid_root
        FUNCTIONAL_DIR = os.path.join(CALC_DIR, functional)
        MOLECULE_DIR = os.path.join(FUNCTIONAL_DIR, molecule_name)
        PBE0_DIR = os.path.join(GRID_DIR, f"grid_{molecule_name}")
        INPUT_DIR = os.path.join(MOLECULE_DIR, "gamess.wfn")
        OUTPUT_DIR = os.path.join(MOLECULE_DIR, "calc.out")
        os.makedirs(MOLECULE_DIR, exist_ok=True)
    else:
        if experiment_root:
            CALC_DIR = experiment_root / "outputs" / "avrane" / "denrho" / "dtestin"
        else:
            CALC_DIR = TEST_MODELS_ROOT.parent / "denrho" / "dtestin"
        FUNCTIONAL_DIR = os.path.join(CALC_DIR, functional.replace("_pyscf", ""))
        INPUT_DIR = os.path.join(
            FUNCTIONAL_DIR,
            f"{atom_name}_+{charge}_{functional.replace('_pyscf', '')}.wfn",
        )

    if not os.path.exists(FUNCTIONAL_DIR):
        os.makedirs(FUNCTIONAL_DIR, exist_ok=True)

    print(f"Saving to {INPUT_DIR}")
    with open(INPUT_DIR, "w") as file:
        wfn_format.write_mo(
            file, mol, mf.mo_coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ
        )
    if not os.path.exists(INPUT_DIR) or os.path.getsize(INPUT_DIR) == 0:
        raise FileNotFoundError(f"Failed to write non-empty WFN file: {INPUT_DIR}")

    if molecule_name:
        multiwfn_input = "\n".join(
            [
                "5",
                "1",
                "100",
                str(PBE0_DIR),
                os.path.join(MOLECULE_DIR, "rho"),
                "5",
                "2",
                "100",
                str(PBE0_DIR),
                os.path.join(MOLECULE_DIR, "grad"),
                "5",
                "3",
                "100",
                str(PBE0_DIR),
                os.path.join(MOLECULE_DIR, "lapl"),
                "q",
                "",
            ]
        )
        with open(OUTPUT_DIR, "a") as output_file:
            subprocess.run(
                [multiwfn_cmd, os.path.basename(INPUT_DIR)],
                input=multiwfn_input,
                text=True,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                cwd=MOLECULE_DIR,
                check=True,
            )


if __name__ == "__main__":
    main()
