from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py

try:
    import dftd3.pyscf as disp
    from pyscf import gto
except ImportError as exc:  # pragma: no cover - environment-specific dependency
    raise SystemExit(
        "This script requires both 'pyscf' and 'dftd3'. "
        "Use the same environment as the legacy dispersion workflow in test_models."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5_DIR = PROJECT_ROOT / "h5_vrho_from_mrks"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dispersions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute standalone PBE0-D3(BJ) dispersion energies for mRKS H5 systems."
    )
    parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-name", default="mrks_pbe0_d3bj.json")
    parser.add_argument("--csv-name", default="mrks_pbe0_d3bj.csv")
    return parser.parse_args()


def needs_ecp(atom_text: str) -> dict[str, str]:
    ecp_atoms = []
    for atom in ("I", "Sb", "Bi"):
        if atom in atom_text:
            ecp_atoms.append(atom)
    return {atom: "def2-qzvp" for atom in ecp_atoms}


def load_system_spec(h5_path: Path) -> tuple[str, int, int]:
    with h5py.File(h5_path, "r") as handle:
        atom_raw = handle["atom"][()]
        atom_text = atom_raw.decode("utf-8") if isinstance(atom_raw, bytes) else str(atom_raw)
        charge = int(handle["charge"][()])
        spin = int(handle["spin"][()])
    return atom_text, charge, spin


def build_molecule(atom_text: str, charge: int, spin: int) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = atom_text
    # mRKS H5 files store nuclear coordinates in bohr.
    # Example sanity check: H2 distance in the archive is ~1.402, which is the
    # equilibrium bond length in bohr, not angstrom.
    mol.unit = "Bohr"
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False
    mol.verbose = 0
    # No SCF is run; the Mole object is only used as a geometry/charge carrier for D3(BJ).
    mol.basis = "def2-qzvp"
    mol.ecp = needs_ecp(atom_text)
    mol.build()
    return mol


def compute_pbe0_d3bj(h5_path: Path) -> float:
    atom_text, charge, spin = load_system_spec(h5_path)
    mol = build_molecule(atom_text, charge, spin)
    d3 = disp.DFTD3Dispersion(mol, xc="PBE0", version="d3bj")
    return float(d3.kernel()[0])


def main() -> None:
    args = parse_args()
    h5_dir = args.h5_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        raise SystemExit(f"No .h5 files found in {h5_dir}")

    results: dict[str, float] = {}
    for h5_path in h5_files:
        system_name = h5_path.stem
        energy = compute_pbe0_d3bj(h5_path)
        results[system_name] = energy
        print(f"{system_name}: {energy:.12f} Ha")

    json_path = output_dir / args.json_name
    csv_path = output_dir / args.csv_name

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "pbe0_d3bj_ha"])
        for system_name, energy in sorted(results.items()):
            writer.writerow([system_name, f"{energy:.12f}"])

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
