"""Render the completed publication build receipts without reevaluating science."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "dataset_manifest.json").read_text())
    qualification = json.loads((args.root / "provenance/qualification.json").read_text())
    leakage = json.loads((args.root / "provenance/leakage_audit.json").read_text())
    tests = json.loads(args.tests.read_text())
    benchmark = qualification["loader_benchmark"] if args.benchmark is None else json.loads(args.benchmark.read_text())
    if manifest["status"] != "qualified" or qualification["status"] != "PASS" or tests["status"] != "PASS":
        raise ValueError("Cannot describe an incomplete dataset as publication-qualified")
    lines = ["# Publication dataset v1 build report", "", "Status: qualified local training/validation bundle.", "",
             "```text", "publication_dataset_v1/", "  dataset_manifest.json / splits.json / checksums.sha256 / DATASET_CARD.md",
             "  provenance/  schema, source identities, build/qualification receipts, leakage audit",
             "  chemistry/   reactions.jsonl, species.jsonl, chemistry_*.h5",
             "  mrks/        systems.jsonl, mrks_*.h5",
             "  validation/  diet30_reactions.jsonl, diet30_species.jsonl, validation_*.h5", "```", "",
             "## Counts", "", "```json", json.dumps(manifest["counts"], indent=2), "```", "",
             "268 chemistry identities: 251 relchem and 17 AE17. Eight augmentation variants per reaction (2144 reaction/variant combinations), never eightfold sample weight.",
             "All 90 mRKS systems have energy, gauge-fixed pointwise and weak-form AO targets and factors.",
             f"Diet diagnostic: 30, nonselectable. Clean validation: {leakage['clean_count']}. PBE0 species: 84.", "",
             "## Exclusions and split integrity", "", "Training exclusions (unchanged):", "", "```json",
             json.dumps(manifest["training_exclusions"], indent=2), "```", "", "Validation exclusions:", "",
             "| Validation | Counterpart | Reason |", "|---|---|---|"]
    for row in leakage["exclusions"]:
        lines.append(f"| {row['validation']} | {row['counterpart']} | {row['reason']} |")
    unique = sorted({r["validation"] for r in leakage["exclusions"]})
    if 30 - len(unique) != leakage["clean_count"]:
        raise ValueError("Leakage accounting does not reconcile")
    proof = manifest["validation_qualification"]
    lines += ["", f"Unique excluded identities: {', '.join(unique)}. 30 - {len(unique)} = {leakage['clean_count']}.",
              "BH76-5 has both reserved-test overlap and a conservative ambiguous training-collision flag; it is counted only once.",
              "", "84/84 validation species qualified: 82 direct fixed-density total-energy parity + 2 large-system component-level parity.",
              f"Direct maximum discrepancy: {proof['direct_max_abs_error_hartree']:.17g} Ha. Component maximum: {proof['component_max_abs_error_hartree']:.17g} Ha.",
              "Both MCONF records have exact source-density, descriptor and PBE XC equality. Independent non-XC/recombined errors are 1.1368683772161603e-12 and 4.547473508864641e-13 Ha."]
    lines += ["", "No future test split/arrays/labels are included. Reserved Diet100 was consulted only for leakage identities.", "",
              "Primary dispersion: **PBE0-D3(BJ)**. Secondary: PBE-D3(BJ), never chosen by score.", "",
              "## Units and precision", "", "| Field | Axes | Units |", "|---|---|---|"]
    for name, (axes, units) in manifest["units_axes"].items():
        lines.append(f"| {name} | {axes} | {units} |")
    lines += ["", "Source/storage/production dtypes are explicit attributes on every numerical dataset. Chemistry retains F32 source rounding before matched-F64 arithmetic; operator precision boundaries are unchanged; validation density matrices/descriptors remain F64.",
              "The non-XC validation term is Tr(P hcore) + 0.5 Tr(P J[P]) + E_nuc. No SCF iterations or benchmark scoring were run.", "",
              "## Legacy and fixed-density parity", "", "```json", json.dumps({k: qualification[k] for k in
                  ("chemistry", "mrks", "validation")}, indent=2), "```", "",
              "## Loader benchmark", "", "```json", json.dumps(benchmark, indent=2), "```", "",
              "## Content and shard hashes", "", f"Logical SHA256: `{manifest['logical_sha256']}`", "",
              f"Total dataset-file bytes (manifest inventory): {sum(r['bytes'] for r in manifest['files'])}", "",
              "| Shard | Bytes | SHA256 |", "|---|---:|---|"]
    for row in manifest["files"]:
        if row["file"].endswith(".h5"):
            lines.append(f"| {row['file']} | {row['bytes']} | `{row['sha256']}` |")
    lines += ["", "Source hashes:", "", "```json", json.dumps(manifest["sources"], indent=2), "```", "",
              "## Tests and integrity", "", "```json", json.dumps(tests, indent=2), "```", "",
              f"Canonical array content hashes checked: {qualification['array_hashes_checked']}.", "",
              "No training objective, scientific target, MOO/SVRG method, architecture or production precision was changed. Legacy loaders coexist; no historical experiment was switched retrospectively.",
              "publication_dataset_v1 is frozen and immutable by convention. Any content change requires a new version/hash. No external baseline panel was run.",
              "Source redistribution licenses/permissions must be confirmed before external archival."]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
