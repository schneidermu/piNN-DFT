"""Build publication v1 in staging; publish only after loader qualification."""

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_models.publication_data.build import (
    chemistry,
    dump,
    mrks,
    schema,
    write_manifest,
)
from train_models.publication_data.contracts import file_sha
from train_models.publication_data.validation import (
    benchmark_rows,
    leakage,
    parallel_fixed_density,
    prepare_checkpoints,
    recover_validation,
    training_geometry_records,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "chemistry-source", "central", "ao-cache", "raw-checkpoints",
                 "checkpoint-tar", "d3-pbe0", "d3-pbe", "diet30", "reserved-identities"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume-staging", action="store_true")
    parser.add_argument("--minnesota-geometries", type=Path, required=True)
    parser.add_argument("--recover-validation-log", type=Path)
    args = parser.parse_args()
    root = Path(str(args.output) + ".staging")
    if args.output.exists():
        raise FileExistsError("Refusing to replace a published dataset")
    if root.exists() and not args.resume_staging:
        raise FileExistsError("Existing staging directory; request explicit resume")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    schema(root)
    repo = Path(__file__).resolve().parents[1]
    sources = {name: {"logical_name": path.name, "sha256": file_sha(path)} for name, path in {
        "chemistry_manifest": args.chemistry_source / "manifest.json",
        "central_manifest": args.central / "manifest.json", "ao_manifest": args.ao_cache / "manifest.json",
        "checkpoint_tar": args.checkpoint_tar, "d3_pbe0": args.d3_pbe0, "d3_pbe": args.d3_pbe,
        "diet30": args.diet30, "reserved_identities_only": args.reserved_identities,
        "minnesota_catalog": repo / "MN_dataset/total_dataframe_sorted_final.csv",
        "training_dispersion": repo / "train_models/dispersions/dispersions.pickle",
        "mrks_dispersion": repo / "train_models/dispersions/dispersions_mrks.pickle"}.items()}
    sources["diet30"]["url"] = "https://github.com/gambort/DietGMTKN55/blob/0eedf4d4136e55d245a25f6ac0a0e92ac7c0662d/GoodSamples/AllElements_030.yaml"
    dump(root / "provenance/sources.json", sources)
    counts = {}
    checkpoint_entries = prepare_checkpoints(args.checkpoint_tar, args.raw_checkpoints)
    dump(root / "provenance/checkpoint_archive_entries.json", checkpoint_entries)
    recovered = None
    completed = root / "provenance/validation_completed_species.jsonl"
    if completed.exists():
        recovered = [json.loads(line) for line in completed.read_text().splitlines()]
    elif args.recover_validation_log is not None:
        recovered = recover_validation(root, args.raw_checkpoints, benchmark_rows(args.diet30),
                                       args.d3_pbe0, args.d3_pbe, args.recover_validation_log)
    phases = {
        "chemistry": lambda: chemistry(root, args.chemistry_source, repo / "train_models/dispersions/dispersions.pickle"),
        "mrks": lambda: mrks(root, args.central, args.ao_cache, repo / "train_models/dispersions/dispersions_mrks.pickle"),
        "validation": lambda: parallel_fixed_density(root, args.raw_checkpoints, benchmark_rows(args.diet30),
                                                    args.d3_pbe0, args.d3_pbe, recovered),
    }
    for name, build in phases.items():
        receipt = root / f"provenance/{name}_conversion.json"
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value["sources"] != sources:
                raise ValueError("Staging source identity changed")
            for item in value["files"]:
                if file_sha(root / item["file"]) != item["sha256"]:
                    raise ValueError("Staging content changed")
            counts[name] = value["counts"]
        else:
            if any((root / name).glob("*.h5")) and not (name == "validation" and recovered is not None):
                raise ValueError(f"Interrupted incomplete {name}; preserve it for inspection")
            counts[name] = build()
            files = [{"file": p.relative_to(root).as_posix(), "sha256": file_sha(p)}
                     for p in sorted((root / name).iterdir()) if p.is_file()]
            dump(receipt, {"sources": sources, "counts": counts[name], "files": files})
    reactions = [json.loads(line) for line in (root / "chemistry/reactions.jsonl").read_text().splitlines()]
    geometries = training_geometry_records(reactions, args.minnesota_geometries)
    sources["minnesota_geometries"] = {"logical_name": args.minnesota_geometries.name,
                                      "sha256": file_sha(args.minnesota_geometries),
                                      "url": "https://comp.chem.umn.edu/db/dbs/tar/mn_databases.tar.gz"}
    dump(root / "provenance/sources.json", sources)
    dump(root / "provenance/training_geometry_identities.json", geometries)
    interface = args.diet30.parent / "InterfaceG16.py"
    sources["diet_benchmark_interface"] = {"logical_name": interface.name, "sha256": file_sha(interface),
        "url": "https://github.com/gambort/DietGMTKN55/blob/0eedf4d4136e55d245a25f6ac0a0e92ac7c0662d/InterfaceG16.py"}
    for path in sorted((repo / "train_models/publication_data").glob("*.py")):
        relative = path.relative_to(repo).as_posix()
        sources["tooling:" + relative] = {"logical_name": relative, "sha256": file_sha(path)}
    for relative in ("tools/build_publication_dataset.py", "tools/qualify_publication_dataset.py"):
        sources["tooling:" + relative] = {"logical_name": relative, "sha256": file_sha(repo / relative)}
    dump(root / "provenance/sources.json", sources)
    clean, audit = leakage(benchmark_rows(args.diet30), args.reserved_identities, reactions,
                           repo / "MN_dataset/total_dataframe_sorted_final.csv", geometries)
    dump(root / "provenance/leakage_audit.json", audit)
    validation = [json.loads(line) for line in (root / "validation/diet30_reactions.jsonl").read_text().splitlines()]
    systems = [json.loads(line) for line in (root / "mrks/systems.jsonl").read_text().splitlines()]
    splits = {"train_relchem": {"ids": [r["id"] for r in reactions if r["task"] == "relchem"]},
              "train_ae17": {"ids": [r["id"] for r in reactions if r["task"] == "ae17"]},
              "train_mrks": {"ids": [r["id"] for r in systems]},
              "diet30_diagnostic": {"ids": [r["id"] for r in validation], "selection_allowed": False},
              "diet30_clean_validation": {"ids": [r["id"] for r in validation if r["source_id"] in clean], "selection_allowed": True}}
    dump(root / "splits.json", splits)
    counts["validation"]["clean_reactions"] = len(clean)
    dump(root / "provenance/build_receipt.json", {"elapsed_seconds": time.perf_counter()-started,
         "status": "awaiting_loader_qualification", "no_training": True, "no_test_dataset": True,
         "historical_operator_preparation": "lap_all90_operator_preparation_metrics.json",
         "python_version": sys.version.split()[0],
         "library_versions": {name: importlib.metadata.version(name)
                              for name in ("numpy", "h5py", "torch", "pyscf", "scipy")}})
    (root / "DATASET_CARD.md").write_text(
        "# Laplacian functional publication dataset v1\n\n"
        "268 Minnesota chemical identities (251 relative chemistry, 17 AE17), each with eight grid augmentations. "
        "Grid variants never multiply reaction weights. 90 mRKS systems retain all E_xc, gauge-fixed pointwise and weak-form AO targets.\n\n"
        "Diet30 fixed PBE0 densities: diagnostic view is not selectable; clean validation excludes reserved/test and potential training overlaps. "
        "PBE0-D3(BJ) is primary; PBE-D3(BJ) is secondary. No test split or reserved test labels/arrays are packaged.\n\n"
        "Scientific units/dtypes are per-dataset attributes and provenance/schema.json. Chemistry preserves legacy F32 rounding before matched-F64 arithmetic. "
        "Operator precision remains the repaired stored-F32/learned-F64/PBE-F32/AO-F64 contract. "
        "Validation stores native F64 densities and precomputed grid descriptors. Its non-XC term is kinetic+nuclear attraction+Coulomb+nuclear repulsion, without any exchange/correlation.\n\n"
        "The training non-XC scalar is preserved from legacy ener[0]; raw Minnesota checkpoint/grid construction is not reconstructed by this conversion. "
        "mRKS Exc is the exact qualified legacy target, not substituted by another source field. "
        "Sources are hash-bound. Minnesota corpus originates from the project/University of Minnesota database; Diet definitions originate from gambort/DietGMTKN55. "
        "Source redistribution licenses/permissions must be confirmed before external archival; this local bundle does not invent a data license.\n")
    manifest = write_manifest(root, counts, sources)
    print(json.dumps({"staging": root.name, "logical_sha256": manifest["logical_sha256"], "counts": counts}), flush=True)


if __name__ == "__main__":
    main()
