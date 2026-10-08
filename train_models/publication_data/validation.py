"""Diet identity audit and fixed-density preprocessing (never SCF)."""

from pathlib import Path

import h5py
import numpy as np

from .build import Shards, dump, jsonl, portable, put
from .contracts import canonical, canonical_id, file_sha, parse_d3


def prepare_checkpoints(archive, raw):
    """Safe extraction or bytewise verification against the supplied archive."""
    import hashlib
    import tarfile
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    checked, checkpoint_count, marker_count = [], 0, 0
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if member.isdir():
                continue
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe checkpoint archive entry")
            target = raw / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            source = handle.extractfile(member)
            if target.exists():
                for block in iter(lambda source=source: source.read(8*1024**2), b""):
                    digest.update(block)
                if digest.hexdigest() != file_sha(target):
                    raise ValueError("Extracted checkpoint differs from archive")
            else:
                with target.open("xb") as output:
                    for block in iter(lambda source=source: source.read(8*1024**2), b""):
                        output.write(block)
                        digest.update(block)
            checked.append({"file": relative.as_posix(), "sha256": digest.hexdigest()})
            checkpoint_count += member.name.endswith(".pbe0.chk")
            marker_count += member.name.endswith(".pbe0.chk.complete")
    if checkpoint_count != 84 or marker_count != 84:
        raise ValueError("Exactly 84 checkpoints and complete markers required")
    return checked


def geometry_fingerprint(species):
    positions = np.asarray(species["Positions"], dtype=np.float64)
    elements = species["Elements"]
    if positions.shape != (len(elements), 3) or not np.isfinite(positions).all():
        raise ValueError("Invalid benchmark geometry")
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    local = sorted([element, sorted([elements[j], round(float(distances[i, j]), 5)]
                                   for j in range(len(elements))) ] for i, element in enumerate(elements))
    return canonical_id("geometry", [int(species["Charge"]), int(species["UHF"]), local])


def reaction_fingerprint(reaction):
    terms = {}
    for species in reaction["Species"].values():
        key = geometry_fingerprint(species)
        terms[key] = terms.get(key, 0.) + float(species["Count"])
    rows = sorted([key, value] for key, value in terms.items() if value)
    if not rows:
        raise ValueError("Degenerate zero reaction identity")
    scale = min(abs(value) for _, value in rows)
    forward = [[key, value/scale] for key, value in rows]
    reverse = [[key, -value/scale] for key, value in rows]
    return canonical_id("chemical_reaction", min(forward, reverse))


def training_geometry_records(training, archive):
    """Map catalog source names to official Minnesota XYZ/G09 identities."""
    import hashlib
    import re
    import tarfile
    records = {}
    names = sorted({c for row in training for c in row["components"]})
    with tarfile.open(archive) as handle:
        members = {m.name.lower(): m for m in handle.getmembers() if m.isfile() and "/._" not in m.name}
        for name in names:
            label, database = name.rsplit("_", 1)
            folder = {"ip13": "IP21", "abde4": "ABDE12"}.get(database.lower(), database)
            xyz_key = f"MN_databases/xyz/{folder}/{label}.xyz".lower()
            g09_key = f"MN_databases/g09inp/{folder}/{label}.g09".lower()
            if xyz_key not in members or g09_key not in members:
                raise ValueError(f"Missing authoritative training geometry: {name}")
            xyz_bytes = handle.extractfile(members[xyz_key]).read()
            g09_bytes = handle.extractfile(members[g09_key]).read()
            xyz = xyz_bytes.decode().splitlines()
            n = int(xyz[0])
            atoms = [line.split() for line in xyz[2:2+n]]
            charge_spin = re.search(r"^\s*(-?\d+)[\s,]+(\d+)\s*$", g09_bytes.decode(), re.MULTILINE)
            if len(atoms) != n or charge_spin is None:
                raise ValueError("Invalid official XYZ/G09 identity")
            records[name] = {"Elements": [a[0] for a in atoms],
                             "Positions": [[float(v) for v in a[1:4]] for a in atoms],
                             "Charge": int(charge_spin[1]), "UHF": int(charge_spin[2])-1,
                             "xyz_source": members[xyz_key].name, "g09_source": members[g09_key].name,
                             "xyz_sha256": hashlib.sha256(xyz_bytes).hexdigest(),
                             "g09_sha256": hashlib.sha256(g09_bytes).hexdigest()}
    return records


def composition_signature(reaction):
    """Only a collision screen: composition never establishes equality."""
    from collections import Counter
    scale = min(abs(float(s["Count"])) for s in reaction["Species"].values())
    rows = sorted([sorted(Counter(s["Elements"]).items()), int(s["Charge"]), int(s["UHF"]),
                   float(s["Count"])/scale] for s in reaction["Species"].values())
    reverse = sorted([*row[:3], -row[3]] for row in rows)
    return canonical(min(rows, reverse))


def benchmark_rows(path):
    import yaml
    source = yaml.safe_load(Path(path).read_text())
    return [(db, int(rid), reaction) for db, rows in sorted(source.items())
            for rid, reaction in sorted(rows.items())]


def leakage(diet30, reserved, training, csv_path, training_geometries):
    """Reserved catalog is consulted only for identity, never packaged/evaluated."""
    import csv
    fingerprints = {}
    for db, rid, reaction in benchmark_rows(reserved):
        fingerprints.setdefault(reaction_fingerprint(reaction), []).append((db, rid, reaction))
    exclusions, clean, comparisons = [], [], []
    for db, rid, reaction in diet30:
        source_id = f"{db}-{rid}"
        matches = fingerprints.get(reaction_fingerprint(reaction), [])
        for other_db, other_rid, other in matches:
            exclusions.append({"validation": source_id, "counterpart": f"Diet100:{other_db}-{other_rid}",
                               "reason": "identical stoichiometry + charge/spin + geometry fingerprint",
                               "reference_energy_agrees": reaction["Energy"] == other["Energy"],
                               "benchmark_weight_agrees": reaction["Weight"] == other["Weight"]})
        comparisons.append({"validation": source_id, "fingerprint": reaction_fingerprint(reaction),
                            "reserved_matches": len(matches)})
        if not matches:
            clean.append(source_id)
    required = {("G21EA-25", "Diet100:G21EA-25"), ("BH76-5", "Diet100:BH76-6")}
    actual = {(r["validation"], r["counterpart"]) for r in exclusions}
    if not required <= actual:
        raise ValueError("Known reserved aliases were not detected")
    # Actual Minnesota source definitions, not a filename-derived reaction ID.
    with Path(csv_path).open(encoding="cp1251") as stream:
        reader = csv.DictReader(stream)
        identity_column, db_column = reader.fieldnames[:2]
        source_rows = {(r[db_column], int(r[identity_column])):
                       {k: float(v) for k, v in r.items() if k not in (identity_column, db_column) and float(v) != 0}
                       for r in reader}
    identities = {(r["database"], r["reaction_id"]) for r in training}
    for reaction in training:
        expected = dict(zip(reaction["components"], reaction["coefficients"]))
        if source_rows[(reaction["database"], reaction["reaction_id"])] != expected:
            raise ValueError("Training source stoichiometry mismatch")
    training_checks = []
    benchmark_identities = [(db, rid, reaction_fingerprint(r), composition_signature(r)) for db, rid, r in diet30]
    for row in training:
        definition = {"Species": {name: {**training_geometries[name], "Count": count}
                                   for name, count in zip(row["components"], row["coefficients"])}}
        geometry_identity = reaction_fingerprint(definition)
        composition_identity = composition_signature(definition)
        for db, rid, fingerprint, composition in benchmark_identities:
            exact = geometry_identity == fingerprint
            possible = composition_identity == composition
            if not exact and not possible:
                continue
            name = f"{db}-{rid}"
            counterpart = f"Minnesota:{row['database']}-{row['reaction_id']}"
            reason = ("identical source stoichiometry, charge/spin and official geometry fingerprint" if exact else
                      "composition/charge/spin/stoichiometry collision; differing geometries do not prove distinct chemistry; conservatively withheld")
            entry = {"validation": name, "counterpart": counterpart, "reason": reason,
                     "exact_geometry_match": exact, "identity_status": "overlap" if exact else "ambiguous_collision"}
            exclusions.append(entry)
            training_checks.append(entry)
            if name in clean:
                clean.remove(name)
    # Retain enough source evidence to inspect every comparison. The canonical
    # Minnesota catalog is packaged; no reserved labels/arrays are packaged.
    audit = {"reserved_source_sha256": file_sha(reserved), "training_source_sha256": file_sha(csv_path),
             "reserved_use": "identity/fingerprint comparison only", "exclusions": exclusions,
             "training_correspondences": training_checks, "comparisons": comparisons,
             "clean_count": len(clean), "diagnostic_count": len(diet30),
             "training_geometry_source": "official Minnesota Database 2.0 XYZ + Gaussian charge/multiplicity",
             "training_identities_compared": len(identities),
             "unknown_training_geometry_policy": "all source names mapped; composition collisions withheld, never inferred equal by formula"}
    return clean, audit


def fixed_density(root, raw, catalog, d3pbe0, d3pbe, grid_level=3, only_names=None):
    from pyscf import dft, lib
    from pyscf.scf import chkfile
    # Fixed preprocessing parallelism; no SCF iterations or model arithmetic.
    lib.num_threads(4)
    source_species, reactions = {}, []
    for db, rid, reaction in catalog:
        components = []
        for label, species in reaction["Species"].items():
            name = f"{db}-{rid}-{label}"
            identity = canonical_id("diet_species", name)
            if name in source_species:
                raise ValueError("Duplicate Diet checkpoint mapping")
            source_species[name] = (identity, species)
            components.append({"species_id": identity, "coefficient": float(species["Count"])})
        reactions.append({"id": canonical_id("diet_reaction", [db, rid]), "source_id": f"{db}-{rid}",
                          "database": db, "reaction_id": rid, "components": components,
                          "reference_energy_kcal_mol": float(reaction["Energy"]),
                          "diet_weight": float(reaction["Weight"]),
                          "fingerprint": reaction_fingerprint(reaction)})
    if len(source_species) != 84 or len(reactions) != 30:
        raise ValueError("Expected 84 unique species / 30 diagnostic reactions")
    checkpoints = {p.name.removesuffix(".pbe0.chk"): p for p in (raw / "chk").glob("*.pbe0.chk")}
    if set(checkpoints) != set(source_species):
        raise ValueError("Checkpoint/catalog mapping mismatch")
    primary, secondary = parse_d3(d3pbe0, source_species), parse_d3(d3pbe, source_species)
    writer, rows = Shards(root, "validation"), []
    try:
        for name, (identity, species) in sorted(source_species.items()):
            if only_names is not None and name not in only_names:
                continue
            path = checkpoints[name]
            marker = Path(str(path) + ".complete")
            if not marker.is_file() or marker.read_text().strip() != "converged":
                raise ValueError(f"Missing/invalid completion marker: {name}")
            with h5py.File(path, "r") as handle:
                if "mol" not in handle or "scf/dm" not in handle:
                    raise ValueError("Incomplete PySCF checkpoint")
                dm = handle["scf/dm"][...]
                source_energy = float(handle["scf/e_tot"][()])
            if dm.dtype != np.float64 or not np.isfinite(dm).all() or not np.isfinite(source_energy):
                raise ValueError("Native checkpoint F64/nonfinite violation")
            mol = chkfile.load_mol(str(path))
            if mol.charge != species["Charge"] or mol.spin != species["UHF"]:
                raise ValueError(f"Charge/spin mismatch: {name}")
            if [mol.atom_symbol(i) for i in range(mol.natm)] != species["Elements"]:
                raise ValueError(f"Element mapping mismatch: {name}")
            if not np.allclose(mol.atom_coords(unit="Angstrom"), species["Positions"], atol=2e-5, rtol=0):
                raise ValueError(f"Geometry mapping mismatch: {name}")
            mf = dft.UKS(mol) if dm.ndim == 3 else dft.RKS(mol)
            mf.xc = "PBE"
            mf.grids.level = grid_level
            mf.grids.build(with_non0tab=True)
            # No kernel/SCF call. J is the total-density Coulomb operator.
            total = dm.sum(axis=0) if dm.ndim == 3 else dm
            hcore = mf.get_hcore()
            coulomb = mf.get_j(mol, total)
            nonxc = float(np.einsum("ij,ji", total, hcore) +
                          .5 * np.einsum("ij,ji", total, coulomb) + mol.energy_nuc())
            n = len(mf.grids.weights)
            features = np.empty((n, 10), dtype=np.float64)
            eps = np.empty(n, dtype=np.float64)
            spin_dm = dm if dm.ndim == 3 else np.stack((dm * .5, dm * .5))
            for start in range(0, n, 4096):
                end = min(start + 4096, n)
                ao = dft.numint.eval_ao(mol, mf.grids.coords[start:end], deriv=2)
                rho = [dft.numint.eval_rho(mol, ao, matrix, xctype="MGGA", with_lapl=True) for matrix in spin_dm]
                features[start:end, :2] = np.stack((rho[0][0], rho[1][0]), axis=1)
                features[start:end, 2:5] = rho[0][1:4].T
                features[start:end, 5:8] = rho[1][1:4].T
                features[start:end, 8:10] = np.stack((rho[0][4], rho[1][4]), axis=1)
                if dm.ndim == 3:
                    eps[start:end] = mf._numint.eval_xc("PBE", np.array(rho)[:, :4], spin=1, deriv=0)[0]
                else:
                    eps[start:end] = mf._numint.eval_xc("PBE", (rho[0] + rho[1])[:4], spin=0, deriv=0)[0]
            excitation = float(np.dot(eps * features[:, :2].sum(axis=1), mf.grids.weights))
            decomposition = nonxc + excitation
            # Direct PySCF fixed-density parity is retained for ordinary systems.
            # For the two 1449-AO MCONF checkpoints, dense veff evaluation is not
            # computationally cheap; their decomposition inputs remain complete.
            parity_checked = mol.nao_nr() <= 1000
            direct_energy = None
            parity_error = None
            if parity_checked:
                direct_vhf = mf.get_veff(mol, dm)
                direct_energy = float(mf.energy_tot(dm=dm, h1e=hcore, vhf=direct_vhf))
                parity_error = abs(decomposition - direct_energy)
                # Frozen F64 fixed-density parity tolerance; no benchmark scoring.
                if not np.isclose(decomposition, direct_energy, rtol=1e-12, atol=1e-8):
                    raise ValueError(f"Fixed-density PBE parity failed: {name}: {decomposition-direct_energy}")
            group, reference = writer.group(identity, features.nbytes + dm.nbytes)
            for field, array in {"features": features, "weights": mf.grids.weights,
                                 "coords64": mf.grids.coords, "dm": dm,
                                 "nonxc": np.asarray(nonxc), "pbe_epsilon": eps}.items():
                put(group, field, array)
            import json
            group.attrs["molecule_json"] = canonical(portable(json.loads(mol.dumps()))).decode()
            row = {"id": identity, "source_id": name, **reference,
                   "checkpoint_sha256": file_sha(path), "complete_sha256": file_sha(marker),
                   "geometry_fingerprint": geometry_fingerprint(species),
                   "charge": mol.charge, "spin": mol.spin, "n_grid": n, "n_ao": mol.nao_nr(),
                   "elements": species["Elements"], "positions_angstrom": species["Positions"],
                   "source_pbe0_e_tot_metadata_only": source_energy,
                   "primary_dispersion_hartree": primary[name], "secondary_dispersion_hartree": secondary[name],
                   "fixed_density_pbe_parity_abs_error": parity_error,
                   "fixed_density_pbe_parity_checked": parity_checked,
                   "fixed_density_pbe_parity_note": "direct PySCF fixed-density PBE parity"
                       if parity_checked else "deferred: nAO > 1000; decomposition inputs retained"}
            rows.append(row)
            print(canonical({"validation_species": name, "parity_error": row["fixed_density_pbe_parity_abs_error"],
                             "parity_checked": parity_checked}).decode(), flush=True)
    finally:
        writer.close()
    jsonl(root / "validation/diet30_species.jsonl", rows)
    jsonl(root / "validation/diet30_reactions.jsonl", reactions)
    dump(root / "provenance/validation_protocol.json", {"grid_level": grid_level,
         "grid": "PySCF default atom_grid, radi_method, becke_scheme and nwchem pruning; no density-based grid pruning",
         "pyscf_version": __import__("pyscf").__version__, "arithmetic": "F64", "threads": 4,
         "nonxc": "Tr(P_total hcore) + 0.5 Tr(P_total J[P_total]) + E_nuc; excludes all XC including hybrid exchange",
         "parity": "direct RKS/UKS PBE fixed-density energy_tot, rtol=1e-12 atol=1e-8 hartree",
         "no_scf": True, "primary_dispersion": "PBE0-D3(BJ)", "secondary_dispersion": "PBE-D3(BJ)"})
    checked = [r for r in rows if r.get("fixed_density_pbe_parity_checked", True)]
    errors = [r["fixed_density_pbe_parity_abs_error"] for r in checked]
    return {"species": len(rows), "diagnostic_reactions": len(reactions), "primary_dispersion_values": len(primary),
            "secondary_dispersion_values": len(secondary), "fixed_density_parity": "PASS",
            "direct_parity_checked_species": len(checked),
            "deferred_large_nao_species": len(rows) - len(checked),
            "max_parity_error_hartree": max(errors) if errors else None}


def _validation_job(arguments):
    """An independent species uses the identical serial numerical routine."""
    import json
    root, raw, catalog, primary, secondary, name = arguments
    root = Path(root)
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    receipt = root / "receipt.json"
    if receipt.exists():
        value = json.loads(receipt.read_text())
        for record in value["files"]:
            if file_sha(root / record["file"]) != record["sha256"]:
                raise ValueError("Validation worker receipt hash mismatch")
        return name, root
    if (root / "validation").exists():
        raise ValueError(f"Interrupted species worker requires inspection: {name}")
    fixed_density(root, raw, catalog, primary, secondary, only_names=(name,))
    dump(receipt, {"species": name, "files": [{"file": p.relative_to(root).as_posix(), "sha256": file_sha(p)}
                                              for p in sorted(root.rglob("*")) if p.is_file()]})
    return name, root


def recover_validation(root, raw, catalog, primary, secondary, log):
    """Recover committed species only with their exact prior parity receipt."""
    import json

    from .contracts import array_sha
    species = {f"{db}-{rid}-{name}": value for db, rid, reaction in catalog
               for name, value in reaction["Species"].items()}
    by_id = {canonical_id("diet_species", name): name for name in species}
    parity = {}
    for line in Path(log).read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "validation_species" in row:
            parity[row["validation_species"]] = row["parity_error"]
    d0, d1 = parse_d3(primary, species), parse_d3(secondary, species)
    rows, files = [], []
    for path in sorted((root / "validation").glob("*.h5")):
        with h5py.File(path) as handle:
            for key, group in handle.items():
                name = by_id[key]
                if name not in parity or parity[name] > 1e-8:
                    raise ValueError(f"Missing prior fixed-density parity: {name}")
                for ds in group.values():
                    if array_sha(ds[...]) != ds.attrs["content_sha256"]:
                        raise ValueError("Interrupted validation content mismatch")
                source = raw / "chk" / (name + ".pbe0.chk")
                marker = Path(str(source) + ".complete")
                with h5py.File(source) as checkpoint:
                    dm = checkpoint["scf/dm"][...]
                    energy = float(checkpoint["scf/e_tot"][()])
                if not np.array_equal(dm, group["dm"][...]) or marker.read_text().strip() != "converged":
                    raise ValueError("Interrupted validation source mismatch")
                value = species[name]
                rows.append({"id": key, "source_id": name, "shard": path.relative_to(root).as_posix(), "group": key,
                             "checkpoint_sha256": file_sha(source), "complete_sha256": file_sha(marker),
                             "geometry_fingerprint": geometry_fingerprint(value), "charge": value["Charge"],
                             "spin": value["UHF"], "n_grid": len(group["weights"]), "n_ao": dm.shape[-1],
                             "elements": value["Elements"], "positions_angstrom": value["Positions"],
                             "source_pbe0_e_tot_metadata_only": energy,
                             "primary_dispersion_hartree": d0[name], "secondary_dispersion_hartree": d1[name],
                             "fixed_density_pbe_parity_abs_error": parity[name],
                             "fixed_density_pbe_parity_checked": True,
                             "fixed_density_pbe_parity_note": "direct PySCF fixed-density PBE parity"})
        files.append({"file": path.relative_to(root).as_posix(), "sha256": file_sha(path)})
        for row in rows:
            if row["shard"] == path.relative_to(root).as_posix():
                row["shard_sha256"] = files[-1]["sha256"]
    dump(root / "provenance/validation_recovery.json", {"files": files, "prior_log_sha256": file_sha(log),
                                                       "species_reused": [r["source_id"] for r in rows]})
    jsonl(root / "provenance/validation_completed_species.jsonl", rows)
    return rows


def validation_accounting(root, rows):
    """Validate the completed, hash-bound 82 direct + 2 component receipts."""
    import json
    receipt = json.loads((root / "provenance/mconf_component_validation.json").read_text())
    if receipt["status"] != "PASS":
        raise ValueError("MCONF component qualification failed")
    components = {r["source_id"]: r for r in receipt["systems"]}
    if set(components) != {"MCONF-1-1", "MCONF-1-2"}:
        raise ValueError("Both MCONF component receipts required")
    direct = []
    for row in rows:
        if row["source_id"] in components:
            proof = components[row["source_id"]]
            if (proof["checkpoint_sha256"] != row["checkpoint_sha256"] or
                    proof["stored_shard_sha256"] != file_sha(root / row["shard"])):
                raise ValueError("MCONF component receipt provenance mismatch")
            for field in ("density_max_abs_diff", "descriptor_max_abs_diff", "xc_energy_abs_diff"):
                if proof[field] != 0:
                    raise ValueError("MCONF exact equality gate failed")
            for field in ("nonxc_component_abs_diff", "recombined_abs_diff"):
                if not np.isfinite(proof[field]) or proof[field] > 1e-8:
                    raise ValueError("MCONF independent component parity failed")
        else:
            error = row["fixed_density_pbe_parity_abs_error"]
            if not row.get("fixed_density_pbe_parity_checked", True) or error is None or not np.isfinite(error) or error > 1e-8:
                raise ValueError("Ordinary direct fixed-density parity missing")
            direct.append(error)
    if len(rows) != 84 or len(direct) != 82:
        raise ValueError("Expected 82 ordinary direct + 2 MCONF component qualifications")
    result = {"status": "PASS", "qualified_species": 84, "direct_species": 82,
              "component_species": 2, "direct_max_abs_error_hartree": max(direct),
              "component_max_abs_error_hartree": max(r["recombined_abs_diff"] for r in components.values()),
              "component_receipt_sha256": file_sha(root / "provenance/mconf_component_validation.json"),
              "component_systems": sorted(components), "component_density_descriptor_xc_exact": True,
              "description": "82 direct fixed-density total-energy parity + 2 large-system component-level parity"}
    dump(root / "provenance/validation_qualification.json", result)
    return result


def parallel_fixed_density(root, raw, catalog, primary, secondary, recovered=None):
    """Bounded independent preprocessing; canonical output order is unchanged."""
    import json
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    rows = [] if recovered is None else recovered
    for row in rows:
        if file_sha(root / row["shard"]) != row["shard_sha256"]:
            raise ValueError("Completed validation shard changed")
    done = {r["source_id"] for r in rows}
    names = sorted(f"{db}-{rid}-{label}" for db, rid, reaction in catalog for label in reaction["Species"])
    work = root.parent / "publication_dataset_v1.validation_work"
    jobs = [(work / canonical_id("job", name), raw, catalog, primary, secondary, name)
            for name in names if name not in done]
    (root / "validation").mkdir(parents=True, exist_ok=True)
    number = len(list((root / "validation").glob("*.h5")))
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        for name, job in pool.map(_validation_job, jobs):
            species = [json.loads(line) for line in (job / "validation/diet30_species.jsonl").read_text().splitlines()]
            if len(species) != 1 or species[0]["source_id"] != name:
                raise ValueError("Validation worker identity mismatch")
            row = species[0]
            import shutil
            relative = f"validation/validation_{number:03d}.h5"
            shutil.copyfile(job / row["shard"], root / relative)
            if file_sha(root / relative) != file_sha(job / row["shard"]):
                raise ValueError("Validation shard publication copy changed")
            row["shard"] = relative
            row["shard_sha256"] = file_sha(root / relative)
            rows.append(row)
            number += 1
            jsonl(root / "provenance/validation_completed_species.jsonl", rows)
            print(canonical({"validation_committed": name, "completed": len(rows)}).decode(), flush=True)
    if len(rows) != 84 or {r["source_id"] for r in rows} != set(names):
        raise ValueError("Incomplete fixed-density validation")
    jsonl(root / "validation/diet30_species.jsonl", sorted(rows, key=lambda r: r["source_id"]))
    # The reaction catalog is identical in every independent worker.
    example = next(p for p in sorted(work.iterdir()) if (p / "receipt.json").exists())
    import shutil
    shutil.copyfile(example / "validation/diet30_reactions.jsonl", root / "validation/diet30_reactions.jsonl")
    protocol = json.loads((example / "provenance/validation_protocol.json").read_text())
    protocol["independent_species_workers"] = 3
    dump(root / "provenance/validation_protocol.json", protocol)
    qualification = validation_accounting(root, rows)
    return {"species": 84, "diagnostic_reactions": 30, "primary_dispersion_values": 84,
            "secondary_dispersion_values": 84, "fixed_density_parity": "PASS",
            "qualification": qualification}
