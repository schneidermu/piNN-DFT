"""Small contract proofs; real-corpus parity is reported by the build qualifier."""

import pickle

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from train_models.publication_data import PublicationDataset, identity_collate
from train_models.publication_data.build import (
    dump,
    inventory,
    jsonl,
    put,
    write_manifest,
)
from train_models.publication_data.contracts import (
    SCHEMA,
    VARIANTS,
    array_sha,
    canonical_id,
    parse_d3,
)
from train_models.publication_data.loader import Handles
from train_models.publication_data.validation import (
    geometry_fingerprint,
    reaction_fingerprint,
)


@pytest.fixture
def bundle(tmp_path):
    for name in ("chemistry", "mrks", "validation", "provenance"):
        (tmp_path / name).mkdir()
    identity = canonical_id("reaction", ["ABDE4", 0])
    references = {}
    with h5py.File(tmp_path / "chemistry/chemistry_000.h5", "w") as handle:
        handle.attrs["schema"] = SCHEMA
        for index, variant in enumerate(VARIANTS):
            group = handle.create_group(variant)
            arrays = {"Grid": np.ones((2, 9), dtype=np.float32) * (index+1),
                      "Weights": np.ones(2, dtype=np.float32), "Densities": np.ones((2, 2), dtype=np.float32),
                      "Gradients": np.ones((2, 3), dtype=np.float32), "fixed_nonxc": np.float32(3),
                      "PBE_local_energies": np.ones(2, dtype=np.float32)}
            for key, array in arrays.items():
                put(group, key, array)
            ref = {"shard": "chemistry/chemistry_000.h5", "group": variant}
            references[variant] = [{**ref, "pbe_record": ref}]
    reaction = {"id": identity, "database": "ABDE4", "reaction_id": 0, "task": "relchem",
                "components": ["A"], "coefficients": [1.], "target_kcal_mol": 2.,
                "variants": references, "component_paths": {v: ["data/A__"+v+".h5"] for v in VARIANTS},
                "identity_sampling_weight": 1/251, "db_weight": 4/251}
    jsonl(tmp_path / "chemistry/reactions.jsonl", [reaction])
    for relative in ("chemistry/species.jsonl", "mrks/systems.jsonl", "validation/diet30_species.jsonl",
                     "validation/diet30_reactions.jsonl"):
        jsonl(tmp_path / relative, [])
    dump(tmp_path / "splits.json", {"train_relchem": {"ids": [identity]}, "train_ae17": {"ids": []},
                                   "train_mrks": {"ids": []}, "diet30_diagnostic": {"ids": [], "selection_allowed": False},
                                   "diet30_clean_validation": {"ids": [], "selection_allowed": True}})
    write_manifest(tmp_path, {}, {}, status="qualified")
    return PublicationDataset(tmp_path)


def test_identity_variant_axis_and_weights(bundle):
    dataset = bundle.chemistry()
    assert len(dataset) == 1
    identity = dataset.ids[0]
    assert len(bundle.reactions[identity]["variants"]) == 8
    assert bundle.reactions[identity]["identity_sampling_weight"] == 1/251
    assert sum(bundle.reactions[i]["db_weight"] for i in dataset.ids) == 4/251
    for index, variant in enumerate(VARIANTS):
        record = dataset.load_variant(("ABDE4", 0), variant)
        assert record["Grid"].shape == (2, 9)
        assert torch.all(record["Grid"] == index+1)


def test_manifest_variant_replay_is_deterministic(bundle):
    dataset = bundle.chemistry()
    selected = VARIANTS[3]
    current = dataset.load_variant(("ABDE4", 0), selected)
    torch.manual_seed(999)
    reference = dataset.load_variant(("ABDE4", 0), selected)
    assert current["canonical_id"] == reference["canonical_id"]
    assert current["variant"] == reference["variant"]
    assert torch.equal(current["Grid"], reference["Grid"])


def test_worker_local_handles_pickle_and_dataloader(bundle):
    dataset = bundle.chemistry()
    dataset[0]
    assert bundle.handles.handles
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored.bundle.handles.handles
    assert torch.equal(restored[0]["Grid"], dataset[0]["Grid"])
    for workers in (0, 2):
        rows = list(DataLoader(dataset, batch_size=1, num_workers=workers, collate_fn=identity_collate))
        assert torch.equal(rows[0][0]["Grid"], dataset[0]["Grid"])
    bundle.close()
    assert not bundle.handles.handles


def test_units_dtypes_and_logical_hash(bundle):
    handle = bundle.handles.open("chemistry/chemistry_000.h5")
    dataset = handle[VARIANTS[0]]["Grid"]
    assert dataset.attrs["storage_dtype"] == "<f4"
    assert "bohr" in dataset.attrs["units"]
    assert dataset.attrs["axes"] == "point,descriptor"
    assert dataset.attrs["content_sha256"] == array_sha(dataset[...])
    assert inventory(bundle.root)[0] == inventory(bundle.root)[0]


def test_canonical_ids_do_not_depend_on_dict_order():
    assert canonical_id("sample", {"a": 1, "b": 2}) == canonical_id("sample", {"b": 2, "a": 1})


def species(positions):
    return {"Elements": ["H", "Cl"], "Positions": positions, "Charge": 0, "UHF": 0, "Count": -1}


def test_geometry_and_renumbering_identity():
    a = species([[0, 0, 0], [0, 0, 1.2]])
    b = species([[1, 2, 3], [1, 3.2, 3]])
    assert geometry_fingerprint(a) == geometry_fingerprint(b)
    assert reaction_fingerprint({"Species": {"one": a}}) == reaction_fingerprint({"Species": {"renamed": b}})
    changed = species([[0, 0, 0], [0, 0, 1.5]])
    assert geometry_fingerprint(changed) != geometry_fingerprint(a)
    charged = {**a, "Charge": 1}
    assert geometry_fingerprint(charged) != geometry_fingerprint(a)


@pytest.mark.parametrize("contents", ["A.gif_ 1\nA.gif_ 2\n", "A.gif_ nan\n", "B.gif_ 1\n"])
def test_bad_dispersion_rejected(tmp_path, contents):
    path = tmp_path / "d3.txt"
    path.write_text(contents)
    with pytest.raises(ValueError):
        parse_d3(path, ["A"])


def test_dispersion_parser(tmp_path):
    path = tmp_path / "d3.txt"
    path.write_text("A.gif_ -0.123\n")
    assert parse_d3(path, ["A"]) == {"A": -.123}


def test_missing_corrupted_shards_fail_closed(bundle):
    bundle.close()
    path = bundle.root / "chemistry/chemistry_000.h5"
    with h5py.File(path, "r+") as handle:
        handle[VARIANTS[0]]["Grid"][0, 0] += 1
    corrupted = PublicationDataset(bundle.root)
    with pytest.raises(ValueError, match="hash mismatch"):
        corrupted.chemistry()[0]
    path.unlink()
    with pytest.raises(ValueError, match="Missing"):
        PublicationDataset(bundle.root)


def test_handle_schema_rejected(tmp_path):
    with h5py.File(tmp_path / "bad.h5", "w") as handle:
        handle.attrs["schema"] = "old"
    manager = Handles(tmp_path)
    with pytest.raises(ValueError, match="schema"):
        manager.open("bad.h5")


def test_reaction_fingerprint_reversal_and_scale():
    a = species([[0, 0, 0], [0, 0, 1.2]])
    b = species([[0, 0, 0], [0, 0, 1.5]])
    forward = {"Species": {"a": {**a, "Count": -1}, "b": {**b, "Count": 1}}}
    reverse = {"Species": {"a": {**a, "Count": 2}, "b": {**b, "Count": -2}}}
    assert reaction_fingerprint(forward) == reaction_fingerprint(reverse)


def test_logical_hash_independent_of_hdf5_serialization(tmp_path):
    path = tmp_path / "data.h5"
    values = np.arange(100, dtype=np.float64)
    hashes = []
    for compression in (None, "lzf"):
        with h5py.File(path, "w") as handle:
            ds = handle.create_dataset("weights", data=values, compression=compression)
            from train_models.publication_data.contracts import annotate
            annotate(ds, "weights")
            ds.attrs["content_sha256"] = array_sha(values)
        hashes.append(inventory(tmp_path)[0])
    assert hashes[0] == hashes[1]


def test_portable_metadata_paths():
    from train_models.publication_data.build import portable
    assert portable({"source": "C:/private/data/source.h5"}) == {"source": "source.h5"}
    assert portable("C:\\private\\data\\source.h5") == "source.h5"


def test_operator_availability_fails_closed(bundle):
    bundle.systems["missing"] = {"has_operator": False}
    bundle.splits["train_mrks"]["ids"] = ["missing"]
    with pytest.raises(ValueError, match="Incomplete operator"):
        bundle.mrks()


def test_leakage_uses_alias_geometry_and_actual_training_stoichiometry(tmp_path, monkeypatch):
    from train_models.publication_data import validation
    a = species([[0, 0, 0], [0, 0, 1.2]])
    b = species([[0, 0, 0], [0, 0, 1.5]])
    charged = {**a, "Charge": -1, "UHF": 1}
    bh = {"Species": {"a": {**a, "Count": -1}, "b": {**b, "Count": 1}}, "Energy": 2, "Weight": 3}
    ea = {"Species": {"a": {**a, "Count": -1}, "c": {**charged, "Count": 1}}, "Energy": 4, "Weight": 5}
    other = {"Species": {"only": {**a, "Count": 1}}, "Energy": 1, "Weight": 1}
    reserved = tmp_path / "reserved.yaml"
    reserved.write_text("identity-only source")
    monkeypatch.setattr(validation, "benchmark_rows", lambda path: [("BH76", 6, bh), ("G21EA", 25, ea)])
    csv = tmp_path / "training.csv"
    csv.write_text("ReactionID,Database,a,b\n0,DBH76,-1,1\n")
    training = [{"database": "DBH76", "reaction_id": 0, "components": ["a", "b"], "coefficients": [-1, 1]}]
    clean, audit = validation.leakage([("BH76", 5, bh), ("G21EA", 25, ea), ("OTHER", 1, other)],
                                      reserved, training, csv, {"a": a, "b": b})
    assert clean == ["OTHER-1"]
    assert any(r["counterpart"] == "Diet100:BH76-6" for r in audit["exclusions"])
    assert any(r["counterpart"] == "Minnesota:DBH76-0" for r in audit["exclusions"])
    training[0]["coefficients"] = [-2, 1]
    with pytest.raises(ValueError, match="stoichiometry"):
        validation.leakage([("BH76", 5, bh), ("G21EA", 25, ea)], reserved, training, csv, {"a": a, "b": b})


def test_semantics_schema_extracts_current_weighting(tmp_path):
    import json

    from train_models.publication_data.build import schema
    schema(tmp_path)
    value = json.loads((tmp_path / "provenance/schema.json").read_text())
    assert value["chemistry_loss_constants"]["FREQ_WEIGHTS"]["ABDE4"] == 1/4
    assert value["chemistry_loss_constants"]["FCHEM_DB_WEIGHTS"]["NCCE31"] == 10
    assert "251" in value["full251_scalar"]


def test_array_verifier_includes_scalar_content_and_detects_mutation(tmp_path):
    from tools.qualify_publication_dataset import verify_arrays
    with h5py.File(tmp_path / "data.h5", "w") as handle:
        put(handle, "nonxc", np.asarray(1.25, dtype=np.float64))
        put(handle, "weights", np.arange(5, dtype=np.float32))
    assert verify_arrays(tmp_path) == 2
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        handle["nonxc"][()] = 1.5
    with pytest.raises(ValueError, match="content hash"):
        verify_arrays(tmp_path)


def test_logical_hash_independent_of_shard_assignment(tmp_path):
    modality = tmp_path / "mrks"
    modality.mkdir()
    source = modality / "mrks_000.h5"
    with h5py.File(source, "w") as handle:
        put(handle.create_group("canonical_system"), "weights", np.arange(3, dtype=np.float64))
    row = {"id": "canonical_system", "shard": "mrks/mrks_000.h5", "group": "canonical_system"}
    jsonl(modality / "systems.jsonl", [row])
    initial = inventory(tmp_path)[0]
    source.rename(modality / "mrks_004.h5")
    row["shard"] = "mrks/mrks_004.h5"
    jsonl(modality / "systems.jsonl", [row])
    assert inventory(tmp_path)[0] == initial


def test_chemistry_dispersion_restores_native_scalar_precision(tmp_path):
    from train_models.publication_data.loader import chemistry_dispersions
    (tmp_path / "chemistry").mkdir()
    dump(tmp_path / "chemistry/dispersion.json", {"species": -.00439715, "missing": 0})
    dump(tmp_path / "chemistry/dispersion_metadata.json", {
        "species": {"source_present": True, "source_dtype": "<f8", "source_shape": []},
        "missing": {"source_present": False, "source_dtype": "<i8", "source_shape": []}})
    values = chemistry_dispersions(tmp_path)
    assert "missing" not in values
    assert values["species"].shape == ()
    assert torch.tensor(values["species"]).dtype == torch.float64
    assert torch.tensor(float(values["species"])).dtype == torch.float32


def test_completed_validation_recovery_uses_hash_and_prior_parity(tmp_path):
    from train_models.publication_data.validation import recover_validation
    root, raw = tmp_path / "staging", tmp_path / "raw"
    (root / "validation").mkdir(parents=True)
    (root / "provenance").mkdir()
    (raw / "chk").mkdir(parents=True)
    value = species([[0, 0, 0], [0, 0, 1.2]])
    name = "BH76-5-a"
    key = canonical_id("diet_species", name)
    dm = np.eye(2, dtype=np.float64)
    with h5py.File(raw / "chk" / (name + ".pbe0.chk"), "w") as handle:
        handle.create_dataset("scf/dm", data=dm)
        handle.create_dataset("scf/e_tot", data=-1)
    (raw / "chk" / (name + ".pbe0.chk.complete")).write_text("converged")
    with h5py.File(root / "validation/validation_000.h5", "w") as handle:
        group = handle.create_group(key)
        put(group, "dm", dm)
        put(group, "weights", np.ones(3, dtype=np.float64))
    d3 = tmp_path / "d3"
    d3.write_text(name + " -0.01\n")
    log = tmp_path / "log"
    log.write_text('{"validation_species":"BH76-5-a","parity_error":1e-13}\n')
    catalog = [("BH76", 5, {"Species": {"a": value}})]
    rows = recover_validation(root, raw, catalog, d3, d3, log)
    assert rows[0]["source_id"] == name
    assert rows[0]["shard_sha256"]
    log.write_text("")
    with pytest.raises(ValueError, match="prior fixed-density parity"):
        recover_validation(root, raw, catalog, d3, d3, log)


def test_chemistry_collator_reuses_legacy_batching_without_mutation(bundle):
    from train_models.publication_data import chemistry_collate
    from train_models.utils import stack_reactions
    rows = [bundle.chemistry()[0], bundle.chemistry()[0]]
    expected = stack_reactions(rows)
    actual = chemistry_collate(rows)
    assert torch.equal(actual["Grid"], expected["Grid"])
    assert torch.equal(actual["Energy"], expected["Energy"])
    assert actual["reaction_indices"] == [0, 1, 2]
    assert "Energy" in rows[0]
    assert actual["variant"] == ["level2", "level2"]


def test_samples_ignore_global_default_device(bundle):
    with torch.device("meta"):
        row = bundle.chemistry()[0]
    assert all(v.device.type == "cpu" for v in row.values() if isinstance(v, torch.Tensor))


def test_split_membership_and_diagnostic_policy_fail_closed(bundle):
    import json
    root = bundle.root
    bundle.close()
    splits = json.loads((root / "splits.json").read_text())
    splits["train_relchem"]["ids"].append(splits["train_relchem"]["ids"][0])
    dump(root / "splits.json", splits)
    write_manifest(root, {}, {}, status="qualified")
    with pytest.raises(ValueError, match="Split membership"):
        PublicationDataset(root)
    splits["train_relchem"]["ids"].pop()
    splits["diet30_diagnostic"]["selection_allowed"] = True
    dump(root / "splits.json", splits)
    write_manifest(root, {}, {}, status="qualified")
    with pytest.raises(ValueError, match="selection policy"):
        PublicationDataset(root)


def test_validation_accounting_requires_both_component_proofs(tmp_path):
    from train_models.publication_data.contracts import file_sha
    from train_models.publication_data.validation import validation_accounting
    (tmp_path / "provenance").mkdir()
    rows = [{"source_id": str(i), "fixed_density_pbe_parity_abs_error": 1e-12} for i in range(82)]
    systems = []
    for name in ("MCONF-1-1", "MCONF-1-2"):
        path = tmp_path / (name + ".h5")
        path.write_bytes(b"immutable synthetic shard")
        rows.append({"source_id": name, "shard": path.name, "checkpoint_sha256": "source"})
        systems.append({"source_id": name, "checkpoint_sha256": "source",
                        "stored_shard_sha256": file_sha(path), "density_max_abs_diff": 0,
                        "descriptor_max_abs_diff": 0, "xc_energy_abs_diff": 0,
                        "nonxc_component_abs_diff": 1e-12, "recombined_abs_diff": 1e-12})
    receipt = tmp_path / "provenance/mconf_component_validation.json"
    dump(receipt, {"status": "PASS", "systems": systems})
    result = validation_accounting(tmp_path, rows)
    assert (result["direct_species"], result["component_species"]) == (82, 2)
    systems[0]["descriptor_max_abs_diff"] = 1e-15
    dump(receipt, {"status": "PASS", "systems": systems})
    with pytest.raises(ValueError, match="exact equality"):
        validation_accounting(tmp_path, rows)
