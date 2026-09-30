"""Provenance and launcher contract for the 11 fresh S5/timing experiments."""

import json
import pickle

import torch
import re
import shlex
from pathlib import Path

import h5py
import numpy as np
import pytest

import prepare_data as mn
import prepare_training_corpus as corpus

ROOT = Path(__file__).resolve().parent
PAIRS = [
    (241, 441),
    (261, 441),
    (221, 441),
    (281, 441),
    (241, 421),
    (261, 421),
    (281, 421),
    (221, 421),
    (241, 461),
    (261, 461),
]


@pytest.mark.parametrize("import_splits", [False, True])
def test_regeneration_records_real_split_and_all_mrks_and_refuses_reuse(
    monkeypatch, tmp_path, import_splits
):
    mn_source = tmp_path / "mn"
    mrks_source = tmp_path / "mrks"
    mn_source.mkdir()
    mrks_source.mkdir()
    (mn_source / "available.h5").touch()
    for i in range(3):
        with h5py.File(mrks_source / f"system_{i}.h5", "w") as handle:
            handle["grid"] = np.zeros((2, 12))
            handle["vrho"] = np.ones((2, 2))
            handle["weights"] = np.ones(2)
            handle["E_xc"] = -1.0
    monkeypatch.setattr(mn, "build_file_index", lambda path: {})
    monkeypatch.setattr(
        mn,
        "group_and_augment_reactions",
        lambda pool, index: {
            key: [{**reaction, "Grid": np.zeros((1, 12))}]
            for key, reaction in pool.items()
        },
    )
    directory = tmp_path / "fresh_dietclean_noval"
    options = {}
    if import_splits:
        samples = [
            dict(
                Name=f"system_{i}",
                Grid=torch.zeros(2, 13),
                Vrho=torch.ones(2),
                Weights=torch.ones(2),
                E_xc=torch.tensor(-1.0),
            )
            for i in range(3)
        ]
        for name, payload in (("train", samples[:2]), ("val", samples[2:])):
            path = tmp_path / f"{name}.pickle"
            path.write_bytes(pickle.dumps(payload))
            options[f"mrks_{name}_pickle"] = str(path)
    manifest_path = corpus.regenerate(
        str(mn_source), str(mrks_source), str(directory), **options
    )
    manifest = corpus.verify(directory)
    assert manifest["minnesota_source_reactions"] == 284
    assert manifest["minnesota_training_reactions"] == 268
    assert len(manifest["excluded_minnesota_reactions"]) == 16
    assert manifest["augmented_reaction_samples"] == manifest["predopt_samples"] == 268
    assert manifest["mrks_systems"] == 3
    if import_splits:
        assert [source["samples"] for source in manifest["mrks_source_pickles"]] == [
            2,
            1,
        ]
        assert manifest["mrks_source_mode"] == "merged-existing-splits"
        for source in manifest["mrks_source_pickles"]:
            assert source["sha256"] == corpus.sha256(Path(source["path"]))
        with (directory / "data_vxc_train.pickle").open("rb") as handle:
            merged = pickle.load(handle)
        for original, restored in zip(samples, merged):
            for key in ("Grid", "Vrho", "Weights", "E_xc"):
                assert torch.equal(original[key], restored[key])

    assert len(manifest["git_commit"]) == 40
    assert manifest["preprocessing_timestamp_utc"]
    assert not any((directory / name).exists() for name in mn.OBSOLETE_PICKLES)
    assert len(mn.load_chk(str(directory))[2]) == 3
    with pytest.raises(FileExistsError, match="overwrite"):
        corpus.regenerate(str(mn_source), str(mrks_source), str(directory))
    wrong_counts = {
        **manifest,
        "minnesota_source_reactions": 285,
        "minnesota_training_reactions": 269,
    }
    manifest_path.write_text(json.dumps(wrong_counts))
    with pytest.raises(ValueError, match="counts"):
        corpus.verify(directory)
    manifest_path.write_text(json.dumps(manifest))
    # Stored manifest detects changed artifacts, not just missing filenames.
    with (directory / "data_vxc_train.pickle").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        corpus.verify(directory)
    manifest["excluded_minnesota_reactions"] = []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact 16"):
        corpus.verify(directory)


def test_missing_sources_cannot_produce_a_manifest(tmp_path):
    directory = tmp_path / "fresh"
    with pytest.raises(ValueError, match="missing or empty"):
        corpus.regenerate(
            str(tmp_path / "missing"), str(tmp_path / "missing"), str(directory)
        )
    assert not directory.exists()


def _runner_arguments(text):
    invocation = text.split("    replay_trial_19_bridge.py", 1)[1]
    tokens = shlex.split(invocation.replace("\\\n", " "))
    return tokens


def test_all_eleven_jobs_use_identical_fresh_training_settings():
    s5_directory = ROOT / "trial19_simple4_sota_sweep"
    timing_directory = ROOT / "trial19_occam3_timing_sweep"
    s5 = (s5_directory / "run_simple4_job.sh").read_text()
    timing = (timing_directory / "run_occam3_job.sh").read_text()
    args_s5 = _runner_arguments(s5)
    args_timing = _runner_arguments(timing)
    args_s5[args_s5.index("--e3-schedule-preset") + 1] = "PRESET"
    args_timing[args_timing.index("--e3-schedule-preset") + 1] = "PRESET"
    assert args_s5 == args_timing
    for runner in (s5, timing):
        assert "checkpoints_dietclean_noval_v1" in runner
        assert "prepare_training_corpus.py --verify-only" in runner
        assert "dietclean_noval_v1" in re.search(r"^OUTPUT_DIR=(.*)$", runner, re.M)[1]
        assert 'if [[ -e "$OUTPUT_DIR" ]]' in runner
        assert "../checkpoints/data_predopt.pickle" not in runner
        assert "--force-preopt" in runner
        assert "--dropout 0.0" in runner
        assert "--snapshot-start-epoch 10" in runner
        assert "--snapshot-every 10" in runner
        assert "--training-state-every 10" in runner
        assert "--n-train 500" in runner
        assert "--resume-training-state" not in runner
        assert "--shared-preopt-checkpoint" not in runner
        assert "--val-" not in runner
        assert "--save-selected-checkpoints" not in runner
    baseline = (s5_directory / "s5_two_step_40_10.slurm").read_text()
    assert "SIMPLE4_PRESET=simple4_two_step_40_10" in baseline
    launchers = sorted(timing_directory.glob("*.slurm"))
    assert len(launchers) == 10
    for job, (repair, finish) in zip(launchers, PAIRS):
        text = job.read_text()
        assert f"S5_TIMING_PRESET=s5_timing_r{repair}_f{finish}" in text
        assert "run_occam3_job.sh" in text
        assert "--resume" not in text


def test_all_eleven_jobs_use_identical_slurm_resources():
    jobs = [
        ROOT / "trial19_simple4_sota_sweep" / "s5_two_step_40_10.slurm",
        *sorted((ROOT / "trial19_occam3_timing_sweep").glob("*.slurm")),
    ]
    resources = []
    for job in jobs:
        resources.append(
            [
                line
                for line in job.read_text().splitlines()
                if line.startswith("#SBATCH")
                and not any(field in line for field in ("--job-name", "--output"))
            ]
        )
    assert len(resources) == 11
    assert all(resource == resources[0] for resource in resources)


@pytest.mark.parametrize("issue", ["duplicate", "missing_exc", "nonfinite", "shape"])
def test_import_rejects_invalid_targets(tmp_path, issue):
    sample = dict(
        Name="system",
        Grid=torch.zeros(2, 13),
        Vrho=torch.ones(2),
        Weights=torch.ones(2),
        E_xc=torch.tensor(-1.0),
    )
    other = {**sample, "Name": "other"}
    if issue == "duplicate":
        other["Name"] = "system"
    elif issue == "missing_exc":
        del other["E_xc"]
    elif issue == "nonfinite":
        other["E_xc"] = torch.tensor(float("nan"))
    else:
        other["Vrho"] = torch.ones(3)
    paths = [tmp_path / "train.pickle", tmp_path / "val.pickle"]
    for path, payload in zip(paths, ([sample], [other])):
        path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError):
        corpus.merge_mrks_splits(*map(str, paths))
