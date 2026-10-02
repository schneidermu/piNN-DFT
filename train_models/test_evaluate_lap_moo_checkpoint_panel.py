"""Low-cost provenance and objective-plumbing tests for checkpoint panels."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TRAIN_MODELS = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_MODELS.parent
for _path in (str(REPO_ROOT), str(TRAIN_MODELS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import evaluate_lap_moo_checkpoint_panel as panel
import lap_moo_training
from lap_moo_protocol import (
    LEGACY_READ_ONLY_PROTOCOL_VERSION,
    canonical_sha256,
    make_protocol_metadata,
)


def _checkpoint_payload() -> tuple[dict, dict]:
    digest = "a" * 64
    protocol = make_protocol_metadata(
        architecture="tiny-test-architecture",
        method="nash_mtl",
        method_hyperparameters={},
        fixed_scalarization=None,
        optimizer={"name": "RAdamW"},
        lr_schedule={"name": "cosine", "total_updates": 5},
        predopt_checkpoint_sha256=digest,
        sampling_manifest_sha256="b" * 64,
        minnesota_data_sha256="c" * 64,
        operator_corpus_manifest_sha256="d" * 64,
        ao_cache_manifest_sha256="e" * 64,
        reaction_dispersions_sha256="f" * 64,
        mrks_dispersions_sha256="1" * 64,
        random_seed=41,
        dtype="float32",
        grid_chunk_size=8,
        ao_cache_chunk_size=4096,
    )
    state = {"weight": torch.tensor([1.0])}
    payload = {
        "checkpoint_kind": "lap-moo-one-stage",
        "protocol_metadata": protocol,
        "protocol_metadata_sha256": canonical_sha256(protocol),
        "sampling_cursor": {"next_update": 2},
        "scheduler_state_dict": {"last_epoch": 2},
        "model_kwargs": {"width": 1},
        "model_state_dict": state,
    }
    expected = {
        "method": "nash_mtl",
        "sampling_manifest_sha256": "b" * 64,
        "predopt_checkpoint_sha256": "a" * 64,
        "updates": 2,
        "model_kwargs": {"width": 1},
    }
    return payload, expected


def test_panel_accepts_hash_bound_checkpoint_with_matching_scheduler_cursor():
    payload, expected = _checkpoint_payload()

    accepted = panel._validate_candidate_checkpoint(payload, **expected)

    assert accepted["method"] == "nash_mtl"
    assert accepted["updates"] == 2
    assert accepted["state"] is payload["model_state_dict"]


def test_panel_accepts_v1_protocol_only_through_its_read_only_path():
    payload, expected = _checkpoint_payload()
    payload["protocol_metadata"]["protocol_version"] = LEGACY_READ_ONLY_PROTOCOL_VERSION
    payload["protocol_metadata"].pop("ao_cache_chunk_size")
    payload["protocol_metadata_sha256"] = canonical_sha256(payload["protocol_metadata"])

    accepted = panel._validate_candidate_checkpoint(payload, **expected)

    assert accepted["method"] == "nash_mtl"
    with pytest.raises(ValueError, match="protocol version"):
        panel.validate_protocol_metadata(payload["protocol_metadata"])


def test_panel_rejects_duplicate_method_ids_with_separate_invocation_guidance():
    candidates = [
        ("nash_mtl", "primary", "25", "primary.pt"),
        ("nash_mtl", "secondary", "25", "secondary.pt"),
    ]
    with pytest.raises(ValueError, match="separate panel invocations"):
        panel._validate_candidate_method_ids(candidates)


def test_gate_parameter_norm_counts_trainable_parameters_only():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(3.0)
    model.register_buffer("diagnostic_buffer", torch.tensor([4.0]))
    model.register_parameter("frozen", torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False))

    assert panel._trainable_parameter_l2_norm(model) == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("hash", "protocol metadata hash mismatch"),
        ("scheduler", "scheduler epoch does not match its cursor"),
        ("stream", "does not match the requested method/stream"),
    ],
)
def test_panel_rejects_tampered_hash_or_incompatible_cursor(mutation, error):
    payload, expected = _checkpoint_payload()
    if mutation == "hash":
        payload["protocol_metadata_sha256"] = "0" * 64
    elif mutation == "scheduler":
        payload["scheduler_state_dict"]["last_epoch"] = 0
    else:
        payload["protocol_metadata"]["sampling_manifest_sha256"] = "2" * 64
        payload["protocol_metadata_sha256"] = canonical_sha256(
            payload["protocol_metadata"]
        )

    with pytest.raises(ValueError, match=error):
        panel._validate_candidate_checkpoint(payload, **expected)


def test_panel_uses_shared_three_objective_factory_on_each_fixed_pair(monkeypatch):
    assert panel.make_three_objective_factories is lap_moo_training.make_three_objective_factories
    calls = []

    class Stream:
        def entry(self, update):
            return {
                "reaction": {"database": "ABDE4", "reaction_id": update},
                "variant_suffix": "default",
                "mrks_system": "H2",
            }

    class GroupStore:
        def load_variant(self, identity, variant):
            return (identity, variant)

    class Systems:
        def load(self, name):
            return {"name": name}

    def canonical_factory(model, reaction, system, **kwargs):
        calls.append((reaction, system, kwargs))
        return {
            "chem": lambda: torch.tensor(1.0),
            "exc": lambda: torch.tensor(2.0),
            "op": lambda: torch.tensor(3.0),
        }

    monkeypatch.setattr(panel, "make_three_objective_factories", canonical_factory)
    rows = panel._evaluate_model(
        torch.nn.Linear(1, 1),
        candidate={"method": "fixture"},
        stream=Stream(),
        group_store=GroupStore(),
        systems=Systems(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        reaction_dispersions={},
        mrks_dispersions={},
        point_chunk_size=4,
        reference_rows=None,
    )

    assert len(rows) == len(calls) == 27
    assert rows[0]["losses"] == {"chem": 1.0, "exc": 2.0, "op": 3.0}
    assert rows[-1]["identity"]["reaction_id"] == 26
    assert rows[0]["ratios_to_predopt"] == {"chem": 1.0, "exc": 1.0, "op": 1.0}
