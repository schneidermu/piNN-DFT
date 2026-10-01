"""Contract tests for the Lap S5 schedule helper."""

import copy
import sys
import types

import pytest


def _install_replay_import_stub():
    """Keep replay-bridge imports isolated from the training dependency stack."""
    if "optuna_joint" in sys.modules:
        return False
    stub = types.ModuleType("optuna_joint")
    for name in (
        "init_distributed",
        "load_chk",
        "load_mrks_dispersions",
        "run_or_reuse_preoptimization",
        "run_trial",
        "set_random_seed",
    ):
        setattr(stub, name, None)
    stub.DEFAULT_MRKS_DISPERSIONS = ""
    sys.modules["optuna_joint"] = stub
    return True


_installed_training_stub = _install_replay_import_stub()

try:
    from .lap_s5_protocol import (
        EXPECTED_BOUNDARIES,
        OMEGA_METADATA,
        SOURCE_PRESET_NAME,
        apply_lap_s5_scale_overrides,
        build_lap_s5_phase_smoke_view,
        build_lap_s5_protocol,
    )
    from .replay_trial_19_bridge import E3_SCHEDULE_PRESETS
except ImportError:
    from lap_s5_protocol import (
        EXPECTED_BOUNDARIES,
        OMEGA_METADATA,
        SOURCE_PRESET_NAME,
        apply_lap_s5_scale_overrides,
        build_lap_s5_phase_smoke_view,
        build_lap_s5_protocol,
    )
    from replay_trial_19_bridge import E3_SCHEDULE_PRESETS

# The replay bridge only needs the temporary names while importing. Do not
# leave a fake training module in sys.modules for unrelated tests or callers.
if _installed_training_stub:
    sys.modules.pop("optuna_joint", None)


@pytest.fixture
def reference_schedule():
    return E3_SCHEDULE_PRESETS[SOURCE_PRESET_NAME]


def test_builder_derives_five_phase_table_from_exact_source_preset(reference_schedule):
    original_source = copy.deepcopy(reference_schedule)

    protocol = build_lap_s5_protocol(reference_schedule)

    phases = protocol["epoch_schedule"]
    assert protocol["source_preset"] == SOURCE_PRESET_NAME
    assert protocol["omega"] == OMEGA_METADATA == 0.5
    assert len(phases) == 5
    assert (
        tuple((phase["start_epoch"], phase["end_epoch"]) for phase in phases)
        == EXPECTED_BOUNDARIES
    )
    for phase in phases:
        start, end = phase["start_epoch"], phase["end_epoch"]
        source_rows = [
            row for row in reference_schedule if start <= row["start_epoch"] <= end
        ]
        assert source_rows
        assert phase["name"] == source_rows[0]["name"]
        assert phase["params"] == source_rows[0]["params"]
        assert all(row["params"] == phase["params"] for row in source_rows)
        assert "omega" not in phase["params"]
        assert "OMEGA" not in phase["params"]

    assert reference_schedule == original_source
    protocol["epoch_schedule"][0]["params"]["vxc_loss_scale"] = -1
    assert reference_schedule == original_source


def test_builder_rejects_noncanonical_reference_values(reference_schedule):
    changed = copy.deepcopy(reference_schedule)
    changed[0]["params"]["accum_iter"] = 9

    with pytest.raises(ValueError, match="canonical reference"):
        build_lap_s5_protocol(changed)


def test_scale_overrides_change_only_allowed_values_and_explicit_clips(
    reference_schedule,
):
    base = build_lap_s5_protocol(reference_schedule)
    adjusted = apply_lap_s5_scale_overrides(
        base,
        {
            "representation_40": {"vxc_loss_scale": 33.0},
            "repair": {
                "reaction_grad_scale": 0.8,
                "exc_loss_scale": 2.5,
            },
        },
        clip_thresholds={"repair": {"exc_grad_clip": 1.5}},
    )

    by_id = {phase["phase_id"]: phase for phase in adjusted["epoch_schedule"]}
    assert by_id["representation_40"]["params"]["vxc_loss_scale"] == 33.0
    assert by_id["repair"]["params"]["reaction_grad_scale"] == 0.8
    assert by_id["repair"]["params"]["exc_loss_scale"] == 2.5
    assert by_id["repair"]["params"]["exc_grad_clip"] == 1.5
    assert base["epoch_schedule"][1]["params"]["vxc_loss_scale"] == 40.0
    assert base["epoch_schedule"][3]["params"]["exc_grad_clip"] == 2.0
    assert adjusted["omega"] == base["omega"] == 0.5

    allowed = {
        "vxc_loss_scale",
        "exc_loss_scale",
        "reaction_grad_scale",
        "reaction_grad_clip",
        "vxc_grad_clip",
        "exc_grad_clip",
    }
    base_by_id = {phase["phase_id"]: phase for phase in base["epoch_schedule"]}
    for phase in adjusted["epoch_schedule"]:
        original = base_by_id[phase["phase_id"]]
        assert (phase["start_epoch"], phase["end_epoch"]) == (
            original["start_epoch"],
            original["end_epoch"],
        )
        for key, value in phase["params"].items():
            if key not in allowed or key not in original["params"]:
                assert value == original["params"][key]


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "start_epoch",
        "end_epoch",
        "lr_train",
        "accum_iter",
        "gradient_merge_strategy",
        "exc_gradient_merge_strategy",
    ],
)
def test_scale_override_rejects_phase_optimizer_accumulation_and_merge_changes(
    reference_schedule, forbidden_key
):
    protocol = build_lap_s5_protocol(reference_schedule)
    with pytest.raises(ValueError, match="Disallowed scale override"):
        apply_lap_s5_scale_overrides(
            protocol,
            {"repair": {forbidden_key: 2}},
        )


def test_scale_override_rejects_a_protocol_with_changed_boundaries(reference_schedule):
    protocol = build_lap_s5_protocol(reference_schedule)
    protocol["epoch_schedule"][2]["start_epoch"] = 178

    with pytest.raises(ValueError, match="invalid epoch boundaries"):
        apply_lap_s5_scale_overrides(protocol, {"repair": {"vxc_loss_scale": 16}})


def test_phase_smoke_view_keeps_source_labels_and_parameters(reference_schedule):
    protocol = build_lap_s5_protocol(reference_schedule)
    smoke = build_lap_s5_phase_smoke_view(protocol, per_phase_step_limit=2)

    assert smoke["production_schedule"] is False
    assert smoke["per_phase_step_limit"] == 2
    assert smoke["omega"] == 0.5
    assert tuple(
        (phase["start_epoch"], phase["end_epoch"]) for phase in smoke["epoch_schedule"]
    ) == (
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
    )
    for source, smoke_phase in zip(protocol["epoch_schedule"], smoke["epoch_schedule"]):
        assert smoke_phase["source_phase_name"] == source["name"]
        assert (smoke_phase["source_start_epoch"], smoke_phase["source_end_epoch"]) == (
            source["start_epoch"],
            source["end_epoch"],
        )
        assert smoke_phase["params"] == source["params"]


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_phase_smoke_view_requires_a_positive_integer_step_limit(
    reference_schedule, limit
):
    protocol = build_lap_s5_protocol(reference_schedule)
    with pytest.raises(ValueError, match="positive integer"):
        build_lap_s5_phase_smoke_view(protocol, per_phase_step_limit=limit)
