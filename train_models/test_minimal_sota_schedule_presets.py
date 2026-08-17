import sys
import types
from types import SimpleNamespace


optuna_joint = types.ModuleType("optuna_joint")
for name in (
    "init_distributed",
    "load_chk",
    "load_mrks_dispersions",
    "run_or_reuse_preoptimization",
    "run_trial",
    "set_random_seed",
):
    setattr(optuna_joint, name, None)
optuna_joint.DEFAULT_MRKS_DISPERSIONS = ""
sys.modules["optuna_joint"] = optuna_joint

from replay_trial_19_bridge import (
    MINIMAL_SOTA_SCHEDULE_PRESETS,
    resolve_trial_19_params,
)


def test_minimal_sota_presets_are_three_constant_phases_covering_500_epochs():
    assert len(MINIMAL_SOTA_SCHEDULE_PRESETS) == 10
    for schedule in MINIMAL_SOTA_SCHEDULE_PRESETS.values():
        assert len(schedule) == 3
        epochs = [
            epoch
            for phase in schedule
            for epoch in range(phase["start_epoch"], phase["end_epoch"] + 1)
        ]
        assert epochs == list(range(1, 501))
        assert schedule[1]["end_epoch"] == 440
        assert schedule[2]["start_epoch"] == 441
        assert schedule[2]["end_epoch"] == 500


def test_minimal_sota_factorial_and_supported_suffix():
    main_presets = {
        name: schedule
        for name, schedule in MINIMAL_SOTA_SCHEDULE_PRESETS.items()
        if "onset" not in name
    }
    observed = {
        (
            schedule[0]["params"]["vxc_loss_scale"],
            schedule[0]["params"]["gradient_merge_strategy"],
            schedule[0]["params"]["reaction_grad_scale"],
        )
        for schedule in main_presets.values()
    }
    expected = {
        (vxc, merge, reaction)
        for vxc in (40.0, 80.0)
        for merge in ("sum", "clip_then_sum")
        for reaction in (0.75, 1.0)
    }
    assert observed == expected

    for schedule in MINIMAL_SOTA_SCHEDULE_PRESETS.values():
        repair = schedule[1]
        assert repair["params"]["vxc_loss_scale"] == 15
        assert repair["params"]["exc_loss_scale"] == 3.0
        assert repair["params"]["reaction_grad_scale"] == 0.75
        assert repair["params"]["gradient_merge_strategy"] == "clip_then_sum"
        finish = schedule[2]
        assert finish["params"]["vxc_loss_scale"] == 7.0
        assert finish["params"]["exc_loss_scale"] == 1.0
        assert finish["params"]["reaction_grad_scale"] == 1.0
        assert finish["params"]["gradient_merge_strategy"] == "sum"


def test_minimal_sota_onset_controls_bracket_epoch_281():
    assert MINIMAL_SOTA_SCHEDULE_PRESETS[
        "minimal_v40_sum_r1_onset261"
    ][1]["start_epoch"] == 261
    assert MINIMAL_SOTA_SCHEDULE_PRESETS[
        "minimal_v40_sum_r1"
    ][1]["start_epoch"] == 281
    assert MINIMAL_SOTA_SCHEDULE_PRESETS[
        "minimal_v40_sum_r1_onset301"
    ][1]["start_epoch"] == 301


def test_minimal_sota_preset_resolves_without_other_schedule_options():
    args = SimpleNamespace(
        minimal_sota_schedule_preset="minimal_v40_sum_r1",
        occam_schedule_preset=None,
        simple_schedule_preset=None,
        e3_schedule_preset=None,
        h9_schedule_preset=None,
        micro_schedule_preset=None,
        extend_epochs=0,
    )
    params = resolve_trial_19_params(args)
    assert params["epoch_schedule"] == MINIMAL_SOTA_SCHEDULE_PRESETS[
        args.minimal_sota_schedule_preset
    ]


if __name__ == "__main__":
    test_minimal_sota_presets_are_three_constant_phases_covering_500_epochs()
    test_minimal_sota_factorial_and_supported_suffix()
    test_minimal_sota_onset_controls_bracket_epoch_281()
    test_minimal_sota_preset_resolves_without_other_schedule_options()
    print("Minimal-SOTA schedule presets validated.")
