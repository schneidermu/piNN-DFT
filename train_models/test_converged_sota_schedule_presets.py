import copy
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
    CONVERGED_SOTA_SCHEDULE_SOURCES,
    build_converged_sota_schedule,
    resolve_trial_19_params,
)


def test_converged_sota_presets_preserve_original_500_epoch_schedule():
    assert set(CONVERGED_SOTA_SCHEDULE_SOURCES) == {
        "converged_e3",
        "converged_s3",
        "converged_s4",
    }
    for name, source in CONVERGED_SOTA_SCHEDULE_SOURCES.items():
        schedule = build_converged_sota_schedule(name, 800)
        assert schedule[:-1] == source[:-1]
        assert schedule[-1]["name"] == source[-1]["name"]
        assert schedule[-1]["start_epoch"] == source[-1]["start_epoch"]
        assert schedule[-1]["params"] == source[-1]["params"]
        assert source[-1]["end_epoch"] == 500
        assert schedule[-1]["end_epoch"] == 800
        for epoch in range(1, 501):
            original = next(
                phase
                for phase in source
                if phase["start_epoch"] <= epoch <= phase["end_epoch"]
            )
            converged = next(
                phase
                for phase in schedule
                if phase["start_epoch"] <= epoch <= phase["end_epoch"]
            )
            assert original["name"] == converged["name"]
            assert original["params"] == converged["params"]


def test_converged_sota_tail_is_fixed_and_covers_epoch_800():
    for name in CONVERGED_SOTA_SCHEDULE_SOURCES:
        schedule = build_converged_sota_schedule(name, 800)
        epochs = [
            epoch
            for phase in schedule
            for epoch in range(phase["start_epoch"], phase["end_epoch"] + 1)
        ]
        assert epochs == list(range(1, 801))
        final_params = schedule[-1]["params"]
        assert final_params["gradient_merge_strategy"] == "sum"
        assert final_params["reaction_grad_scale"] == 1.0
        assert final_params["vxc_loss_scale"] == 7.0
        assert final_params["exc_loss_scale"] == 1.0


def test_converged_sota_preset_resolves_without_other_schedule_options():
    args = SimpleNamespace(
        converged_sota_schedule_preset="converged_s3",
        minimal_sota_schedule_preset=None,
        occam_schedule_preset=None,
        simple_schedule_preset=None,
        e3_schedule_preset=None,
        h9_schedule_preset=None,
        micro_schedule_preset=None,
        extend_epochs=0,
        n_train=800,
    )
    params = resolve_trial_19_params(args)
    assert params["epoch_schedule"] == build_converged_sota_schedule(
        args.converged_sota_schedule_preset, 800
    )


def test_build_converged_sota_schedule_does_not_mutate_source():
    before = copy.deepcopy(CONVERGED_SOTA_SCHEDULE_SOURCES["converged_e3"])
    build_converged_sota_schedule("converged_e3", 900)
    assert CONVERGED_SOTA_SCHEDULE_SOURCES["converged_e3"] == before


if __name__ == "__main__":
    test_converged_sota_presets_preserve_original_500_epoch_schedule()
    test_converged_sota_tail_is_fixed_and_covers_epoch_800()
    test_converged_sota_preset_resolves_without_other_schedule_options()
    test_build_converged_sota_schedule_does_not_mutate_source()
    print("Converged-SOTA schedule presets validated.")
