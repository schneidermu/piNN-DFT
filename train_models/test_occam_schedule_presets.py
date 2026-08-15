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

from replay_trial_19_bridge import OCCAM_SCHEDULE_PRESETS, resolve_trial_19_params


def test_occam_schedule_presets_cover_exactly_500_epochs():
    assert len(OCCAM_SCHEDULE_PRESETS) == 10
    for schedule in OCCAM_SCHEDULE_PRESETS.values():
        epochs = [
            epoch
            for phase in schedule
            for epoch in range(phase["start_epoch"], phase["end_epoch"] + 1)
        ]
        assert epochs == list(range(1, 501))


def test_occam_schedule_presets_share_canonical_boundaries():
    for schedule in OCCAM_SCHEDULE_PRESETS.values():
        assert len(schedule) == 302
        assert schedule[299]["end_epoch"] == 300
        assert schedule[300]["start_epoch"] == 301
        assert schedule[300]["end_epoch"] == 400
        assert schedule[301]["start_epoch"] == 401
        assert schedule[301]["end_epoch"] == 500
        assert schedule[0]["params"]["vxc_loss_scale"] == 64.0
        assert schedule[299]["params"]["vxc_loss_scale"] == 16.0
        assert schedule[300]["params"]["vxc_loss_scale"] == 16.0
        assert schedule[301]["params"]["vxc_loss_scale"] == 8.0
        repair_order = schedule[300]["params"]["exc_loss_scale"]
        assert repair_order in {2.0, 3.0}
        assert schedule[300]["params"]["reaction_grad_scale"] == (
            repair_order / (repair_order + 1.0)
        )


def test_occam_preset_resolves_without_other_schedule_options():
    args = SimpleNamespace(
        occam_schedule_preset="occam_geometric_triple",
        simple_schedule_preset=None,
        e3_schedule_preset=None,
        h9_schedule_preset=None,
        micro_schedule_preset=None,
        extend_epochs=0,
    )
    params = resolve_trial_19_params(args)
    assert params["epoch_schedule"] == OCCAM_SCHEDULE_PRESETS[args.occam_schedule_preset]


if __name__ == "__main__":
    test_occam_schedule_presets_cover_exactly_500_epochs()
    test_occam_schedule_presets_share_canonical_boundaries()
    test_occam_preset_resolves_without_other_schedule_options()
    print("Occam schedule presets validated.")
