from replay_trial_19_bridge import E3_SCHEDULE_PRESETS


PRESETS = [name for name in E3_SCHEDULE_PRESETS if name.startswith("simple4_")]


def test_simple_e3_curricula_cover_500_epochs():
    assert len(PRESETS) == 10
    for name in PRESETS:
        schedule = E3_SCHEDULE_PRESETS[name]
        assert schedule[0]["start_epoch"] == 1
        assert schedule[-1]["end_epoch"] == 500
        assert all(
            left["end_epoch"] + 1 == right["start_epoch"]
            for left, right in zip(schedule, schedule[1:])
        )


def test_simple_e3_curricula_keep_supported_anchor_and_repair():
    for name in PRESETS:
        schedule = E3_SCHEDULE_PRESETS[name]
        assert schedule[0]["name"] == "simple_potential_anchor"
        repair = next(phase for phase in schedule if phase["name"] == "simple_chemical_repair")
        assert repair["start_epoch"] == 281
        assert repair["params"]["exc_loss_scale"] == 3.0
        assert repair["params"]["vxc_loss_scale"] == 15
        assert repair["params"]["reaction_grad_scale"] == 0.75


if __name__ == "__main__":
    test_simple_e3_curricula_cover_500_epochs()
    test_simple_e3_curricula_keep_supported_anchor_and_repair()
