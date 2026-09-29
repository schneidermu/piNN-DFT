import sys
import types


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

from replay_trial_19_bridge import E3_SCHEDULE_PRESETS, OCCAM3_TIMING_PAIRS


def test_occam3_timing_presets():
    assert len(OCCAM3_TIMING_PAIRS) == 10
    assert len(set(OCCAM3_TIMING_PAIRS)) == 10
    baseline = E3_SCHEDULE_PRESETS["occam2_o1_four_stage"]
    for repair_start, finish_start in OCCAM3_TIMING_PAIRS:
        schedule = E3_SCHEDULE_PRESETS[
            f"occam3_r{repair_start}_f{finish_start}"
        ]
        assert len(schedule) == 4
        assert [phase["start_epoch"] for phase in schedule] == [
            1, 141, repair_start, finish_start
        ]
        assert [phase["end_epoch"] for phase in schedule] == [
            140, repair_start - 1, finish_start - 1, 500
        ]
        for phase, reference in zip(schedule, baseline):
            assert phase["params"] == reference["params"]
        assert [
            epoch
            for phase in schedule
            for epoch in range(phase["start_epoch"], phase["end_epoch"] + 1)
        ] == list(range(1, 501))


if __name__ == "__main__":
    test_occam3_timing_presets()
    print("Occam3 timing presets validated.")
