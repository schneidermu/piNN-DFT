import re
import sys
import types
from pathlib import Path


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

from replay_trial_19_bridge import E3_SCHEDULE_PRESETS, S5_TIMING_PAIRS


ROOT = Path(__file__).parent
S5 = E3_SCHEDULE_PRESETS["simple4_two_step_40_10"]
PARAMETER_KEYS = (
    "accum_iter",
    "gradient_merge_strategy",
    "reaction_grad_clip",
    "reaction_grad_scale",
    "vxc_grad_clip",
    "vxc_loss_scale",
    "exc_loss_scale",
    "exc_grad_clip",
    "exc_grad_scale",
    "exc_gradient_merge_strategy",
)
EXPECTED_PAIRS = (
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
)


def _phase_at(schedule, epoch):
    return next(
        phase
        for phase in schedule
        if phase["start_epoch"] <= epoch <= phase["end_epoch"]
    )


def _timing_schedule(repair_start, finish_start):
    return E3_SCHEDULE_PRESETS[
        f"s5_timing_r{repair_start}_f{finish_start}"
    ]


def _assert_parameters_equal(left, right):
    assert left == right
    assert set(PARAMETER_KEYS) <= left.keys()
    for key in PARAMETER_KEYS:
        assert left[key] == right[key]


def test_s5_timing_pairs_are_the_requested_ten():
    assert S5_TIMING_PAIRS == EXPECTED_PAIRS
    assert len(set(S5_TIMING_PAIRS)) == 10


def test_s5_baseline_has_the_expected_absolute_epoch_phases():
    expected = (
        (1, 72, 75.0, 1.0, 0.6, "clip_then_sum"),
        (73, 176, 40.0, 1.0, 1.0, "sum"),
        (177, 280, 10.0, 1.0, 1.0, "sum"),
        (281, 440, 15.0, 3.0, 0.75, "clip_then_sum"),
        (441, 500, 7.0, 1.0, 1.0, "sum"),
    )
    for start, end, vxc, exc, reaction, merge in expected:
        params = _phase_at(S5, start)["params"]
        for epoch in range(start, end + 1):
            assert _phase_at(S5, epoch)["params"] == params
        assert params["vxc_loss_scale"] == vxc
        assert params["exc_loss_scale"] == exc
        assert params["reaction_grad_scale"] == reaction
        assert params["gradient_merge_strategy"] == merge
    repair = _phase_at(S5, 281)["params"]
    assert repair["exc_gradient_merge_strategy"] == "clip_then_sum"
    assert repair["exc_grad_clip"] == 2.0


def test_r281_f441_is_epoch_by_epoch_exact_s5():
    timing = _timing_schedule(281, 441)
    assert timing == S5
    for epoch in range(1, 501):
        _assert_parameters_equal(
            _phase_at(timing, epoch)["params"],
            _phase_at(S5, epoch)["params"],
        )


def test_each_timing_variant_preserves_the_s5_prefix():
    for repair_start, finish_start in S5_TIMING_PAIRS:
        timing = _timing_schedule(repair_start, finish_start)
        for epoch in range(1, repair_start):
            _assert_parameters_equal(
                _phase_at(timing, epoch)["params"],
                _phase_at(S5, epoch)["params"],
            )


def test_each_timing_variant_uses_the_s5_repair_objective():
    baseline_repair = _phase_at(S5, 281)["params"]
    assert baseline_repair["vxc_loss_scale"] == 15.0
    assert baseline_repair["exc_loss_scale"] == 3.0
    assert baseline_repair["reaction_grad_scale"] == 0.75
    assert baseline_repair["gradient_merge_strategy"] == "clip_then_sum"
    assert baseline_repair["exc_gradient_merge_strategy"] == "clip_then_sum"

    for repair_start, finish_start in S5_TIMING_PAIRS:
        timing = _timing_schedule(repair_start, finish_start)
        for epoch in range(repair_start, finish_start):
            _assert_parameters_equal(
                _phase_at(timing, epoch)["params"],
                baseline_repair,
            )


def test_each_timing_variant_uses_the_s5_finalization_objective():
    baseline_finish = _phase_at(S5, 441)["params"]
    assert baseline_finish["vxc_loss_scale"] == 7.0
    assert baseline_finish["exc_loss_scale"] == 1.0
    assert baseline_finish["reaction_grad_scale"] == 1.0
    assert baseline_finish["gradient_merge_strategy"] == "sum"

    for repair_start, finish_start in S5_TIMING_PAIRS:
        timing = _timing_schedule(repair_start, finish_start)
        for epoch in range(finish_start, 501):
            _assert_parameters_equal(
                _phase_at(timing, epoch)["params"],
                baseline_finish,
            )


def test_every_timing_schedule_covers_each_epoch_once():
    for repair_start, finish_start in S5_TIMING_PAIRS:
        schedule = _timing_schedule(repair_start, finish_start)
        epochs = [
            epoch
            for phase in schedule
            for epoch in range(
                phase["start_epoch"], phase["end_epoch"] + 1
            )
        ]
        assert epochs == list(range(1, 501))


def test_both_launchers_save_snapshots_and_resumable_state_every_ten_epochs():
    runner_paths = (
        ROOT / "trial19_simple4_sota_sweep" / "run_simple4_job.sh",
        ROOT / "trial19_occam3_timing_sweep" / "run_occam3_job.sh",
    )
    for runner_path in runner_paths:
        runner = runner_path.read_text(encoding="utf-8")
        assert "--training-state-every 10" in runner
        assert "--snapshot-start-epoch 10" in runner
        assert "--snapshot-every 10" in runner
        assert "source /home/mmedvedev/anaconda3/etc/profile.d/conda.sh" in runner
        assert "conda activate ML_param" in runner
        for required in (
            "--name PBE-LGxGc_6_32",
            "--model-type gc_svelu_mirror",
            "--n-predopt 2",
            "--n-train 500",
            "--force-preopt",
            "--seed 41",
            "--include-mrks-dispersion",
        ):
            assert required in runner
        assert "REQUIRED_SCIENTIFIC_FIX=\"29b36ac2faa29d31a758e0067499f10b41c3ab39\"" in runner
        assert "git merge-base --is-ancestor" in runner
        assert 'echo "Git commit: $(git rev-parse HEAD)"' in runner
        assert "import pyscf; print(\"PySCF:\", pyscf.__version__)" in runner
        assert "Refusing to reuse existing output directory" in runner
    s5_runner = runner_paths[0].read_text(encoding="utf-8")
    timing_runner = runner_paths[1].read_text(encoding="utf-8")
    assert "replay_trial_19_simple4_${SIMPLE4_TAG}_500_gc_svelu_mirror_dietclean_noval_v1" in s5_runner
    assert "replay_trial_19_s5_timing_${S5_TIMING_TAG}_dietclean_noval_v1" in timing_runner


def test_all_sbatch_files_select_the_expected_unique_s5_runs():
    baseline_path = ROOT / "trial19_simple4_sota_sweep" / "s5_two_step_40_10.slurm"
    baseline = baseline_path.read_text(encoding="utf-8")
    assert "SIMPLE4_PRESET=simple4_two_step_40_10" in baseline
    assert "SIMPLE4_TAG=s5_repro_fixed" in baseline

    slurm_dir = ROOT / "trial19_occam3_timing_sweep"
    files = sorted(slurm_dir.glob("h*.slurm"))
    assert len(files) == 10
    seen_outputs = {"replay_trial_19_simple4_s5_repro_fixed_500_gc_svelu_mirror"}
    found_pairs = []
    for path in files:
        match = re.fullmatch(r"h\d+_r(\d+)_f(\d+)\.slurm", path.name)
        assert match
        repair_start, finish_start = map(int, match.groups())
        found_pairs.append((repair_start, finish_start))
        content = path.read_text(encoding="utf-8")
        preset = f"S5_TIMING_PRESET=s5_timing_r{repair_start}_f{finish_start}"
        tag = f"S5_TIMING_TAG=r{repair_start}_f{finish_start}"
        assert preset in content
        assert tag in content
        assert "run_occam3_job.sh" in content
        output = f"replay_trial_19_s5_timing_r{repair_start}_f{finish_start}_fixed"
        assert output not in seen_outputs
        seen_outputs.add(output)
    assert tuple(found_pairs) == EXPECTED_PAIRS


if __name__ == "__main__":
    test_s5_timing_pairs_are_the_requested_ten()
    test_s5_baseline_has_the_expected_absolute_epoch_phases()
    test_r281_f441_is_epoch_by_epoch_exact_s5()
    test_each_timing_variant_preserves_the_s5_prefix()
    test_each_timing_variant_uses_the_s5_repair_objective()
    test_each_timing_variant_uses_the_s5_finalization_objective()
    test_every_timing_schedule_covers_each_epoch_once()
    test_both_launchers_save_snapshots_and_resumable_state_every_ten_epochs()
    test_all_sbatch_files_select_the_expected_unique_s5_runs()
    print("S5 timing schedules and launchers validated.")
