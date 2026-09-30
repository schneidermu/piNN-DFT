"""Exercise submission orchestration with fake executables only."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FLOOR = "29b36ac2faa29d31a758e0067499f10b41c3ab39"
JOBS = ["train_models/trial19_simple4_sota_sweep/s5_two_step_40_10.slurm"] + [
    f"train_models/trial19_occam3_timing_sweep/{name}"
    for name in (
        "h01_r241_f441.slurm",
        "h02_r261_f441.slurm",
        "h03_r221_f441.slurm",
        "h04_r281_f441.slurm",
        "h05_r241_f421.slurm",
        "h06_r261_f421.slurm",
        "h07_r281_f421.slurm",
        "h08_r221_f421.slurm",
        "h09_r241_f461.slurm",
        "h10_r261_f461.slurm",
    )
]


def run_fake(tmp_path, mode="ok"):
    repo = tmp_path / "checkout with spaces"
    repo.mkdir()
    wrapper = repo / "submit_dietclean_s5_chain.sh"
    wrapper.write_text((ROOT / wrapper.name).read_text())
    for name in ["train_models/prepare_dietclean_noval_v1.slurm", *JOBS]:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for name in ("data", "h5_vrho_from_mrks"):
        path = repo / "train_models" / name
        path.mkdir()
        (path / "input.h5").touch()
    if mode == "existing":
        (repo / "train_models/checkpoints_dietclean_noval_v1").mkdir()
    if mode == "missing":
        (repo / "train_models/data/input.h5").unlink()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/bash\nexit 0\n")
    git.chmod(0o755)
    fake = bin_dir / "sbatch"
    fake.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
p = Path(os.environ["CALLS"])
a = json.loads(p.read_text()) if p.exists() else []
a.append({"args":sys.argv[1:], "corpus":os.environ.get("CHECKPOINTS_DIR")})
p.write_text(json.dumps(a))
if os.environ["MODE"] == "failed": sys.exit(1)
print("invalid" if os.environ["MODE"] == "invalid" else f"{1000+len(a)};charisma")
""")
    fake.chmod(0o755)
    calls = tmp_path / "calls.json"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CALLS": str(calls),
        "MODE": mode,
    }
    result = subprocess.run(
        ["bash", str(wrapper)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    return result, json.loads(calls.read_text()) if calls.exists() else [], repo


def test_parallel_siblings_and_cluster_id_parsing(tmp_path):
    result, calls, repo = run_fake(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 12
    assert calls[0]["args"][-1] == "train_models/prepare_dietclean_noval_v1.slurm"
    assert [call["args"][-1] for call in calls[1:]] == JOBS
    for call in calls:
        assert call["args"][0] == "--parsable"
        assert call["corpus"] == str(
            repo / "train_models/checkpoints_dietclean_noval_v1"
        )
    for call in calls[1:]:
        assert [arg for arg in call["args"] if arg.startswith("--dependency")] == [
            "--dependency=afterok:1001"
        ]
        assert "--export=ALL,CHECKPOINTS_DIR" in call["args"]
    assert "CPU preprocessing: 1001" in result.stdout
    assert "1012" in result.stdout
    assert "afterany" not in (ROOT / "submit_dietclean_s5_chain.sh").read_text()


@pytest.mark.parametrize(
    "mode,count", [("existing", 0), ("missing", 0), ("failed", 1), ("invalid", 1)]
)
def test_failure_prevents_downstream_submissions(tmp_path, mode, count):
    result, calls, _ = run_fake(tmp_path, mode)
    assert result.returncode != 0
    assert len(calls) == count


def test_cpu_and_runner_contract():
    text = (ROOT / "train_models/prepare_dietclean_noval_v1.slurm").read_text()
    directives = "\n".join(
        line for line in text.splitlines() if line.startswith("#SBATCH")
    )
    assert 'constraint="type_d"' in directives
    assert "--nodes=1" in directives and "--ntasks=1" in directives
    assert "gpu" not in directives.lower() and "--gres" not in directives
    assert "conda activate ML_param" in text
    assert "set -euo pipefail" in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert text.count("python prepare_training_corpus.py") == 2
    assert text.index("--verify-only") > text.index("--mn-dir data")
    assert "rm -rf" not in text
    for name in (
        "trial19_simple4_sota_sweep/run_simple4_job.sh",
        "trial19_occam3_timing_sweep/run_occam3_job.sh",
    ):
        runner = (ROOT / "train_models" / name).read_text()
        assert FLOOR in runner
        assert "git merge-base --is-ancestor" in runner
        assert "prepare_training_corpus.py --verify-only" in runner
        assert "--force-preopt" in runner
