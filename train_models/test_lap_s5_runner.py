"""Portable SCF runtime selection tests for the local Lap-S5 runner."""

from pathlib import Path

import pytest
import torch
import train_lap_s5 as runner
from train_lap_s5 import build_scf_command, windows_path_to_wsl


def test_s5_model_uses_shared_factory_with_explicit_lap_protocol(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_build_model(args, device):
        captured["args"] = args
        captured["device"] = device
        return sentinel

    monkeypatch.setattr(runner, "build_model", fake_build_model)
    model = runner._build_s5_lap_model("cpu", torch.float32)

    assert model is sentinel
    assert captured["args"].name == "PBE-Lap-LGxGc_6_32"
    assert captured["args"].model_type == "lap"
    assert captured["args"].dropout == 0.0
    assert captured["args"].dtype is torch.float32
    assert captured["device"] == "cpu"


def test_windows_paths_map_to_wsl_mounts():
    assert (
        windows_path_to_wsl(r"C:\Dev\pilot\checkpoint.pt")
        == "/mnt/c/Dev/pilot/checkpoint.pt"
    )
    assert windows_path_to_wsl(r"D:\data\H2\source.npz") == "/mnt/d/data/H2/source.npz"


def test_unc_paths_fail_closed_for_wsl_scfs():
    with pytest.raises(ValueError, match="Cannot map this path"):
        windows_path_to_wsl(r"\\server\share\checkpoint.pt")


def test_wsl_scf_command_uses_converted_inputs_and_output():
    command = build_scf_command(
        "wsl",
        Path(r"C:\Dev\repo\train_models\run_lap_scf_real_smoke.py"),
        Path(r"C:\Dev\pilot\lap_s5_phase_smoke.pt"),
        Path(r"C:\Users\schne\Downloads\H2.npz"),
        Path(r"C:\Dev\pilot\scf_smoke.json"),
    )

    assert command == [
        "wsl.exe",
        "-e",
        "python3",
        "/mnt/c/Dev/repo/train_models/run_lap_scf_real_smoke.py",
        "--checkpoint",
        "/mnt/c/Dev/pilot/lap_s5_phase_smoke.pt",
        "--npz",
        "/mnt/c/Users/schne/Downloads/H2.npz",
        "--output",
        "/mnt/c/Dev/pilot/scf_smoke.json",
        "--grid-level",
        "1",
    ]
