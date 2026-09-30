"""Compute-node provenance verification requires no Git executable."""

import json

import pytest

from launch_provenance import FLOOR, digest, verify


def fixture(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("ref: refs/heads/vxc_training\n")
    (tmp_path / ".git/refs/heads").mkdir(parents=True)
    (tmp_path / ".git/refs/heads/vxc_training").write_text("a" * 40)
    source = tmp_path / "model.py"
    source.write_text("x = 1\n")
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps(
            dict(
                root=str(tmp_path),
                git_commit="a" * 40,
                ancestry_floor=FLOOR,
                source_sha256={"model.py": digest(source)},
            )
        )
    )
    return record, source


def test_verifies_without_git(tmp_path, monkeypatch):
    record, _ = fixture(tmp_path)
    monkeypatch.setenv("PATH", "")
    assert verify(tmp_path, record) == "a" * 40


@pytest.mark.parametrize("change", ["source", "head", "floor"])
def test_rejects_changed_submission(tmp_path, change):
    record, source = fixture(tmp_path)
    if change == "source":
        source.write_text("x = 2\n")
    elif change == "head":
        (tmp_path / ".git/refs/heads/vxc_training").write_text("b" * 40)
    else:
        payload = json.loads(record.read_text())
        payload["ancestry_floor"] = "bad"
        record.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        verify(tmp_path, record)


def test_packed_refs(tmp_path):
    record, _ = fixture(tmp_path)
    (tmp_path / ".git/refs/heads/vxc_training").unlink()
    (tmp_path / ".git/packed-refs").write_text("a" * 40 + " refs/heads/vxc_training\n")
    assert verify(tmp_path, record) == "a" * 40
