"""Capture Git provenance at submission; verify it without Git on compute nodes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FLOOR = "29b36ac2faa29d31a758e0067499f10b41c3ab39"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout_head(root: Path) -> str:
    git_dir = root / ".git"
    if git_dir.is_file():
        git_dir = (
            root / git_dir.read_text().strip().removeprefix("gitdir: ")
        ).resolve()
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    common = git_dir
    if (git_dir / "commondir").exists():
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    for directory in (git_dir, common):
        path = directory / ref
        if path.is_file():
            return path.read_text().strip()
    packed = common / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref:
                    return commit
    raise ValueError(f"Cannot resolve checkout HEAD: {ref}")


def capture(root: Path) -> Path:
    """Use login-node Git to check ancestry and snapshot tracked source files."""
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", FLOOR, "HEAD"], cwd=root, check=True
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    hashes = {
        name: digest(root / name)
        for name in names
        if name and Path(name).suffix in {".py", ".sh", ".slurm", ".csv"}
    }
    if not hashes:
        raise ValueError("No tracked launch/scientific sources found.")
    directory = root / "train_models/launch_manifests"
    directory.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="submission_", suffix=".json", dir=directory, delete=False
    ) as handle:
        json.dump(
            {
                "root": str(root),
                "git_commit": commit,
                "ancestry_floor": FLOOR,
                "source_sha256": hashes,
            },
            handle,
            indent=2,
        )
        return Path(handle.name)


def verify(root: Path, path: Path) -> str:
    """Fail if the checkout moved or tracked scientific/launch inputs changed."""
    record = json.loads(path.read_text())
    commit = record["git_commit"]
    if (
        record.get("root") != str(root)
        or record.get("ancestry_floor") != FLOOR
        or len(commit) != 40
        or checkout_head(root) != commit
        or not record.get("source_sha256")
    ):
        raise ValueError("Submission provenance does not match checkout HEAD/root.")
    for name, expected in record["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Source changed since submission: {name}")
    return commit


def current_commit(root: Path) -> str:
    """Verify the submitted snapshot, or use local Git outside batch launches."""
    path = os.environ.get("PINN_LAUNCH_PROVENANCE")
    if path:
        return verify(root, Path(path))
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if sys.argv[1:] == ["--capture"]:
        print(capture(root))
    else:
        print(verify(root, Path(os.environ["PINN_LAUNCH_PROVENANCE"])))
