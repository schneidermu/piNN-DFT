"""Run every training test file in its own process.

Some historical tests install global sys.modules stubs during collection; a
single pytest invocation cannot reliably combine those files. This runner keeps
their existing isolation assumptions and runs all files, including Lap tests.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    env = dict(os.environ, OMP_NUM_THREADS="1")
    totals = {"passed": 0, "skipped": 0, "failed": 0, "error": 0}
    failures = []
    for path in sorted(directory.glob("test*.py")):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path.name, "-q"],
            cwd=directory,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        summary = result.stdout.strip().splitlines()[-1]
        print(f"{path.name}: {summary}", flush=True)
        for count, kind in re.findall(r"(\d+) (passed|skipped|failed|error)", summary):
            totals[kind] += int(count)
        if result.returncode not in (0, 5):
            failures.append(path.name)
            print(result.stdout, flush=True)
    print(f"Totals: {totals}; failed files: {failures}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
