"""The examples fly in real time (too slow for CI): here we only check that they start.
To try them: python examples/<name>.py --no-window"""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted(p for p in (Path(__file__).resolve().parent.parent / "examples").glob("*.py") if p.name != "common.py")


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_starts(path):
    r = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--real" in r.stdout
