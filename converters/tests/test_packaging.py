# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Packaging and public hygiene: the `martine-convert` command, the RealityScan convention used for the paper, no
path of the authors' machines in the code."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cli_module_entry_point():
    from martine_converters import cli
    assert cli.main(["list"]) == 0


def test_pyproject_declares_martine_convert():
    tomllib = pytest.importorskip("tomllib")
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert meta["project"]["scripts"]["martine-convert"] == "martine_converters.cli:main"
    assert meta["project"]["requires-python"] == ">=3.10"
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    assert (ROOT / "uv.lock").is_file()


def test_martine_convert_command_runs():
    exe = shutil.which("martine-convert", path=str(Path(sys.executable).parent))
    if exe is None:
        pytest.skip("package not installed in this environment (uv sync installs it)")
    r = subprocess.run([exe, "list"], capture_output=True, text=True)
    assert r.returncode == 0 and "realityscan" in r.stdout


def test_realityscan_convention_is_the_papers():
    from martine_converters import conventions
    text = conventions.describe()
    line = next(x for x in text.splitlines() if x.startswith("realityscan"))
    assert "+0.5" in line and "used for the paper" in line and "ASSUMPTION" not in line
    readme = (ROOT / "converters" / "README.md").read_text()
    assert "to be confirmed" not in readme.lower()


def test_no_private_paths_in_code():
    bad = re.compile(r"/home/babrument|/projects/m25115|\bm25115\b|mesonet")
    hits = [f"{p.relative_to(ROOT)}:{i}" for d in ("converters/martine_converters", "converters/tests", "scripts")
            for p in sorted((ROOT / d).rglob("*.py")) for i, line in enumerate(p.read_text().splitlines(), 1)
            if bad.search(line) and p.name != "test_packaging.py"]
    assert not hits, hits
