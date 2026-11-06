# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Test configuration.

* ``MARTINE_DATA``: root of the downloaded dataset; the tests that need it are skipped when it is not set.
* ``MARTINE_TEST_OBJECT``: object used by the real-data tests (default rock).
* ``MARTINE_METHOD_REPOS``: folder of the method checkouts for the real-code tests (see method_code/real_code.py).
* ``MARTINE_COLMAP``: COLMAP executable for the triangulation tests.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

DATA = Path(os.environ.get("MARTINE_DATA") or "MARTINE_DATA-not-set")  # tests needing it are skipped
OBJECT = os.environ.get("MARTINE_TEST_OBJECT", "rock")


def _published_tiers(modality):
    d = DATA / OBJECT / modality
    tiers = [p.name for p in d.glob("d*") if p.name[1:].isdigit() and (p / "sfm.json").is_file()] if d.is_dir() else []
    return sorted(tiers, key=lambda t: int(t[1:]))


# every published tier is tested (discovered on disk: d1/d2/d4/d8 MVS, d2/d4/d8 MVPS once d8 is out)
MVS_TIERS = tuple(_published_tiers("mvs") or ["d1", "d2", "d4"])
MVPS_TIERS = tuple(_published_tiers("mvps") or ["d2", "d4"])


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the published dataset (MARTINE_DATA)")
    config.addinivalue_line("markers", "slow: long (triangulation, image conversion)")


def pytest_runtest_setup(item):
    if item.get_closest_marker("data") is not None and not DATA.is_dir():
        pytest.skip("MARTINE_DATA not set: the published dataset is needed")


def sfm(modality: str, tier: str) -> Path:
    p = DATA / OBJECT / modality / tier / "sfm.json"
    if not p.is_file():
        pytest.skip(f"published data not available: {p}")
    return p


@pytest.fixture(scope="session")
def gt_points() -> np.ndarray:
    p = DATA / OBJECT / "gt" / "gt_pcd.npy"
    if not p.is_file():
        pytest.skip(f"no {p}")
    pts = np.load(p, mmap_mode="r")
    idx = np.random.default_rng(0).choice(len(pts), size=2000, replace=False)
    return np.asarray(pts[np.sort(idx)], dtype=np.float64)
