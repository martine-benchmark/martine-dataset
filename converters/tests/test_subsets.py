# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""View subsets (--n-views, `martine-convert subsets`): k-means on the MVS camera centres, MVPS poses paired to the
nearest selected MVS view without duplicate; independent of the tier. The subsets of the published objects for
N = 5, 10, 20, 40 are precomputed in the package (view_subsets.json), so MVPS alone suffices for them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA, MVPS_TIERS, MVS_TIERS, OBJECT
from martine_converters import cli as convert
from martine_converters import subsets
from martine_converters.export import MANIFEST
from martine_converters.sfm import SfmError
from martine_converters.subsets import kmeans_views, load_table, locate_pair, match_views, paired_subsets

W, H, F_MM = 640, 480, 50.0


def ring(n, radius=700.0, phase=0.0, z_amp=150.0):
    a = 2 * np.pi * np.arange(n) / n + phase
    return np.stack([radius * np.cos(a), radius * np.sin(a), z_amp * np.sin(3 * a)], axis=1)


def write_sfm(path: Path, centers, modality: str, scale: int = 1, lights: int = 3) -> None:
    """Minimal published-like sfm.json (images need not exist): MVS views NNN.exr, MVPS images/pose_KK/lighting_LL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    intr = {"intrinsicId": "1", "width": str(W // scale), "height": str(H // scale), "sensorWidth": "36",
            "type": "pinhole", "focalLength": repr(F_MM), "pixelRatio": "1", "principalPoint": ["0", "0"],
            "distortionType": "none", "distortionParams": ""}
    poses, views = [], []
    for k, c in enumerate(centers):
        pid = str(1000 + 7 * k)  # hash-like ids, not in capture order
        poses.append({"poseId": pid, "pose": {"transform": {"rotation": [repr(float(x)) for x in np.eye(3).ravel()],
                                                            "center": [repr(float(x)) for x in c]}}})
        paths = [f"images/{k:03d}.exr"] if modality == "mvs" else \
            [f"images/pose_{k:02d}/lighting_{li:02d}.exr" for li in range(lights)]
        for li, p in enumerate(paths):
            views.append({"viewId": str(50000 + 10 * k + li), "poseId": pid, "intrinsicId": "1", "path": p})
    path.write_text(json.dumps({"version": ["1", "2", "12"], "intrinsics": [intr], "poses": poses, "views": views}))


@pytest.fixture
def dataset(tmp_path):
    """<root>/obj/{mvs,mvps}/{d4,d8}/sfm.json; the MVPS poses are near, not on, the MVS views."""
    rng = np.random.default_rng(0)
    mvs = ring(48)
    mvps = ring(40, phase=0.05) + rng.normal(0, 3, (40, 3))
    for tier, scale in (("d4", 1), ("d8", 2)):
        write_sfm(tmp_path / "obj" / "mvs" / tier / "sfm.json", mvs, "mvs", scale)
        write_sfm(tmp_path / "obj" / "mvps" / tier / "sfm.json", mvps, "mvps", scale)
    return tmp_path, mvs, mvps


def test_kmeans_views_gives_n_distinct_views_reproducibly():
    centers = ring(48)
    sel = kmeans_views(centers, 5)
    assert len(sel) == len(set(sel)) == 5 and list(sel) == sorted(sel)
    assert list(kmeans_views(centers, 5)) == list(sel)
    gaps = np.diff(sorted(sel) + [sorted(sel)[0] + 48])
    assert gaps.min() >= 6  # spread around the ring, not clustered


def test_kmeans_views_bounds():
    centers = ring(10)
    assert list(kmeans_views(centers, 10)) == list(range(10))
    for n in (0, 11):
        with pytest.raises(ValueError):
            kmeans_views(centers, n)


def test_match_views_is_nearest_without_duplicate():
    src = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    dst = np.array([[0.4, 0, 0], [10.0, 0, 0], [0.6, 0, 0]])  # both src are nearest to dst[0] or dst[2]
    idx, dist = match_views(src, dst)
    assert list(idx) == [0, 2] and np.allclose(dist, [0.4, 0.4])
    with pytest.raises(ValueError):
        match_views(dst, src)


def test_paired_subsets_follow_the_mvs_kmeans(dataset):
    root, mvs, mvps = dataset
    res = paired_subsets(root / "obj" / "mvs" / "d4" / "sfm.json", root / "obj" / "mvps" / "d4" / "sfm.json", 10)
    assert len(res["mvs"]) == len(res["mvps"]) == len(set(res["mvps"])) == len(res["distance_mm"]) == 10
    sel = kmeans_views(mvs, 10)
    assert res["mvs"] == [f"{k:03d}" for k in sel]
    for name_s, name_p, d in zip(res["mvs"], res["mvps"], res["distance_mm"]):
        expected = np.linalg.norm(mvs[int(name_s)] - mvps[int(name_p[5:])])
        assert d == pytest.approx(expected, abs=1e-9)
        assert d <= np.linalg.norm(mvps - mvs[int(name_s)], axis=1).min() + 60  # near, ring step ~110 mm


def test_subsets_do_not_depend_on_the_tier(dataset):
    root = dataset[0]
    lists = [paired_subsets(root / "obj" / "mvs" / t / "sfm.json", root / "obj" / "mvps" / t / "sfm.json", 5)
             for t in ("d4", "d8")]
    assert lists[0] == lists[1]


def test_cli_subsets_prints_and_writes(dataset, tmp_path, capsys):
    root = dataset[0]
    out = tmp_path / "subsets_v5.json"
    assert convert.main(["subsets", "--data", str(root), "--object", "obj", "--n-views", "5", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res == paired_subsets(root / "obj" / "mvs" / "d4" / "sfm.json", root / "obj" / "mvps" / "d4" / "sfm.json",
                                 5) | {"object": "obj", "n_views": 5, "source": "computed"}
    text = capsys.readouterr().out
    assert f"--views {','.join(res['mvs'])}" in text and f"--views {','.join(sorted(res['mvps']))}" in text
    assert "mm" in text


@pytest.mark.parametrize("modality", ["mvs", "mvps"])
def test_converters_n_views_selects_the_paired_subset(dataset, tmp_path, modality):
    """Same _scene code path for every command (mvs, ps, mvps, check-normals); cameras only here."""
    root = dataset[0]
    out = tmp_path / modality
    assert convert.main(["mvs", "--data", str(root), "--object", "obj", "--tier", "d8", "--modality", modality,
                         "--n-views", "5", "--format", "colmap", "--points", "none", "--no-images",
                         "--out", str(out)]) == 0
    expected = paired_subsets(root / "obj" / "mvs" / "d4" / "sfm.json", root / "obj" / "mvps" / "d4" / "sfm.json", 5)
    assert json.loads((out / MANIFEST).read_text())["units"] == sorted(expected[modality])


def test_n_views_and_views_are_exclusive(dataset, tmp_path):
    root = dataset[0]
    with pytest.raises(SystemExit) as e:
        convert.main(["mvs", "--data", str(root), "--object", "obj", "--tier", "d4", "--format", "colmap",
                      "--n-views", "5", "--views", "0:3", "--out", str(tmp_path / "x")])
    assert e.value.code == 2


def test_n_views_too_large_is_an_error(dataset, tmp_path, capsys):
    root = dataset[0]
    assert convert.main(["subsets", "--data", str(root), "--object", "obj", "--n-views", "41"]) == 2
    assert "41" in capsys.readouterr().err


def test_mvps_with_n_views_needs_the_mvs_sfm(tmp_path):
    write_sfm(tmp_path / "alone" / "sfm.json", ring(12), "mvps")
    with pytest.raises(SfmError):
        locate_pair(tmp_path / "alone" / "sfm.json", "mvs")


@pytest.mark.data
@pytest.mark.parametrize("n", [5, 10, 20, 40])
def test_published_subsets_are_the_same_at_every_tier(n):
    lists = {(tm, tp): paired_subsets(DATA / OBJECT / "mvs" / tm / "sfm.json", DATA / OBJECT / "mvps" / tp / "sfm.json",
                                      n) for tm in MVS_TIERS for tp in MVPS_TIERS}
    first = next(iter(lists.values()))
    assert all(v == first for v in lists.values()), {k: v["mvs"] for k, v in lists.items()}
    assert len(set(first["mvps"])) == n


def no_deps():
    raise ValueError("scikit-learn and scipy must not be needed here")


def test_table_covers_the_published_objects():
    table = load_table()
    assert table["n_views"] == [5, 10, 20, 40] and len(table["objects"]) == 28
    for obj, by_n in table["objects"].items():
        assert sorted(by_n, key=int) == ["5", "10", "20", "40"], obj
        for n, e in by_n.items():
            assert len(e["mvs"]) == len(set(e["mvs"])) == len(set(e["mvps"])) == len(e["distance_mm"]) == int(n), obj


def test_mvps_alone_uses_the_table(tmp_path, monkeypatch):
    """A user who downloaded the MVPS images only: no MVS sfm.json, no scikit-learn or scipy."""
    monkeypatch.setattr(subsets, "_deps", no_deps)
    write_sfm(tmp_path / "rock" / "mvps" / "d8" / "sfm.json", ring(84), "mvps")
    out = tmp_path / "out"
    assert convert.main(["mvs", "--data", str(tmp_path), "--object", "rock", "--tier", "d8", "--modality", "mvps",
                         "--n-views", "10", "--format", "colmap", "--points", "none", "--no-images",
                         "--out", str(out)]) == 0
    expected = load_table()["objects"]["rock"]["10"]["mvps"]
    assert json.loads((out / MANIFEST).read_text())["units"] == sorted(expected)
    # same with --sfm: the object is read from the published layout
    assert subsets.subset_names(tmp_path / "rock" / "mvps" / "d8" / "sfm.json", 10) == sorted(expected)


def test_cli_subsets_from_the_table_needs_no_data(capsys, monkeypatch):
    monkeypatch.setattr(subsets, "_deps", no_deps)
    assert convert.main(["subsets", "--object", "rock", "--n-views", "5"]) == 0
    e = load_table()["objects"]["rock"]["5"]
    text = capsys.readouterr().out
    assert f"--views {','.join(e['mvs'])}" in text and f"--views {','.join(sorted(e['mvps']))}" in text


def test_other_n_without_mvs_is_a_clear_error(tmp_path, capsys):
    write_sfm(tmp_path / "rock" / "mvps" / "d8" / "sfm.json", ring(84), "mvps")
    assert convert.main(["mvs", "--data", str(tmp_path), "--object", "rock", "--tier", "d8", "--modality", "mvps",
                         "--n-views", "7", "--format", "colmap", "--points", "none", "--no-images",
                         "--out", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert "precomputed" in err and "5, 10, 20, 40" in err and "mvs" in err


def test_regenerate_writes_the_table_format(dataset, tmp_path):
    root = dataset[0]
    out = tmp_path / "table.json"
    assert convert.main(["subsets", "--data", str(root), "--regenerate", str(out)]) == 0
    t = json.loads(out.read_text())
    assert t["n_views"] == [5, 10, 20, 40] and list(t["objects"]) == ["obj"]
    ref = paired_subsets(root / "obj" / "mvs" / "d4" / "sfm.json", root / "obj" / "mvps" / "d4" / "sfm.json", 20)
    assert t["objects"]["obj"]["20"]["mvs"] == ref["mvs"] and t["objects"]["obj"]["20"]["mvps"] == ref["mvps"]


@pytest.mark.data
@pytest.mark.slow
def test_table_equals_the_recomputation():
    table = load_table()
    objs = sorted(p.name for p in DATA.iterdir() if (p / "mvs").is_dir() and (p / "mvps").is_dir())
    assert sorted(table["objects"]) == objs
    for obj in objs:
        for n in table["n_views"]:
            ref = paired_subsets(DATA / obj / "mvs" / "d8" / "sfm.json", DATA / obj / "mvps" / "d8" / "sfm.json", n)
            e = table["objects"][obj][str(n)]
            assert (e["mvs"], e["mvps"]) == (ref["mvs"], ref["mvps"]), (obj, n)
            assert e["distance_mm"] == pytest.approx(ref["distance_mm"], abs=0.006), (obj, n)
