# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Sparse points of the COLMAP/PGSR/Meshroom formats: the paper's strict protocol. One triangulation of the images
of the converted tier only, default SIFT, no fallback and no point threshold: the model is kept whatever its size
(an empty one included; the method then fails and the cell is missing)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA
from martine_converters import cli as convert
from martine_converters.colmap_model import ColmapPoint, read_model
from martine_converters.conventions import pixel_offset
from martine_converters.export import ConversionError, Options, TriOptions
from martine_converters.formats import colmap as colmap_fmt
from martine_converters.sfm import Scene, Unit, read_sfm, select_units
from martine_converters.triangulate import FeatureView, Triangulation, view_coverage
from method_code import methods
from synthetic import build, cameras

COLMAP = os.environ.get("MARTINE_COLMAP", shutil.which("colmap") or "colmap")
POINTS = np.array([[0.0, 0.0, 0.0], [10.0, -5.0, 3.0], [-8.0, 6.0, -4.0], [3.0, 12.0, 7.0]])
BASE = ["mvs", "--sfm", "x/sfm.json", "--format", "colmap", "--out", "o"]


def _scene(cams, path="obj/mvs/d4/sfm.json"):
    units = tuple(Unit(name=f"{k:03d}", camera=c, images=(), mask=None, pose_id=str(k), view_ids=(str(k),),
                       intrinsic_id="0") for k, c in enumerate(cams))
    return Scene(sfm_path=Path(path), modality="mvs", frame="gt", version=(1, 2, 12), units=units)


def _tri_seen_by_all(scene: Scene, X: np.ndarray = POINTS) -> Triangulation:
    """A triangulation of X observed in every view, observations in the COLMAP convention (AliceVision + 0.5)."""
    obs = {f"{u.name}.png": ([], []) for u in scene.units}
    points = {}
    for pid, x in enumerate(X, start=1):
        track = []
        for img_id, u in enumerate(scene.units, start=1):
            uv, _ = u.camera.project(x[None])
            xy, ids = obs[f"{u.name}.png"]
            xy.append(uv[0] + pixel_offset("colmap"))
            ids.append(pid)
            track.append((img_id, len(xy) - 1))
        points[pid] = ColmapPoint(pid, x, np.array([128, 128, 128]), 0.2, track)
    observations = {n: (np.array(xy, float).reshape(-1, 2), np.array(ids, np.int64)) for n, (xy, ids) in obs.items()}
    return Triangulation(points, observations, {"n_points": len(points)})


def _empty(scene: Scene) -> Triangulation:
    return Triangulation({}, {f"{u.name}.png": (np.zeros((0, 2)), np.zeros(0, np.int64)) for u in scene.units},
                         {"n_points": 0})


def _two_tier_dataset(root: Path) -> Path:
    for t in ("d4", "d8"):
        build(root / "obj" / "mvps" / t, n=6, lights=2)
    return root / "obj" / "mvps" / "d8" / "sfm.json"


# --- strict protocol -----------------------------------------------------------------------------------------------
def test_single_attempt_at_the_converted_tier(tmp_path, monkeypatch):
    """Nothing triangulated at d8: no retry at d4 nor with another SIFT, the empty result is returned as is."""
    target = read_sfm(_two_tier_dataset(tmp_path))
    calls = []

    def fake(scene, opts, work):
        calls.append((Path(scene.sfm_path).parent.name, opts.tri.peak_threshold))
        return _empty(scene)

    monkeypatch.setattr(colmap_fmt, "_triangulate_once", fake)
    tri = colmap_fmt.run_triangulation(target, Options(), tmp_path / "work")
    assert calls == [("d8", None)]
    assert len(tri.points) == 0
    assert tri.stats["n_points"] == 0 and tri.stats["views_without_points"] == 6
    assert tri.stats["source_tier"] == "d8" and "attempts" not in tri.stats


def test_few_points_are_kept(tmp_path, monkeypatch):
    target = read_sfm(_two_tier_dataset(tmp_path))
    monkeypatch.setattr(colmap_fmt, "_triangulate_once", lambda scene, opts, work: _tri_seen_by_all(scene, POINTS[:1]))
    tri = colmap_fmt.run_triangulation(target, Options(), tmp_path / "work")
    assert len(tri.points) == 1 and tri.stats["min_view_points"] == 1


def test_colmap_failure_is_not_retried(tmp_path, monkeypatch):
    target = read_sfm(_two_tier_dataset(tmp_path))
    calls = []

    def fake(scene, opts, work):
        calls.append(Path(scene.sfm_path).parent.name)
        raise ConversionError("colmap point_triangulator failed")

    monkeypatch.setattr(colmap_fmt, "_triangulate_once", fake)
    with pytest.raises(ConversionError, match="point_triangulator"):
        colmap_fmt.run_triangulation(target, Options(), tmp_path / "work")
    assert calls == ["d8"]


def test_too_few_views_for_a_track_is_refused_at_once(tmp_path, monkeypatch):
    target = select_units(read_sfm(_two_tier_dataset(tmp_path)), "0:2")
    monkeypatch.setattr(colmap_fmt, "_triangulate_once", lambda *a: pytest.fail("no triangulation expected"))
    with pytest.raises(ConversionError, match="min_track|views"):
        colmap_fmt.run_triangulation(target, Options(tri=TriOptions(min_track=3)), tmp_path / "work")


def test_empty_model_is_written(tmp_path, monkeypatch):
    """The CLI writes the (empty) sparse model and records it in the manifest instead of failing."""
    sfm = _two_tier_dataset(tmp_path)
    monkeypatch.setattr(colmap_fmt, "_triangulate_once", lambda scene, opts, work: _empty(scene))
    out = tmp_path / "out"
    assert convert.main(["mvs", "--sfm", str(sfm), "--format", "colmap-mvs", "--out", str(out), "--no-images"]) == 0
    man = json.loads((out / "martine_conversion.json").read_text())
    assert man["triangulation"]["n_points"] == 0 and man["triangulation"]["source_tier"] == "d8"
    _, imgs, pts = read_model(out / "sparse" / "0")
    assert len(imgs) == 6 and len(pts) == 0


def test_no_fallback_options():
    for flag in (["--tri-tier", "d4"], ["--tri-min-points", "0"], ["--tri-min-view-points", "0"],
                 ["--tri-fallback-peak-threshold", "0"]):
        with pytest.raises(SystemExit):
            convert.build_parser().parse_args(BASE + flag)
    fields = set(TriOptions.__dataclass_fields__)
    assert not fields & {"tier", "min_points", "min_view_points", "fallback_peak_threshold"}


def test_default_sift():
    t = convert._options(convert.build_parser().parse_args(BASE)).tri
    assert t.peak_threshold is None and not t.domain_size_pooling and t.max_image_size == 3200


def test_view_coverage_counts_points_per_view():
    s = _scene(cameras(6))
    assert view_coverage(_tri_seen_by_all(s), s) == {"n_points": 4, "min_view_points": 4, "views_without_points": 0}


# --- explicit SIFT settings (off by default, not used for the paper) ------------------------------------------------
class _Stop(Exception):
    pass


def _feature_views(n=3):
    return [FeatureView(f"{u.name}.png", u.camera) for u in _scene(cameras(n)).units]


def test_sift_options_reach_the_colmap_executable(tmp_path, monkeypatch):
    from martine_converters import triangulate as T
    cmds = []

    def fake_run(cmd, env, log):
        cmds.append([str(c) for c in cmd])
        raise _Stop

    monkeypatch.setattr(T, "_run", fake_run)
    monkeypatch.setattr(T, "_option_prefixes", lambda exe, env: ("SiftExtraction", "SiftMatching"))
    monkeypatch.setattr(T.shutil, "which", lambda exe: "/usr/bin/colmap")
    opts = TriOptions(backend="colmap", colmap="colmap", peak_threshold=0.002, domain_size_pooling=True)
    with pytest.raises(_Stop):
        T.triangulate(_feature_views(), tmp_path, None, tmp_path / "w", opts)
    cmd = cmds[0]
    assert cmd[1] == "feature_extractor"
    assert float(cmd[cmd.index("--SiftExtraction.peak_threshold") + 1]) == 0.002
    assert cmd[cmd.index("--SiftExtraction.domain_size_pooling") + 1] == "1"
    cmds.clear()
    with pytest.raises(_Stop):  # defaults: COLMAP's own values, nothing passed
        T.triangulate(_feature_views(), tmp_path, None, tmp_path / "w2", TriOptions(backend="colmap", colmap="colmap"))
    assert not any("peak_threshold" in c or "domain_size_pooling" in c for c in cmds[0])


def test_sift_options_reach_pycolmap(tmp_path, monkeypatch):
    pycolmap = pytest.importorskip("pycolmap")
    from martine_converters import triangulate as T
    seen = {}

    def fake_extract(*args, extraction_options=None, **kw):
        seen["sift"] = extraction_options.sift
        raise _Stop

    monkeypatch.setattr(pycolmap, "extract_features", fake_extract)
    opts = TriOptions(backend="pycolmap", peak_threshold=0.002, domain_size_pooling=True)
    with pytest.raises(_Stop):
        T.triangulate(_feature_views(), tmp_path, None, tmp_path / "w", opts)
    assert seen["sift"].peak_threshold == pytest.approx(0.002) and seen["sift"].domain_size_pooling


def test_cli_sift_options():
    a = convert.build_parser().parse_args(BASE + ["--tri-peak-threshold", "0.002", "--tri-dsp",
                                                  "--tri-max-image-size", "4800"])
    t = convert._options(a).tri
    assert t.peak_threshold == 0.002 and t.domain_size_pooling and t.max_image_size == 4800


# --- published data ------------------------------------------------------------------------------------------------
@pytest.mark.data
@pytest.mark.slow
def test_vase_d8_triangulates_at_d8_only(tmp_path):
    """vase d8 (weak texture): one attempt with the d8 images, whatever the number of points; the points that exist
    reproject into the d8 cameras (COLMAP dense convention)."""
    if not (DATA / "vase" / "mvs" / "d8" / "sfm.json").is_file():
        pytest.skip("published vase d8 not available")
    out = tmp_path / "vase_d8"
    assert convert.main(["mvs", "--data", str(DATA), "--object", "vase", "--tier", "d8", "--format", "colmap-mvs",
                         "--out", str(out), "--no-images", "--colmap", COLMAP, "--tri-threads", "8"]) == 0
    tri = json.loads((out / "martine_conversion.json").read_text())["triangulation"]
    print("vase d8:", tri)
    assert tri["source_tier"] == "d8" and tri["peak_threshold"] is None and "attempts" not in tri
    cams, imgs, pts = read_model(out / "sparse" / "0")
    assert all(c.width == 1196 for c in cams.values())
    assert len(pts) == tri["n_points"]
    if pts:
        X = np.array([p.xyz for p in pts.values()])
        loaded = methods.colmap(out / "sparse" / "0", dense=True)
        errs = np.concatenate([np.linalg.norm(loaded[im.name.rsplit(".", 1)[0]](X[im.point3D_ids - 1])[0] - im.xys,
                                              axis=1) for im in imgs.values() if len(im.point3D_ids)])
        assert errs.mean() < 1.0
