# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""End-to-end conversions through the command line on the published data (images included), and the
known-pose triangulation with COLMAP (executable) and pycolmap."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA, OBJECT, sfm
from martine_converters import cli as convert
from martine_converters.colmap_model import read_model
from martine_converters.imageio import read_image, read_mask
from martine_converters.sfm import read_sfm
from method_code import methods

pytestmark = [pytest.mark.data, pytest.mark.slow]
VIEWS = "0:84:12"
COLMAP = os.environ.get("MARTINE_COLMAP") or shutil.which("colmap") or "colmap"


def run(*args):
    assert convert.main([str(a) for a in args]) == 0


@pytest.fixture(autouse=True)
def _needs_d4():
    sfm("mvs", "d4")  # every conversion here runs at d4: skip when that tier is not downloaded


@pytest.mark.parametrize("fmt", ["colmap", "colmap-mvs", "pgsr", "meshroom", "realityscan", "neus2", "neuralangelo"])
def test_mvs_cli_with_images(tmp_path, fmt):
    out = tmp_path / fmt
    run("mvs", "--data", DATA, "--object", OBJECT, "--tier", "d4", "--format", fmt, "--out", out, "--views", VIEWS,
        "--points", "none")
    man = json.loads((out / "martine_conversion.json").read_text())
    names = man["units"]
    assert len(names) == 7 and man["format"] == fmt
    s = read_sfm(sfm("mvs", "d4"))
    cam = s.unit(names[0]).camera
    if fmt in ("colmap", "colmap-mvs", "pgsr", "neuralangelo", "neus2"):
        img = read_image(out / "images" / f"{names[0]}.png")
        assert img.shape[:2] == (cam.height, cam.width) and img.dtype == np.uint8
        # NeuS2 and PGSR get RGBA with alpha = mask, as in the paper (PGSR render.py zeroes the TSDF depth outside it)
        assert img.shape[2] == (4 if fmt in ("neus2", "pgsr") else 3)
        m = read_mask(DATA / OBJECT / "mvs" / "d4" / "masks" / f"{names[0]}.png")
        if fmt in ("neus2", "pgsr"):
            assert np.array_equal(img[..., 3] > 0, m)
        if fmt != "neus2":
            assert img[~m][..., :3].max() == 0 and img[m][..., :3].mean() > 5
    if fmt in ("colmap", "colmap-mvs"):
        assert (out / "masks" / f"{names[0]}.png.png").is_file()
        assert (out / "sparse" / "0" / "cameras.txt").is_file()
    assert not list(out.rglob("*.sh"))  # inputs only
    if fmt == "meshroom":
        d = json.loads((out / "sfm.json").read_text())
        assert all(Path(v["path"]).is_file() and Path(v["path"]).is_absolute() for v in d["views"])
        assert (out / "masks" / f"{names[0]}.png").is_file()
    if fmt == "realityscan":
        for n in names:
            assert (out / "images" / f"{n}.xmp").is_file() and (out / "images" / f"{n}.exr").is_file()
            assert (out / "images" / f"{n}.exr.mask.png").is_file()
    if fmt in ("neus2", "neuralangelo"):
        assert (out / "normalization.json").is_file()


def test_ps_and_mvps_cli(tmp_path):
    sfm("mvps", "d4")
    out = tmp_path / "ps"
    run("ps", "--data", DATA, "--object", OBJECT, "--tier", "d4", "--method", "unimsps", "--out", out,
        "--views", "0:84:42", "--mask-dilate-d1", "5")
    man = json.loads((out / "martine_conversion.json").read_text())
    assert man["mask_dilate_px"] == 1 and man["mask_dilate_d1_px"] == 5  # 5 px at d1 = 1 px at d4
    poses = sorted(p.name for p in (out / "inputs").iterdir())
    assert len(poses) == 2 and len(list((out / "inputs" / poses[0]).glob("lighting_*.png"))) == 30
    # fake PS output: the normals of a sphere would need GT; use a constant camera-facing normal instead
    nd = tmp_path / "normals"
    nd.mkdir()
    s = read_sfm(sfm("mvps", "d4"))
    import cv2
    for p in poses:
        c = s.unit(p).camera
        n = np.zeros((c.height, c.width, 3), np.uint16)
        n[...] = [32768, 32768, 65535]
        cv2.imwrite(str(nd / f"{p}.png"), n)
    for fmt in ("rnbneus2", "supernormal"):
        o = tmp_path / fmt
        run("mvps", "--data", DATA, "--object", OBJECT, "--tier", "d4", "--format", fmt, "--normals", nd, "--out", o,
            "--views", "0:84:42", "--no-normal-check")
        assert len(json.loads((o / "views.json").read_text())) == 2


def test_cli_errors_are_reported(tmp_path, capsys):
    assert convert.main(["mvs", "--data", str(DATA), "--object", OBJECT, "--tier", "d4", "--format", "colmap",
                         "--out", str(tmp_path / "x"), "--views", "999", "--no-images", "--points", "none"]) == 2
    assert "out of range" in capsys.readouterr().err
    assert convert.main(["mvs", "--format", "colmap", "--out", str(tmp_path / "y")]) == 2  # no input given


def test_meshroom_default_keeps_raw_sfm_for_obj_export(tmp_path):
    """The Meshroom OBJ is W times its SfM world, so raw SfM yields an OBJ in the GT frame."""
    source = json.loads(Path(sfm("mvs", "d4")).read_text())
    raw_poses = {p["poseId"]: p["pose"]["transform"] for p in source["poses"]}
    out = tmp_path / "meshroom_default"
    run("mvs", "--data", DATA, "--object", OBJECT, "--tier", "d4", "--format", "meshroom", "--out", out,
        "--views", "0:84:42", "--points", "none", "--no-images")
    manifest = json.loads((out / "martine_conversion.json").read_text())
    converted = json.loads((out / "sfm.json").read_text())
    assert manifest["frame"] == "raw"
    for pose in converted["poses"]:
        expected = raw_poses[pose["poseId"]]
        actual = pose["pose"]["transform"]
        np.testing.assert_allclose(np.asarray(actual["center"], float), np.asarray(expected["center"], float))
        np.testing.assert_allclose(np.asarray(actual["rotation"], float), np.asarray(expected["rotation"], float))

    override = tmp_path / "meshroom_explicit_gt"
    run("mvs", "--data", DATA, "--object", OBJECT, "--tier", "d4", "--format", "meshroom", "--out", override,
        "--views", "0:84:42", "--points", "none", "--no-images", "--frame", "gt")
    assert json.loads((override / "martine_conversion.json").read_text())["frame"] == "gt"
    explicit_gt = json.loads((override / "sfm.json").read_text())
    W = np.diag([1.0, -1.0, -1.0])
    for pose in explicit_gt["poses"]:
        expected = raw_poses[pose["poseId"]]
        actual = pose["pose"]["transform"]
        np.testing.assert_allclose(np.asarray(actual["center"], float), W @ np.asarray(expected["center"], float))
        np.testing.assert_allclose(np.asarray(actual["rotation"], float).reshape(3, 3),
                                   W @ np.asarray(expected["rotation"], float).reshape(3, 3))


@pytest.mark.parametrize("backend", ["colmap", "pycolmap"])
def test_triangulation(tmp_path, backend, gt_points):
    if backend == "colmap" and not Path(COLMAP).exists():
        pytest.skip("COLMAP executable not found (MARTINE_COLMAP)")
    if backend == "pycolmap":
        pytest.importorskip("pycolmap")
    for fmt in ("colmap", "colmap-mvs"):
        out = tmp_path / f"{backend}_{fmt}"
        run("mvs", "--data", DATA, "--object", OBJECT, "--tier", "d4", "--format", fmt, "--out", out, "--views",
            "0:84:3", "--no-images", "--tri-backend", backend, "--colmap", COLMAP, "--tri-keep-work")
        cams, imgs, pts = read_model(out / "sparse" / "0")
        man = json.loads((out / "martine_conversion.json").read_text())
        tri = man["triangulation"]
        print(f"{backend} {fmt}: {tri}")
        assert len(pts) > 500
        # observations are consistent with the cameras of THIS model (keypoints shifted for colmap-mvs)
        X = np.array([p.xyz for p in pts.values()])
        loaded = methods.colmap(out / "sparse" / "0", dense=(fmt == "colmap-mvs"))
        errs = []
        for im in imgs.values():
            uv, _ = loaded[im.name.rsplit(".", 1)[0]](X[im.point3D_ids - 1])
            shift = 0.0 if fmt == "colmap-mvs" else 0.5
            errs.append(np.linalg.norm(uv - (im.xys - shift), axis=1))
        e = np.concatenate(errs)
        # distance to the GT (report; the points are only an initialisation)
        from scipy.spatial import cKDTree
        g = np.load(DATA / OBJECT / "gt" / "gt_pcd.npy", mmap_mode="r")
        d, _ = cKDTree(np.asarray(g[::5], float)).query(X)
        print(f"{backend} {fmt}: {len(X)} points, reprojection mean {e.mean():.3f} px (p99 {np.percentile(e, 99):.3f}),"
              f" GT distance median {np.median(d):.3f} mm, <= 1 mm {100 * np.mean(d <= 1):.1f} %")
        assert e.mean() < 1.0 and np.median(d) < 1.0


def test_meshroom_landmarks(tmp_path):
    """AliceVision's DepthMap needs landmarks (neighbours, depth range): the meshroom sfm.json carries the
    triangulated points, observations in the AliceVision pixel convention."""
    if not Path(COLMAP).exists():
        pytest.skip("COLMAP executable not found (MARTINE_COLMAP)")
    out = tmp_path / "meshroom"
    run("mvs", "--data", DATA, "--object", OBJECT, "--tier", "d4", "--format", "meshroom", "--out", out,
        "--views", "0:84:3", "--no-images", "--colmap", COLMAP)
    d = json.loads((out / "sfm.json").read_text())
    assert len(d["structure"]) > 500
    loaded = methods.meshroom(out / "sfm.json")
    stem = {v["viewId"]: Path(v["path"]).stem for v in d["views"]}
    errs = []
    for lm in d["structure"][::10]:
        X = np.array(lm["X"], float)[None]
        for ob in lm["observations"]:
            uv, _ = loaded[stem[ob["observationId"]]](X)
            errs.append(np.linalg.norm(uv[0] - np.array(ob["x"], float)))
    print(f"meshroom landmarks: {len(d['structure'])}, reprojection mean {np.mean(errs):.3f} px")
    assert np.mean(errs) < 1.0
