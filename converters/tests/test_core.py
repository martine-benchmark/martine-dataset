# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Unit tests of the core: conventions, camera, sfm reading, COLMAP text model, tone maps, box downscale."""
from __future__ import annotations

import json

import numpy as np
import pytest
from martine_converters import conventions
from martine_converters.camera import W_CORR, Camera, qvec_to_rotmat, rotmat_to_qvec
from martine_converters.colmap_model import ColmapCamera, ColmapImage, ColmapPoint, read_model, write_model
from martine_converters.imageio import tonemap
from martine_converters.sfm import SfmError, intrinsic_to_pixels, read_sfm, select_units


def rand_rot(rng):
    q = rng.normal(size=4)
    return qvec_to_rotmat(q / np.linalg.norm(q))


# --- conventions ---------------------------------------------------------------------------------------------------
def test_convention_table():
    expected = {"alicevision": 0.0, "colmap": 0.5, "colmap-mvs": 0.0, "pgsr": 0.5, "meshroom": 0.0,
                "realityscan": 0.5, "neus2": 0.5, "neuralangelo": 0.5, "rnbneus2": 0.5, "supernormal": 0.0}
    assert {k: v.offset for k, v in conventions.CONVENTIONS.items()} == expected
    assert not conventions.CONVENTIONS["realityscan"].verified
    assert all(c.verified for k, c in conventions.CONVENTIONS.items() if k != "realityscan")


def test_pixel_offset_override():
    assert conventions.pixel_offset("realityscan", 0.0) == 0.0
    assert conventions.pixel_offset("neus2") == 0.5
    with pytest.raises(ValueError):
        conventions.pixel_offset("neus2", 0.25)
    with pytest.raises(KeyError):
        conventions.pixel_offset("gomvs")


# --- camera ----------------------------------------------------------------------------------------------------------
def test_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        R = rand_rot(rng)
        q = rotmat_to_qvec(R)
        assert q[0] >= 0 and abs(np.linalg.norm(q) - 1) < 1e-12
        assert np.allclose(qvec_to_rotmat(q), R, atol=1e-12)
    for R in (np.eye(3), np.diag([1, -1, -1.0]), np.diag([-1, 1, -1.0]), np.diag([-1, -1, 1.0])):
        assert np.allclose(qvec_to_rotmat(rotmat_to_qvec(R)), R, atol=1e-12)


def test_camera_projection_and_frames():
    rng = np.random.default_rng(2)
    R, C = rand_rot(rng), rng.normal(size=3) * 100
    cam = Camera(100, 80, 120.0, 121.0, 49.7, 40.2, R, C)
    X = C + (R.T @ np.array([3.0, -2.0, 50.0]))[None]
    uv, z = cam.project(X)
    assert np.allclose(uv, [[120 * 3 / 50 + 49.7, 121 * -2 / 50 + 40.2]]) and np.allclose(z, 50)
    assert np.allclose(cam.projection(0.5)[:, :3] @ X[0] + cam.projection(0.5)[:, 3],
                       np.array([*(uv[0] + 0.5), 1.0]) * 50)
    g = cam.transformed(W_CORR)
    assert np.allclose(g.project(X @ W_CORR.T)[0], uv)
    with pytest.raises(ValueError):
        Camera(1, 1, 1, 1, 0, 0, np.diag([1, 1, -1.0]), np.zeros(3))
    assert not cam.R.flags.writeable


# --- sfm -------------------------------------------------------------------------------------------------------------
def _intr(**kw):
    d = {"intrinsicId": "1", "width": "4000", "height": "3000", "sensorWidth": "36", "type": "pinhole",
         "focalLength": "50", "pixelRatio": "1", "principalPoint": ["1.5", "-2.25"], "distortionType": "none",
         "distortionParams": ""}
    d.update(kw)
    return d


def test_intrinsics_versions():
    w, h, fx, fy, cx, cy = intrinsic_to_pixels(_intr(), (1, 2, 12))
    assert (w, h) == (4000, 3000) and fx == fy == 50 * 4000 / 36 and (cx, cy) == (2001.5, 1497.75)
    _, _, fx, fy, _, _ = intrinsic_to_pixels(_intr(pixelRatio="2"), (1, 2, 12))
    assert fy == pytest.approx(50 * 4000 / 36) and fx == pytest.approx(fy / 2)  # IntrinsicScaleOffset.cpp:228-234
    _, _, fx, fy, _, _ = intrinsic_to_pixels(_intr(pixelRatio="2"), (1, 2, 8))
    assert fx == pytest.approx(50 * 4000 / 36) and fy == pytest.approx(2 * fx)  # compatibility mode
    _, _, fx, _, cx, cy = intrinsic_to_pixels(_intr(principalPoint=["2001.5", "1497.75"], pxFocalLength="1000"),
                                              (1, 2, 0))
    assert (fx, cx, cy) == (1000.0, 2001.5, 1497.75)  # pixel focal and absolute principal point before 1.2.1
    with pytest.raises(SfmError):
        intrinsic_to_pixels(_intr(distortionType="radialk3", distortionParams=["0.1", "0", "0"]), (1, 2, 12))
    with pytest.raises(SfmError):
        intrinsic_to_pixels(_intr(type="equidistant"), (1, 2, 12))


def write_fake_sfm(tmp_path, n_poses=3, lights=1, rng=None):
    """Small sfm.json in the published layout (MVS if lights == 1, MVPS otherwise); returns path and raw poses."""
    rng = rng or np.random.default_rng(3)
    views, poses, raw = [], [], {}
    for k in range(n_poses):
        R_c2w = rand_rot(rng)
        C = rng.normal(size=3) * 500
        pid = str(1000 + 17 * k)
        poses.append({"poseId": pid, "pose": {"transform": {"rotation": [repr(float(x)) for x in R_c2w.reshape(-1)],
                                                            "center": [repr(float(x)) for x in C]}}})
        raw[pid] = (R_c2w, C)
        for li in range(lights):
            path = f"images/{k:03d}.exr" if lights == 1 else f"images/pose_{k:02d}/lighting_{li:02d}.exr"
            views.append({"viewId": str(5000 + 31 * k + li), "poseId": pid, "intrinsicId": "1", "path": path,
                          "width": "4000", "height": "3000"})
    data = {"version": ["1", "2", "12"], "intrinsics": [_intr()], "poses": poses, "views": views[::-1]}
    p = tmp_path / "sfm.json"
    p.write_text(json.dumps(data))
    return p, raw


def test_read_sfm_mvs_and_mvps(tmp_path):
    p, raw = write_fake_sfm(tmp_path, n_poses=3)
    s = read_sfm(p)
    assert s.modality == "mvs" and s.names == ["000", "001", "002"]
    R_c2w, C = raw[s.units[0].pose_id]
    cam = s.units[0].camera
    assert np.allclose(cam.R, (W_CORR @ R_c2w).T) and np.allclose(cam.C, W_CORR @ C)
    raw_cam = read_sfm(p, frame="raw").units[0].camera
    assert np.allclose(raw_cam.R, R_c2w.T) and np.allclose(raw_cam.C, C)
    assert (cam.cx, cam.cy) == (2001.5, 1497.75)
    (tmp_path / "mvps").mkdir()
    q, _ = write_fake_sfm(tmp_path / "mvps", n_poses=2, lights=3)
    s2 = read_sfm(q)
    assert s2.modality == "mvps" and s2.names == ["pose_00", "pose_01"]
    assert s2.units[0].image_keys == ("lighting_00", "lighting_01", "lighting_02")
    vids = {v["viewId"]: v["path"] for v in json.loads(q.read_text())["views"]}
    for u in s2.units:  # view ids follow the image order (the file lists the views in reverse)
        assert all(vids[vid].endswith(f"{u.name}/{k}.exr") for vid, k in zip(u.view_ids, u.image_keys))


def test_select_units(tmp_path):
    p, _ = write_fake_sfm(tmp_path, n_poses=6)
    s = read_sfm(p)
    assert select_units(s, "0:6:2").names == ["000", "002", "004"]
    assert select_units(s, "001,5").names == ["001", "005"]
    with pytest.raises(SfmError):
        select_units(s, "nope")


@pytest.mark.parametrize("fmt", ["colmap", "colmap-mvs", "pgsr", "meshroom", "realityscan"])  # NeRF formats need masks
def test_converters_write_inputs_only(tmp_path, fmt):
    """The converters write the inputs of a method, never a script that runs it."""
    from martine_converters.export import Options
    from martine_converters.formats import write_mvs

    p, _ = write_fake_sfm(tmp_path, n_poses=3)
    out = tmp_path / fmt
    write_mvs(read_sfm(p), out, fmt, Options(images=False, points="none", workers=1))
    assert out.is_dir()
    assert [f.name for f in out.rglob("*.sh")] == []


# --- COLMAP text model --------------------------------------------------------------------------------------------
def test_colmap_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    cams = {1: ColmapCamera(1, "PINHOLE", 640, 480, (500.123456789, 501.0, 320.5, 240.25))}
    imgs = {i: ColmapImage(i, rotmat_to_qvec(rand_rot(rng)), rng.normal(size=3) * 1e3, 1, f"{i:03d}.png",
                           rng.uniform(0, 600, size=(2, 2)), np.array([1, 2])) for i in (1, 2)}
    pts = {1: ColmapPoint(1, rng.normal(size=3), np.array([1, 2, 3]), 0.25, [(1, 0), (2, 0)]),
           2: ColmapPoint(2, rng.normal(size=3), np.array([4, 5, 6]), 0.5, [(1, 1), (2, 1)])}
    imgs[3] = ColmapImage(3, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, "with space.png")
    write_model(tmp_path, cams, imgs, pts)
    c2, i2, p2 = read_model(tmp_path)
    assert c2[1].params == cams[1].params
    for k in imgs:
        assert np.array_equal(i2[k].qvec, imgs[k].qvec) and np.array_equal(i2[k].tvec, imgs[k].tvec)
        assert np.array_equal(i2[k].xys, imgs[k].xys) and i2[k].name == imgs[k].name
    assert p2[2].track == pts[2].track and np.array_equal(p2[1].xyz, pts[1].xyz)


# --- images -------------------------------------------------------------------------------------------------------
def _paper_exr_to_ldr(img):
    """dataset_temp/sfm_converters.py exr_to_ldr(exposure=0) restated."""
    img = img.copy()
    if img.max() > 1.0:
        p95 = np.percentile(img[img > 0], 95)
        img = img * (0.9 / max(p95, 1e-6))
    if img.max() > 1.0:
        img = img / (1.0 + img)
    return (np.power(np.clip(img, 0, 1), 1 / 2.2) * 255).astype(np.uint8)


def test_tonemaps():
    rng = np.random.default_rng(5)
    for scale in (0.5, 3.0):
        img = (rng.random((20, 30, 3)) * scale).astype(np.float32)
        assert np.array_equal(tonemap(img, "paper"), _paper_exr_to_ldr(img))
    img = rng.random((8, 8, 3)).astype(np.float32)
    t16 = tonemap(img, "linear16", scale=1.0)
    assert t16.dtype == np.uint16 and np.abs(t16 / 65535.0 - img).max() <= 0.5 / 65535 + 1e-7
    m = np.zeros((8, 8), bool)
    m[2:6, 2:6] = True
    t = tonemap(img, "object99", mask=m)
    assert t.dtype == np.uint8 and t[m].max() == 255


def test_minimal_enclosing_ball():
    from martine_converters.normalize import minimal_enclosing_ball

    rng = np.random.default_rng(7)
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.0]]) * 3 + [5, -2, 1]
    c, r = minimal_enclosing_ball(np.vstack([tet, rng.uniform(-1, 1, (200, 3)) + [5, -2, 1]]))
    assert np.allclose(c, [5, -2, 1], atol=1e-9) and abs(r - 3 * np.sqrt(3)) < 1e-9
    for _ in range(20):
        P = rng.normal(size=(500, 3)) * [30, 10, 5] + rng.normal(size=3) * 100
        c, r = minimal_enclosing_ball(P)
        d = np.linalg.norm(P - c, axis=1)
        assert d.max() <= r * (1 + 1e-9)
        # optimality: no small move of the centre reduces the max distance
        for v in rng.normal(size=(50, 3)):
            assert np.linalg.norm(P - (c + 1e-3 * v / np.linalg.norm(v)), axis=1).max() >= r - 1e-9
