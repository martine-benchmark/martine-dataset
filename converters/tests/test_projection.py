# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""For every format and every published tier: convert the cameras, load the produced files the way the method does,
project GT points and compare with the AliceVision projection of the published sfm.json. Required: <= 1e-3 px.

This is what makes the converters resolution-agnostic by test: the same code is checked against d1, d2 and d4.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from conftest import MVPS_TIERS, MVS_TIERS, sfm
from martine_converters.export import Options
from martine_converters.formats import write_mvs
from martine_converters.formats.mvps import write_mvps
from martine_converters.sfm import read_sfm
from method_code import methods, real_code
from method_code.reference import av_cameras, project

TOL = 1e-3  # px
RESULTS = {}


def compare(ref_sfm, loaded, X, label, tol=TOL):
    ref = av_cameras(ref_sfm)
    assert set(loaded) == set(ref), f"{label}: views {sorted(set(loaded) ^ set(ref))[:5]}"
    worst, n = 0.0, 0
    for name, cam in ref.items():
        uv, z = project(cam, X)
        ok = (z > 0) & (uv[:, 0] > -0.5) & (uv[:, 1] > -0.5) & (uv[:, 0] < cam["w"] - 0.5) & (uv[:, 1] < cam["h"] - 0.5)
        uvm, zm = loaded[name](X[ok])
        assert np.all(zm > 0), f"{label} {name}: points behind the camera"
        err = np.abs(uvm - uv[ok]).max() if ok.any() else 0.0
        worst, n = max(worst, float(err)), n + int(ok.sum())
    RESULTS[label] = worst
    print(f"{label:<40} max |du| = {worst:.2e} px over {n} projections")
    assert n > 1000
    assert worst <= tol, f"{label}: {worst:.3e} px > {tol}"
    return worst


def convert(tmp_path, tier, fmt, modality="mvs", **kw):
    out = tmp_path / f"{fmt}_{tier}"
    scene = read_sfm(sfm(modality, tier))
    opts = Options(images=False, points="none", workers=4, **kw)
    if modality == "mvs":
        write_mvs(scene, out, fmt, opts)
    else:
        write_mvps(scene, out, fmt, None, opts)
    return out, scene


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_colmap_sparse(tmp_path, tier, gt_points):
    out, _ = convert(tmp_path, tier, "colmap")
    compare(sfm("mvs", tier), methods.colmap(out / "sparse" / "0", dense=False), gt_points, f"colmap {tier}")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_colmap_dense(tmp_path, tier, gt_points):
    out, _ = convert(tmp_path, tier, "colmap-mvs")
    compare(sfm("mvs", tier), methods.colmap(out / "sparse" / "0", dense=True), gt_points, f"colmap-mvs {tier}")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_colmap_with_pycolmap(tmp_path, tier, gt_points):
    """COLMAP's own projection (pycolmap Image.project_point = Camera::ImgFromCam(cam_from_world X))."""
    pycolmap = pytest.importorskip("pycolmap")
    for fmt, shift in (("colmap", 0.5), ("colmap-mvs", 0.0)):
        out, _ = convert(tmp_path, tier, fmt)
        rec = pycolmap.Reconstruction(str(out / "sparse" / "0"))
        loaded = {}
        for img in rec.images.values():
            def proj(X, img=img, shift=shift):
                uv = np.array([img.project_point(x) for x in X], dtype=float)
                z = (X @ img.cam_from_world().rotation.matrix().T + img.cam_from_world().translation)[:, 2] \
                    if callable(getattr(img, "cam_from_world", None)) else \
                    (X @ img.cam_from_world.rotation.matrix().T + img.cam_from_world.translation)[:, 2]
                return uv - shift, z
            loaded[img.name.rsplit(".", 1)[0]] = proj
        compare(sfm("mvs", tier), loaded, gt_points[:300], f"{fmt} {tier} (pycolmap)")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_pgsr(tmp_path, tier, gt_points):
    out, _ = convert(tmp_path, tier, "pgsr")
    real = real_code.pgsr()
    label = f"pgsr {tier}" + (" (PGSR code: colmap_loader" + (", graphics_utils)" if real and real["has_torch"]
                                                                    else ")") if real else "")
    # (PGSR's getWorld2View2 returns float32: still within TOL)
    compare(sfm("mvs", tier), methods.pgsr(out, real=real), gt_points, label)


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_meshroom(tmp_path, tier, gt_points):
    out, _ = convert(tmp_path, tier, "meshroom")
    compare(sfm("mvs", tier), methods.meshroom(out / "sfm.json"), gt_points, f"meshroom {tier}")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_meshroom_with_pyalicevision(tmp_path, tier, gt_points):
    """AliceVision's own reader and camera model (pyalicevision, Meshroom 2025.1 bundle) on the written sfm.json."""
    pytest.importorskip("pyalicevision")
    from pyalicevision import camera, geometry, numeric, sfmData, sfmDataIO  # noqa: F401 (typemaps)

    out, _ = convert(tmp_path, tier, "meshroom")
    data = sfmData.SfMData()
    assert sfmDataIO.load(data, str(out / "sfm.json"), sfmDataIO.ALL)
    loaded = {}
    for vid, view in data.getViews().items():
        intr = data.getIntrinsicPtr(view.getIntrinsicId())
        pose = data.getPose(view)  # keep the CameraPose alive: getTransform() refers to its memory
        T = pose.getTransform()
        Rm, Cm = np.array(T.rotation(), dtype=float), np.array(T.center(), dtype=float).reshape(3)

        def proj(X, intr=intr, Rm=Rm, Cm=Cm):
            # pose parsed by AliceVision (Pose3: x_cam = R (X - C)), intrinsics applied by AliceVision cam2ima
            xc = (X - Cm) @ Rm.T
            uv = np.array([np.asarray(intr.cam2ima(np.array(p, dtype=np.float64))).reshape(-1)
                           for p in xc[:, :2] / xc[:, 2:]])
            return uv, xc[:, 2]
        loaded[os.path.splitext(os.path.basename(view.getImage().getImagePath()))[0]] = proj
    compare(sfm("mvs", tier), loaded, gt_points[:300], f"meshroom {tier} (pyalicevision)")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
@pytest.mark.parametrize("delta", [0.5, 0.0])
def test_realityscan(tmp_path, tier, delta, gt_points):
    out, scene = convert(tmp_path, tier, "realityscan", pixel_centre=delta)
    sizes = {u.name: (u.camera.width, u.camera.height) for u in scene.units}
    compare(sfm("mvs", tier), methods.realityscan(out / "images", delta, sizes), gt_points,
            f"realityscan {tier} (assumed centre +{delta})")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_neus2(tmp_path, tier, gt_points):
    out, _ = convert(tmp_path, tier, "neus2")
    compare(sfm("mvs", tier), methods.ngp(json.loads((out / "transforms.json").read_text())), gt_points,
            f"neus2 {tier}")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_neuralangelo(tmp_path, tier, gt_points):
    out, _ = convert(tmp_path, tier, "neuralangelo")
    compare(sfm("mvs", tier), methods.neuralangelo(out), gt_points, f"neuralangelo {tier}")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVS_TIERS)
def test_neuralangelo_real_rays(tmp_path, tier, gt_points):
    """Neuralangelo's own get_center_and_ray: the ray of the pixel at the AliceVision projection of X passes
    through X (needs torch + the neuralangelo checkout)."""
    camera = real_code.neuralangelo_camera()
    if camera is None:
        pytest.skip("neuralangelo code (torch) not importable")
    import torch

    out, _ = convert(tmp_path, tier, "neuralangelo")
    m = json.loads((out / "transforms.json").read_text())
    ref = av_cameras(sfm("mvs", tier))
    c, r = np.array(m["sphere_center"]), m["sphere_radius"]
    f32 = torch.float32  # Neuralangelo's Pose() casts to float32 (projects/nerf/utils/camera.py:40-41)
    K = torch.tensor([[m["fl_x"], m["sk_x"], m["cx"]], [m["sk_y"], m["fl_y"], m["cy"]], [0, 0, 1]], dtype=f32)
    worst = 0.0
    for fr in m["frames"][::7]:
        name = fr["file_path"].split("/")[-1].rsplit(".", 1)[0]
        c2w = torch.tensor(fr["transform_matrix"], dtype=f32) * torch.tensor([1, -1, -1, 1.0])
        c2w[:3, -1] = (c2w[:3, -1] - torch.tensor(c, dtype=f32)) / r
        w2c = camera.Pose().invert(c2w[:3])
        uv, z = project(ref[name], gt_points)
        ok = z > 0
        # ray of the pixel centre (i, j) = index coordinates uv: Neuralangelo uses i + 0.5 for that centre
        xy = torch.tensor(uv[ok] + 0.5, dtype=f32)
        grid = camera.img2cam(camera.to_hom(xy), K)
        d = camera.cam2world(grid, w2c) - camera.cam2world(torch.zeros_like(grid), w2c)
        o = camera.cam2world(torch.zeros(1, 3, dtype=f32), w2c)
        Xn = torch.tensor((gt_points[ok] - c) / r, dtype=f32)
        v = Xn - o
        cross = torch.linalg.norm(torch.cross(v, d, dim=-1), dim=-1) / torch.linalg.norm(d, dim=-1)
        dist = cross / torch.linalg.norm(v, dim=-1)  # angle (rad)
        worst = max(worst, float((dist * m["fl_x"]).max()))  # in pixels
    print(f"neuralangelo {tier} (get_center_and_ray, float32) max = {worst:.2e} px")
    # Neuralangelo computes rays in float32 (Pose() casts, camera.py:40-41): its own rounding floor is a few 1e-3 px
    # at d1 (the float64 re-implementation of the same loader, test_neuralangelo, is exact to 1e-11 px)
    assert worst <= 1e-2


@pytest.mark.data
@pytest.mark.parametrize("tier", MVPS_TIERS)
def test_rnbneus2(tmp_path, tier, gt_points):
    out, scene = convert(tmp_path, tier, "rnbneus2", modality="mvps")
    size = (scene.units[0].camera.width, scene.units[0].camera.height)
    compare(sfm("mvps", tier), methods.rnbneus2(out, size), gt_points, f"rnbneus2 {tier} (float64 reading)")
    loader = real_code.rnb_loader()
    if loader is not None:  # the original RnbDataLoader (casts the matrices to float32)
        (out / "normal").mkdir(exist_ok=True)
        import cv2
        for i in range(len(scene.units)):
            cv2.imwrite(str(out / "normal" / f"{i:03d}.png"), np.zeros((2, 2, 3), np.uint8))
        compare(sfm("mvps", tier), methods.rnbneus2(out, size, loader=loader), gt_points,
                f"rnbneus2 {tier} (RnbDataLoader, float32)")


@pytest.mark.data
@pytest.mark.parametrize("tier", MVPS_TIERS)
def test_supernormal(tmp_path, tier, gt_points):
    out, _ = convert(tmp_path, tier, "supernormal", modality="mvps")
    compare(sfm("mvps", tier), methods.supernormal(out, dtype=np.float64), gt_points,
            f"supernormal {tier} (float64 reading)")
    compare(sfm("mvps", tier), methods.supernormal(out), gt_points, f"supernormal {tier} (float32 as the loader)")
    real = real_code.supernormal_load_K_Rt_from_P()
    if real is not None:
        compare(sfm("mvps", tier), methods.supernormal(out, load_K_Rt_from_P=real), gt_points,
                f"supernormal {tier} (load_K_Rt_from_P)")


@pytest.mark.data
@pytest.mark.parametrize("fmt", ["neus2", "colmap", "colmap-mvs", "pgsr", "meshroom", "neuralangelo",
                                 "realityscan"])
def test_mvs_formats_on_mvps_sfm(tmp_path, fmt, gt_points):
    """MVS formats also accept an MVPS sfm.json (one LED per pose)."""
    out = tmp_path / fmt
    scene = read_sfm(sfm("mvps", "d4"))
    write_mvs(scene, out, fmt, Options(images=False, points="none", lighting="lighting_03"))
    if fmt == "neus2":
        loaded = methods.ngp(json.loads((out / "transforms.json").read_text()))
    elif fmt in ("colmap", "colmap-mvs"):
        loaded = methods.colmap(out / "sparse" / "0", dense=fmt == "colmap-mvs")
    elif fmt == "pgsr":
        loaded = methods.pgsr(out)
    elif fmt == "meshroom":
        loaded = methods.meshroom(out / "sfm.json")
    elif fmt == "neuralangelo":
        loaded = methods.neuralangelo(out)
    else:
        sizes = {u.name: (u.camera.width, u.camera.height) for u in scene.units}
        loaded = methods.realityscan(out / "images", 0.5, sizes)
    compare(sfm("mvps", "d4"), loaded, gt_points, f"{fmt} on mvps d4")
