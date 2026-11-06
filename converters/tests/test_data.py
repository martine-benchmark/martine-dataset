# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Checks on the published data: tier relation, reader vs AliceVision (pyalicevision), GT-free normalisation vs
the GT."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import DATA, MVPS_TIERS, MVS_TIERS, OBJECT, sfm
from martine_converters.camera import W_CORR
from martine_converters.export import load_mask_checked
from martine_converters.normalize import bounding_sphere
from martine_converters.sfm import read_sfm
from method_code.reference import av_cameras, project

pytestmark = pytest.mark.data


@pytest.mark.parametrize("modality,tiers", [("mvs", MVS_TIERS), ("mvps", MVPS_TIERS)])
def test_tiers_are_centre_aligned(modality, tiers, gt_points):
    """u_dk = (u_ref + 0.5) / (k / k_ref) - 0.5 for every view: the tiers follow the AliceVision convention."""
    ref = {t: av_cameras(sfm(modality, t)) for t in tiers}
    k_of = {t: int(t[1:]) for t in tiers}
    base = tiers[0]
    worst = 0.0
    for name, c0 in ref[base].items():
        u0, _ = project(c0, gt_points)
        for t in tiers[1:]:
            s = k_of[t] // k_of[base]
            uk, _ = project(ref[t][name], gt_points)
            worst = max(worst, np.abs(uk - ((u0 + 0.5) / s - 0.5)).max())
    print(f"{modality} tier relation {tiers}: max {worst:.2e} px")
    assert worst < 1e-6


@pytest.mark.parametrize("tier", MVS_TIERS)
def test_reader_against_reference_and_pyalicevision(tier, gt_points):
    s = read_sfm(sfm("mvs", tier))
    ref = av_cameras(sfm("mvs", tier))
    for u in s.units:
        uv, _ = u.camera.project(gt_points)
        assert np.abs(uv - project(ref[u.name], gt_points)[0]).max() < 1e-9
    try:
        from pyalicevision import camera, geometry, numeric, sfmData, sfmDataIO  # noqa: F401 (typemaps)
    except ImportError:
        pytest.skip("pyalicevision not importable")
    data = sfmData.SfMData()
    assert sfmDataIO.load(data, str(sfm("mvs", tier)), sfmDataIO.ALL)
    by_stem = {u.name: u for u in s.units}
    worst = 0.0
    for vid, view in list(data.getViews().items())[::10]:
        name = view.getImage().getImagePath().rsplit("/", 1)[-1].rsplit(".", 1)[0]
        intr = data.getIntrinsicPtr(view.getIntrinsicId())
        pose = data.getPose(view)  # keep the CameraPose alive: getTransform() refers to its memory
        T = pose.getTransform()
        X_raw = gt_points[:200] @ W_CORR.T  # published sfm.json world = W * GT frame
        Rm, Cm = np.array(T.rotation(), dtype=float), np.array(T.center(), dtype=float).reshape(3)
        xc = (X_raw - Cm) @ Rm.T  # AliceVision Pose3, then AliceVision intrinsics (cam2ima)
        uv_av = np.array([np.asarray(intr.cam2ima(np.array(p, dtype=np.float64))).reshape(-1)
                          for p in xc[:, :2] / xc[:, 2:]])
        worst = max(worst, np.abs(uv_av - by_stem[name].camera.project(gt_points[:200])[0]).max())
    print(f"reader vs pyalicevision {tier}: {worst:.2e} px")
    assert worst < 1e-6


def test_bounding_sphere_contains_gt_and_is_tier_independent():
    g = np.load(DATA / OBJECT / "gt" / "gt_pcd.npy", mmap_mode="r")
    pts = np.asarray(g[:: max(1, len(g) // 200000)], float)
    spheres = {}
    for tier in MVS_TIERS:
        s = read_sfm(sfm("mvs", tier))
        units = list(s.units)
        spheres[tier] = bounding_sphere([u.camera for u in units], lambda i: load_mask_checked(units[i]))
        d = np.linalg.norm(pts - spheres[tier].center, axis=1)
        print(f"{tier}: centre {np.round(spheres[tier].center, 3)}, radius {spheres[tier].radius:.3f} mm, "
              f"GT max dist {d.max():.3f} mm ({100 * d.max() / spheres[tier].radius:.1f} % of r)")
        assert d.max() <= spheres[tier].radius  # GT-free sphere encloses the GT
        assert spheres[tier].radius <= 1.6 * d.max()  # and stays reasonably tight
    # the radius is what matters and is stable; the centre of a minimal enclosing ball of a roundish hull is
    # ill-conditioned (moving it by e changes the radius by O(e^2 / r)), so it may move by a few voxels
    first = MVS_TIERS[0]
    for tier in MVS_TIERS[1:]:
        assert np.linalg.norm(spheres[tier].center - spheres[first].center) < 0.05 * spheres[first].radius
        assert abs(spheres[tier].radius / spheres[first].radius - 1) < 0.01


def test_bounding_sphere_vs_paper_silhouettes_v2():
    """Report the paper's normalisation (RNb-NeuS2 compute_scaling_from_silhouettes_v2, 20 px margin) next to ours."""
    from method_code import real_code

    root = real_code.REPOS / "RNb-NeuS2" / "rnb_neus2" / "scaling.py"
    if not root.is_file():
        pytest.skip("RNb-NeuS2 checkout not available")
    pytest.importorskip("scipy")
    scaling = real_code._load(root, "rnb_scaling")
    s = read_sfm(sfm("mvs", "d4"))
    units = list(s.units)
    masks = [load_mask_checked(u).astype(np.float64) for u in units]
    cams = [{"fx": u.camera.fx, "fy": u.camera.fy, "cx": u.camera.cx, "cy": u.camera.cy,
             "R_cam2world": u.camera.R.T, "center": u.camera.C} for u in units]
    c_p, sf = scaling.compute_scaling_from_silhouettes_v2(cams, masks, sphere_scale=1.0, margin_px=20)
    ours = bounding_sphere([u.camera for u in units], lambda i: masks[i] > 0.5)
    r_p = 1.0 / sf
    print(f"paper v2 (d4, 20 px): centre {np.round(c_p, 3)} r {r_p:.3f} | ours: centre {np.round(ours.center, 3)} "
          f"r {ours.radius:.3f} | centre distance {np.linalg.norm(c_p - ours.center):.3f} mm")
    assert np.linalg.norm(c_p - ours.center) < 0.2 * r_p
    assert 0.7 < ours.radius / r_p < 1.4
