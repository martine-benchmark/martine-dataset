# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Synthetic MVPS scene in the published layout: a sphere seen by a ring of cameras, with masks, multi-light EXR
images (Lambertian), and the camera-frame normal maps a PS method would output (OpenGL axes, 16-bit PNG).

Everything is rendered in the AliceVision pixel convention (the ray of pixel index (i, j) goes through K^-1 (i, j, 1)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from martine_converters.camera import W_CORR, Camera
from martine_converters.imageio import write_image, write_mask
from martine_converters.normals import encode16

W, H = 320, 240
F = 900.0
PP = (3.25, -2.5)  # principal point offset (AliceVision)
CENTER = np.array([5.0, -3.0, 2.0])
RADIUS = 60.0
LIGHTS = np.array([[0.3, -0.2, -1.0], [-0.4, 0.1, -1.0], [0.1, 0.5, -1.0]])  # camera frame (OpenCV), towards light


def look_at(C, target, up=np.array([0.0, 0.0, 1.0])):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # rows = camera axes in world: R (world -> camera)


def cameras(n=6, dist=700.0):
    out = []
    for k in range(n):
        a = 2 * np.pi * k / n
        C = np.array([dist * np.cos(a), dist * np.sin(a), 150.0 * np.sin(3 * a)])
        out.append(Camera(W, H, F, F, W / 2 + PP[0], H / 2 + PP[1], look_at(C, np.zeros(3)), C))
    return out


def render(cam: Camera):
    """Mask and world normals of the sphere at every pixel centre (AliceVision convention)."""
    j, i = np.mgrid[0:H, 0:W].astype(np.float64)
    d_cam = np.stack([(i - cam.cx) / cam.fx, (j - cam.cy) / cam.fy, np.ones_like(i)], -1)
    d = d_cam @ cam.R  # world directions R^T d
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    oc = cam.C - CENTER
    b = d @ oc
    disc = b * b - (oc @ oc - RADIUS ** 2)
    mask = disc > 0
    t = -b - np.sqrt(np.maximum(disc, 0))
    P = cam.C + t[..., None] * d
    n = (P - CENTER) / RADIUS
    n[~mask] = 0
    return mask, n


def build(root: Path, n=6, lights=3, frame_is_gt=True):
    """Write sfm.json (raw AliceVision world = W GT), images/pose_KK/lighting_LL.exr, masks/pose_KK.png and
    normals/pose_KK.png (PS output). Returns the GT-frame cameras and the world normal maps."""
    root.mkdir(parents=True, exist_ok=True)
    cams = cameras(n)
    views, poses, nmaps = [], [], {}
    for k, cam in enumerate(cams):
        raw = cam.transformed(W_CORR) if frame_is_gt else cam  # W is its own inverse
        pid = str(70000 + k)
        poses.append({"poseId": pid, "pose": {"transform": {
            "rotation": [repr(float(x)) for x in raw.R.T.reshape(-1)], "center": [repr(float(x)) for x in raw.C]}}})
        mask, n_w = render(cam)
        name = f"pose_{k:02d}"
        write_mask(root / "masks" / f"{name}.png", mask)
        n_cv = n_w @ cam.R.T
        n_gl = n_cv * np.array([1.0, -1.0, -1.0])
        enc = encode16(n_gl)
        enc[~mask] = 32768
        write_image(root / "normals" / f"{name}.png", enc)
        nmaps[name] = (mask, n_w)
        for li in range(lights):
            L = LIGHTS[li] / np.linalg.norm(LIGHTS[li])
            shade = np.clip(n_cv @ L, 0, None) * mask * (0.2 + 0.3 * li)
            img = np.repeat(shade[..., None], 3, axis=2).astype(np.float32) * np.array([1.0, 0.8, 0.6], np.float32)
            write_image(root / "images" / name / f"lighting_{li:02d}.exr", img)
            views.append({"viewId": str(90000 + 100 * k + li), "poseId": pid, "intrinsicId": "7",
                          "path": f"images/{name}/lighting_{li:02d}.exr", "width": str(W), "height": str(H)})
    intr = {"intrinsicId": "7", "width": str(W), "height": str(H), "sensorWidth": "36", "type": "pinhole",
            "focalLength": repr(float(F * 36 / W)), "pixelRatio": "1", "principalPoint": [repr(float(PP[0])), repr(float(PP[1]))],
            "distortionType": "none", "distortionParams": ""}
    (root / "sfm.json").write_text(json.dumps({"version": ["1", "2", "12"], "intrinsics": [intr], "poses": poses,
                                               "views": views}))
    return cams, nmaps
