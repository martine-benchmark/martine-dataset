# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Reference AliceVision projection of a published sfm.json, written independently of ``martine_converters.sfm``
(same formulas as the evaluation loader of eval-3d, core/camera.py: fx = focalLength * width / sensorWidth,
cx = width / 2 + principalPoint, R_c2w = W @ rotation (row-major), C = W @ center, W = diag(1, -1, -1))."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

W = np.diag([1.0, -1.0, -1.0])


def av_cameras(sfm_path) -> Dict[str, dict]:
    """name (image stem for MVS, pose folder for MVPS) -> {K, R_w2c, C, w, h} in the GT frame."""
    d = json.loads(Path(sfm_path).read_text())
    intr = {str(i["intrinsicId"]): i for i in d["intrinsics"]}
    poses = {str(p["poseId"]): p["pose"]["transform"] for p in d["poses"]}
    per_pose: Dict[str, list] = {}
    for v in d["views"]:
        if str(v["poseId"]) in poses and str(v["intrinsicId"]) in intr:
            per_pose.setdefault(str(v["poseId"]), []).append(v)
    mvps = any(len(vs) > 1 for vs in per_pose.values())
    out = {}
    for pid, vs in per_pose.items():
        v = vs[0]
        i = intr[str(v["intrinsicId"])]
        w, h = int(i["width"]), int(i["height"])
        f = float(i["focalLength"]) * w / float(i["sensorWidth"])
        assert float(i.get("pixelRatio", 1)) == 1.0
        cx = w / 2.0 + float(i["principalPoint"][0])
        cy = h / 2.0 + float(i["principalPoint"][1])
        Rc2w = W @ np.array([float(x) for x in poses[pid]["rotation"]]).reshape(3, 3)
        C = W @ np.array([float(x) for x in poses[pid]["center"]])
        name = Path(v["path"]).parent.name if mvps else Path(v["path"]).stem
        out[name] = {"K": np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), "R": Rc2w.T, "C": C, "w": w, "h": h}
    return out


def project(cam: dict, X: np.ndarray):
    xc = (X - cam["C"]) @ cam["R"].T
    uvw = xc @ cam["K"].T
    return uvw[:, :2] / uvw[:, 2:], xc[:, 2]
