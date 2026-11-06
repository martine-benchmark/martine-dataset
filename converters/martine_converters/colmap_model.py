# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""COLMAP text model (cameras.txt, images.txt, points3D.txt): writer and reader. Pure numpy.

Format: https://colmap.github.io/format.html (quaternion = world-to-camera rotation, Hamilton, scalar first;
``t = -R C``). Floats are written with 17 significant digits (exact round trip).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CAMERA_MODELS = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4}


def _f(x: float) -> str:
    return format(float(x), ".17g")


@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: Tuple[float, ...]


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


@dataclass
class ColmapPoint:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: List[Tuple[int, int]]  # (image_id, point2D_idx)


def write_model(path, cameras: Dict[int, ColmapCamera], images: Dict[int, ColmapImage],
                points: Dict[int, ColmapPoint]) -> None:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    lines = ["# Camera list with one line of data per camera:",
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
             f"# Number of cameras: {len(cameras)}"]
    for c in sorted(cameras.values(), key=lambda c: c.id):
        if c.model not in CAMERA_MODELS or len(c.params) != CAMERA_MODELS[c.model]:
            raise ValueError(f"camera {c.id}: unsupported model/params {c.model} {c.params}")
        lines.append(" ".join([str(c.id), c.model, str(c.width), str(c.height)] + [_f(p) for p in c.params]))
    (d / "cameras.txt").write_text("\n".join(lines) + "\n")

    n_obs = sum(len(im.point3D_ids) for im in images.values())
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)",
             f"# Number of images: {len(images)}, mean observations per image: "
             f"{n_obs / max(len(images), 1):.3f}"]
    for im in sorted(images.values(), key=lambda i: i.id):
        lines.append(" ".join([str(im.id)] + [_f(v) for v in im.qvec] + [_f(v) for v in im.tvec]
                              + [str(im.camera_id), im.name]))
        lines.append(" ".join(f"{_f(x)} {_f(y)} {int(p)}" for (x, y), p in zip(im.xys, im.point3D_ids)))
    (d / "images.txt").write_text("\n".join(lines) + "\n")

    lines = ["# 3D point list with one line of data per point:",
             "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
             f"# Number of points: {len(points)}"]
    for p in sorted(points.values(), key=lambda p: p.id):
        lines.append(" ".join([str(p.id)] + [_f(v) for v in p.xyz] + [str(int(v)) for v in p.rgb] + [_f(p.error)]
                              + [f"{i} {j}" for i, j in p.track]))
    (d / "points3D.txt").write_text("\n".join(lines) + "\n")


def read_model(path) -> Tuple[Dict[int, ColmapCamera], Dict[int, ColmapImage], Dict[int, ColmapPoint]]:
    d = Path(path)
    cameras: Dict[int, ColmapCamera] = {}
    for line in (d / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        x = line.split()
        cameras[int(x[0])] = ColmapCamera(int(x[0]), x[1], int(x[2]), int(x[3]), tuple(float(v) for v in x[4:]))
    images: Dict[int, ColmapImage] = {}
    rows = [ln for ln in (d / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    i = 0
    while i < len(rows):
        if not rows[i].strip():
            i += 1
            continue
        x = rows[i].split()
        pts = rows[i + 1].split() if i + 1 < len(rows) else []
        arr = np.array(pts, dtype=np.float64).reshape(-1, 3) if pts else np.zeros((0, 3))
        images[int(x[0])] = ColmapImage(int(x[0]), np.array(x[1:5], float), np.array(x[5:8], float), int(x[8]),
                                        " ".join(x[9:]), arr[:, :2], arr[:, 2].astype(np.int64))
        i += 2
    points: Dict[int, ColmapPoint] = {}
    p3 = d / "points3D.txt"
    if p3.exists():
        for line in p3.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            x = line.split()
            tr = [int(v) for v in x[8:]]
            points[int(x[0])] = ColmapPoint(int(x[0]), np.array(x[1:4], float), np.array(x[4:7], int),
                                            float(x[7]), list(zip(tr[0::2], tr[1::2])))
    return cameras, images, points
