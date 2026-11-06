# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Normal maps: decoding/encoding, camera conventions, camera <-> world frames, and a GT-free convention check.

Camera-frame conventions:

* ``opencv``: x right, y down, z forward (into the scene). Visible normals have z < 0.
* ``opengl``: x right, y up, z towards the camera (DiLiGenT; output of Uni-MS-PS and SDM-UniPS). Visible normals
  have z > 0. ``n_opencv = diag(1, -1, -1) n_opengl``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .camera import CV_TO_GL, Camera
from .imageio import read_image

CONVENTIONS = ("opengl", "opencv")


def decode(img: np.ndarray) -> np.ndarray:
    """Encoded normal map -> float32 (H, W, 3) in [-1, 1]. Integer images use ``n = 2 v / max - 1``; float images
    (EXR) are taken as is."""
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"normal map must be (H, W, >=3), got {a.shape}")
    a = a[..., :3]
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0 * 2.0 - 1.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0 * 2.0 - 1.0
    return a.astype(np.float32)


def encode16(n: np.ndarray) -> np.ndarray:
    """[-1, 1] -> uint16 ``round((n + 1) / 2 * 65535)`` (Uni-MS-PS ``normal_to_rgb_16bits`` up to rounding)."""
    return np.round((np.clip(n, -1.0, 1.0) + 1.0) * 0.5 * 65535.0).astype(np.uint16)


def unit(n: np.ndarray, valid: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-length normals and validity (norm > 0.1 and ``valid``); invalid pixels are set to 0."""
    n = np.asarray(n, dtype=np.float32)
    norm = np.linalg.norm(n, axis=-1)
    ok = norm > 0.1
    if valid is not None:
        ok &= np.asarray(valid, bool)
    out = np.zeros_like(n)
    out[ok] = n[ok] / norm[ok, None]
    return out, ok


def read_normal_map(path) -> np.ndarray:
    """PNG (8/16-bit, RGB = xyz) or EXR (float, R G B = xyz) -> float32 (H, W, 3)."""
    return decode(read_image(Path(path)))


def to_opencv(n: np.ndarray, convention: str) -> np.ndarray:
    if convention == "opencv":
        return np.asarray(n, dtype=np.float32)
    if convention == "opengl":
        return np.asarray(n, dtype=np.float32) * np.diag(CV_TO_GL).astype(np.float32)
    raise ValueError(f"unknown normal convention {convention!r}")


def from_opencv(n: np.ndarray, convention: str) -> np.ndarray:
    return to_opencv(n, convention)  # diag(1, -1, -1) is its own inverse


def camera_to_world(n_cv: np.ndarray, cam: Camera) -> np.ndarray:
    """Camera-frame (OpenCV axes) normals -> world normals ``R^T n``."""
    return (np.asarray(n_cv, dtype=np.float64) @ cam.R).astype(np.float32)


def world_to_camera(n_w: np.ndarray, cam: Camera) -> np.ndarray:
    return (np.asarray(n_w, dtype=np.float64) @ cam.R.T).astype(np.float32)


# --- GT-free convention check -----------------------------------------------------------------------------------
def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    k = 2 * r + 1
    p = np.pad(a.astype(np.float64), r, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def check_convention(n_raw: np.ndarray, mask: np.ndarray, ring: int = 4) -> Dict[str, object]:
    """Score the axis convention of a camera-frame normal map with the silhouette only.

    At the occluding contour the normal is perpendicular to the viewing ray and its image-plane part points out of
    the mask. For every sign hypothesis ``(sx, sy)`` mapping the stored (x, y) to OpenCV image axes, the score is
    the mean cosine between ``(sx n_x, sy n_y)`` and the outward mask gradient on a ``ring``-pixel band inside the
    silhouette. Returns the scores, the best hypothesis, the implied convention and the fraction of object pixels
    whose stored z says "facing the camera" in that convention.
    """
    m = np.asarray(mask, bool)
    n = np.asarray(n_raw, dtype=np.float64)
    blur = _box_blur(m.astype(np.float64), ring)
    gy, gx = np.gradient(blur)
    band = m & (blur < 0.999) & (np.hypot(gx, gy) > 1e-6)
    if band.sum() < 20:
        raise ValueError("silhouette band too small to check the normal convention")
    d = -np.stack([gx[band], gy[band]], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nxy = n[band][:, :2]
    nn = np.linalg.norm(nxy, axis=1)
    ok = nn > 1e-6
    d, nxy = d[ok], nxy[ok] / nn[ok, None]
    scores = {}
    for sx in (1, -1):
        for sy in (1, -1):
            scores[(sx, sy)] = float(np.mean(sx * nxy[:, 0] * d[:, 0] + sy * nxy[:, 1] * d[:, 1]))
    best = max(scores, key=scores.get)
    convention = {(1, 1): "opencv", (1, -1): "opengl"}.get(best, f"x{'+' if best[0] > 0 else '-'}"
                                                                   f"y{'+' if best[1] > 0 else '-'}")
    z = n[m][:, 2]
    zv = z[np.abs(z) > 1e-6]
    facing = float(np.mean(zv < 0)) if convention == "opencv" else float(np.mean(zv > 0))
    return {"scores": {f"{'+' if k[0] > 0 else '-'}x{'+' if k[1] > 0 else '-'}y": v for k, v in scores.items()},
            "best": best, "convention": convention, "facing_fraction": facing, "n_band": int(ok.sum())}
