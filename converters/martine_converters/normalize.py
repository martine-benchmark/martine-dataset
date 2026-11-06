# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""GT-free scene normalisation for the neural methods: bounding sphere of the visual hull of the masks.

1. Masks are max-pooled to at most ``pool_target`` pixels on their long side (integer factor, so any published
   tier gives the same pooled grid); the camera is mapped exactly to the pooled grid.
2. A first voxel grid on a cube around the least-squares intersection of the mask-centroid rays is carved by every
   mask (a voxel is removed when it projects inside an image but outside the mask); a second, finer grid on the
   bounding box of the survivors is carved again.
3. The sphere is the minimal enclosing ball of the surviving voxel centres (Badoiu-Clarkson iterations), inflated
   by half a voxel diagonal and ``margin`` (relative).

Only cameras and masks are used. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .camera import Camera


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float
    info: dict = field(default_factory=dict)

    def scale_mat(self) -> np.ndarray:
        """4x4 normalised -> world (IDR / NeuS convention: ``X = r x + c``)."""
        S = np.eye(4)
        S[:3, :3] *= self.radius
        S[:3, 3] = self.center
        return S

    def to_dict(self) -> dict:
        return {"center": [float(x) for x in self.center], "radius": float(self.radius),
                "normalized_to_world": self.scale_mat().tolist(), **self.info}


def pool_mask(mask: np.ndarray, target: int = 1200) -> Tuple[np.ndarray, int]:
    """Max-pool a boolean mask by the smallest integer factor ``k`` giving a long side <= ``target`` (edges padded
    with False)."""
    m = np.asarray(mask, bool)
    k = max(1, int(np.ceil(max(m.shape) / float(target))))
    if k == 1:
        return m, 1
    h, w = m.shape
    H, W = -(-h // k), -(-w // k)
    pad = np.zeros((H * k, W * k), bool)
    pad[:h, :w] = m
    return pad.reshape(H, k, W, k).any(axis=(1, 3)), k


def _pooled_coords(uv: np.ndarray, k: int) -> np.ndarray:
    # AliceVision convention: pixel index i has its centre at i; pooled block b covers indices [b k, b k + k - 1],
    # so its centre (b k + (k - 1) / 2) has pooled coordinate b.
    return (uv - (k - 1) / 2.0) / k


def _carve(points: np.ndarray, views: Sequence[Tuple[Camera, np.ndarray, int]],
           min_seen: float = 0.5) -> np.ndarray:
    """Keep a point if it falls in the mask of every view whose image it projects into, and if it projects into at
    least ``min_seen`` of the images (points outside an image are not carved by that view)."""
    keep = np.ones(len(points), bool)
    seen = np.zeros(len(points), np.int32)
    for cam, pm, k in views:
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            break
        uv, z = cam.project(points[idx])
        pc = np.rint(_pooled_coords(uv, k)).astype(np.int64)
        H, W = pm.shape
        inside = (z > 0) & (pc[:, 0] >= 0) & (pc[:, 1] >= 0) & (pc[:, 0] < W) & (pc[:, 1] < H)
        bad = np.zeros(idx.size, bool)
        sel = np.nonzero(inside)[0]
        bad[sel] = ~pm[pc[sel, 1], pc[sel, 0]]
        keep[idx[bad]] = False
        seen[idx[inside]] += 1
    return keep & (seen >= min_seen * len(views))


def _centroid_rays_center(views: Sequence[Tuple[Camera, np.ndarray, int]]) -> np.ndarray:
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for cam, pm, k in views:
        ys, xs = np.nonzero(pm)
        if xs.size == 0:
            continue
        u = xs.mean() * k + (k - 1) / 2.0
        v = ys.mean() * k + (k - 1) / 2.0
        d = cam.R.T @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
        d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ cam.C
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _initial_half_size(views, center) -> float:
    half = 0.0
    for cam, pm, k in views:
        ys, xs = np.nonzero(pm)
        if xs.size == 0:
            continue
        z = (cam.R @ (center - cam.C))[2]
        u = np.array([xs.min(), xs.max()]) * k + (k - 1) / 2.0
        v = np.array([ys.min(), ys.max()]) * k + (k - 1) / 2.0
        uc, vc = cam.project(center[None])[0][0]
        half = max(half, np.abs(u - uc).max() * z / cam.fx, np.abs(v - vc).max() * z / cam.fy)
    return 1.5 * half


def _grid(lo: np.ndarray, hi: np.ndarray, n: int) -> Tuple[np.ndarray, float]:
    step = float(np.max(hi - lo)) / n
    axes = [lo[i] + step * (np.arange(max(1, int(np.ceil((hi[i] - lo[i]) / step)))) + 0.5) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return g, step


def _circumball(S: np.ndarray):
    """Smallest ball with all points of ``S`` (1 to 4 points) on its boundary, or None if degenerate."""
    p0 = S[0]
    if len(S) == 1:
        return p0.copy(), 0.0
    A = S[1:] - p0  # centre = p0 + A^T x with 2 A A^T x = |A|^2 (centre in the affine hull of S)
    G = A @ A.T
    if abs(np.linalg.det(G)) < 1e-12 * max(1.0, np.abs(G).max()) ** len(A):
        return None
    x = np.linalg.solve(2.0 * G, (A * A).sum(axis=1))
    c = p0 + A.T @ x
    return c, float(np.linalg.norm(S - c, axis=1).max())


def _small_meb(S: np.ndarray):
    """Exact minimal enclosing ball of at most 5 points (enumeration of supports) -> (centre, radius, support)."""
    from itertools import combinations

    best = None
    for k in range(1, min(4, len(S)) + 1):
        for idx in combinations(range(len(S)), k):
            b = _circumball(S[list(idx)])
            if b is None:
                continue
            c, r = b
            if np.all(np.linalg.norm(S - c, axis=1) <= r * (1 + 1e-10) + 1e-12) and (best is None or r < best[1]):
                best = (c, r, S[list(idx)])
    return best


def minimal_enclosing_ball(points: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
    """Exact minimal enclosing ball (pivoting: add the farthest point to the support, re-solve the <= 5-point
    problem; the radius increases strictly, so it terminates)."""
    P = np.asarray(points, dtype=np.float64)
    c = P.mean(axis=0)
    support = P[[int(np.argmax(((P - c) ** 2).sum(axis=1)))]]
    c, r = support[0].copy(), 0.0
    for _ in range(max_iter):
        d = np.sqrt(((P - c) ** 2).sum(axis=1))
        j = int(np.argmax(d))
        if d[j] <= r * (1 + 1e-12) + 1e-12:
            return c, float(d.max())
        c, r, support = _small_meb(np.vstack([support, P[j]]))
    raise RuntimeError("minimal enclosing ball did not converge")


def bounding_sphere(cameras: Sequence[Camera], load_mask: Callable[[int], np.ndarray], margin: float = 0.05,
                    coarse: int = 64, fine: int = 128, pool_target: int = 1200) -> Sphere:
    """Bounding sphere of the visual hull. ``load_mask(i)`` returns the boolean mask of ``cameras[i]``."""
    views: List[Tuple[Camera, np.ndarray, int]] = []
    for i, cam in enumerate(cameras):
        m = np.asarray(load_mask(i), bool)
        if m.shape != (cam.height, cam.width):
            raise ValueError(f"mask {i}: shape {m.shape} != camera {(cam.height, cam.width)}")
        pm, k = pool_mask(m, pool_target)
        views.append((cam, pm, k))
    if not views:
        raise ValueError("no mask for the normalisation")
    c0 = _centroid_rays_center(views)
    h0 = _initial_half_size(views, c0)
    for _ in range(4):
        g, step = _grid(c0 - h0, c0 + h0, coarse)
        pts = g[_carve(g, views)]
        if len(pts) == 0:
            raise ValueError("visual hull is empty (inconsistent masks/cameras?)")
        touches = np.any(pts.min(axis=0) < c0 - h0 + step) or np.any(pts.max(axis=0) > c0 + h0 - step)
        if not touches:
            break
        h0 *= 2.0  # the hull reaches the box: the object was not fully seen at this size, enlarge
    lo, hi = pts.min(axis=0) - step, pts.max(axis=0) + step
    g2, step2 = _grid(lo, hi, fine)
    pts2 = g2[_carve(g2, views)]
    if len(pts2) == 0:
        pts2, step2 = pts, step
    c, r = minimal_enclosing_ball(pts2)
    r_out = (r + 0.5 * np.sqrt(3.0) * step2) * (1.0 + margin)
    info = {"method": "visual-hull bounding sphere", "margin": margin, "voxel": step2, "n_voxels": int(len(pts2)),
            "hull_bbox": [pts2.min(axis=0).tolist(), pts2.max(axis=0).tolist()], "n_views": len(views),
            "hull_radius": r}
    return Sphere(center=c, radius=float(r_out), info=info)
