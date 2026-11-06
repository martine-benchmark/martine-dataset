# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Pinhole camera in the AliceVision pixel convention, plus rotation helpers. Pure numpy.

Conventions used everywhere in this package:

* camera frame = OpenCV (x right, y down, z forward);
* ``R`` maps world to camera: ``x_cam = R (X - C)``;
* intrinsics ``fx, fy, cx, cy`` in the AliceVision convention: the centre of pixel index ``(i, j)`` is at image
  coordinate ``(i, j)``. :meth:`Camera.K` adds the pixel-centre offset of a target method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# AliceVision raw world -> evaluation (GT) frame (eval-3d, core/camera.py: WORLD_CORR). A rotation (det = +1).
W_CORR = np.diag([1.0, -1.0, -1.0])
# OpenCV camera axes <-> OpenGL camera axes (x right, y up, z backward).
CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def _frozen(a) -> np.ndarray:
    out = np.array(a, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray = field(repr=False)
    C: np.ndarray = field(repr=False)

    def __post_init__(self):
        R = np.asarray(self.R, dtype=np.float64)
        C = np.asarray(self.C, dtype=np.float64).reshape(3)
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {R.shape}")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0:
            raise ValueError("R is not a rotation matrix")
        object.__setattr__(self, "R", _frozen(R))
        object.__setattr__(self, "C", _frozen(C))

    # --- derived quantities -----------------------------------------------------------------------------------
    @property
    def t(self) -> np.ndarray:
        return -self.R @ self.C

    def K(self, offset: float = 0.0) -> np.ndarray:
        """3x3 intrinsics with the principal point shifted by ``offset`` (the target method's convention)."""
        return np.array([[self.fx, 0.0, self.cx + offset],
                         [0.0, self.fy, self.cy + offset],
                         [0.0, 0.0, 1.0]])

    def w2c(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def c2w(self) -> np.ndarray:
        """4x4 camera-to-world, OpenCV camera axes."""
        M = np.eye(4)
        M[:3, :3] = self.R.T
        M[:3, 3] = self.C
        return M

    def projection(self, offset: float = 0.0) -> np.ndarray:
        """3x4 ``P = K [R | t]``."""
        return self.K(offset) @ self.w2c()[:3]

    def project(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """World points (N, 3) -> (pixel coordinates in the AliceVision convention (N, 2), depth (N,))."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        xc = (X - self.C) @ self.R.T
        z = xc[:, 2]
        uv = np.stack([self.fx * xc[:, 0] / z + self.cx, self.fy * xc[:, 1] / z + self.cy], axis=1)
        return uv, z

    # --- transformations (return new cameras) -------------------------------------------------------------------
    def transformed(self, M: np.ndarray) -> "Camera":
        """Same camera after a world rotation ``X' = M X`` (``M`` 3x3 rotation)."""
        M = np.asarray(M, dtype=np.float64)
        return Camera(self.width, self.height, self.fx, self.fy, self.cx, self.cy, self.R @ M.T, M @ self.C)

    def intrinsics_equal(self, other: "Camera", tol: float = 0.0) -> bool:
        a = np.array([self.width, self.height, self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        b = np.array([other.width, other.height, other.fx, other.fy, other.cx, other.cy], dtype=np.float64)
        return bool(np.all(np.abs(a - b) <= tol))


# --- rotations -------------------------------------------------------------------------------------------------
def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (qw, qx, qy, qz), Hamilton, scalar first, qw >= 0 (COLMAP convention).
    Shepperd's method (numerically stable for every rotation)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (qw, qx, qy, qz) -> rotation matrix (same formula as COLMAP's read_write_model.qvec2rotmat)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])
