# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Image and mask I/O, tone mapping and exact box downscaling.

Arrays are always RGB(A) in memory; the BGR order of OpenCV is handled here only. EXR files are read with OpenCV
(``OPENCV_IO_ENABLE_OPENEXR``) or, when installed, the ``OpenEXR`` package.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np

try:  # optional at import time: camera-only conversions do not need any image library
    import cv2  # noqa: E402
except ImportError:  # pragma: no cover - exercised only without OpenCV
    cv2 = None

TONEMAPS = ("paper", "object99", "linear8", "linear16")


def _need_cv2():
    if cv2 is None:
        raise ImportError("opencv-python(-headless) is required for image/mask conversion "
                          "(pip install opencv-python-headless); use --no-images for cameras only")


# --- reading ------------------------------------------------------------------------------------------------------
def read_exr(path) -> np.ndarray:
    """Float32 RGB (H, W, 3) from an EXR file."""
    path = str(path)
    try:
        import OpenEXR  # type: ignore

        with OpenEXR.File(path) as f:
            ch = f.channels()
            if "RGB" in ch:
                return np.ascontiguousarray(ch["RGB"].pixels, dtype=np.float32)[..., :3]
            if "RGBA" in ch:
                return np.ascontiguousarray(ch["RGBA"].pixels, dtype=np.float32)[..., :3]
            if all(c in ch for c in "RGB"):
                return np.stack([np.asarray(ch[c].pixels, dtype=np.float32) for c in "RGB"], axis=-1)
    except (ImportError, AttributeError):  # no OpenEXR, or bindings older than 3.3 (no OpenEXR.File)
        pass
    _need_cv2()
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path} (OpenCV built without OpenEXR? set OPENCV_IO_ENABLE_OPENEXR=1)")
    img = img.astype(np.float32)
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    return img[..., 2::-1].copy() if img.shape[2] >= 3 else np.repeat(img[..., :1], 3, axis=2)


def read_image(path) -> np.ndarray:
    """Any image -> RGB array (float32 for EXR, native integer type otherwise)."""
    path = Path(path)
    if path.suffix.lower() == ".exr":
        return read_exr(path)
    _need_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return img[..., [2, 1, 0, 3]].copy()
    return img[..., ::-1].copy()


def read_mask(path) -> np.ndarray:
    """Boolean mask (True = object) from a PNG (any channel count, > half range = object)."""
    img = read_image(path)
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype == np.uint8:
        return img > 127
    if img.dtype == np.uint16:
        return img > 32767
    return img > 0.5


# --- writing ------------------------------------------------------------------------------------------------------
def write_image(path, img: np.ndarray, jpeg_quality: int = 95) -> None:
    """Write an RGB/RGBA/grey uint8 or uint16 array (PNG, JPG, TIF) or a float array (EXR)."""
    _need_cv2()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[..., ::-1]
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., [2, 1, 0, 3]]
    params = []
    suffix = path.suffix.lower()
    if suffix == ".exr":
        arr = arr.astype(np.float32)
        params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
    elif suffix in (".jpg", ".jpeg"):
        if arr.dtype != np.uint8:
            raise ValueError("JPEG needs uint8")
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif arr.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"{path}: integer image expected, got {arr.dtype}")
    if not cv2.imwrite(str(path), np.ascontiguousarray(arr), params):
        raise IOError(f"cannot write {path}")


def write_mask(path, mask: np.ndarray) -> None:
    write_image(path, np.where(np.asarray(mask, bool), 255, 0).astype(np.uint8))


def write_exr_rgb(path, img: np.ndarray) -> None:
    """Float32 EXR with channels named R, G, B = img[..., 0..2] (what ``pyexr.read(...)[..., :3]`` returns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.float32)[..., :3])
    try:
        import OpenEXR  # type: ignore

        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        with OpenEXR.File(header, {"RGB": arr}) as f:
            f.write(str(path))
        return
    except (ImportError, AttributeError):  # no OpenEXR, or bindings older than 3.3
        pass
    write_image(path, arr)


def place_file(src, dst, mode: str = "copy") -> None:
    """Copy, symlink or hardlink ``src`` to ``dst`` (replacing ``dst``)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copyfile(src, dst)
    else:
        raise ValueError(f"unknown copy mode {mode!r}")


# --- tone mapping -------------------------------------------------------------------------------------------------
def tonemap(img: np.ndarray, mode: str = "paper", mask: Optional[np.ndarray] = None,
            scale: Optional[float] = None) -> np.ndarray:
    """Linear float RGB -> uint8/uint16.

    * ``paper``: ``exr_to_ldr(exposure=0)`` of the paper's runs (dataset_temp/sfm_converters.py): if max > 1, the
      95th percentile of the positive values is mapped to 0.9 and Reinhard ``x/(1+x)`` is applied; then clip, gamma
      1/2.2, x255 and truncation to uint8.
    * ``object99``: divide by the 99.5th percentile of the object pixels (``mask``), clip, gamma 1/2.2, round (the
      images of the known-pose triangulation of the staging, led_reorder/scripts/colmap_tri.py).
    * ``linear8`` / ``linear16``: ``x * scale`` (default: 1 / max), clip, round. No gamma: for photometric stereo.
    """
    x = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "paper":
        if x.max() > 1.0:
            pos = x[x > 0]
            p95 = float(np.percentile(pos, 95)) if pos.size else 1.0
            x = x * (0.9 / max(p95, 1e-6))
            if x.max() > 1.0:
                x = x / (1.0 + x)
        x = np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.2)
        return (x * 255).astype(np.uint8)
    if mode == "object99":
        m = np.asarray(mask, bool) if mask is not None else None
        vals = x[m] if m is not None and m.any() else x[x > 0]
        ref = float(np.percentile(vals, 99.5)) if vals.size else 1.0
        x = np.power(np.clip(x / max(ref, 1e-12), 0.0, 1.0), 1.0 / 2.2)
        return np.round(x * 255.0).astype(np.uint8)
    if mode in ("linear8", "linear16"):
        top = 255.0 if mode == "linear8" else 65535.0
        s = scale if scale is not None else 1.0 / max(float(x.max()), 1e-12)
        y = np.clip(x * s, 0.0, 1.0)
        return np.round(y * top).astype(np.uint8 if mode == "linear8" else np.uint16)
    raise ValueError(f"unknown tone map {mode!r} (known: {TONEMAPS})")
