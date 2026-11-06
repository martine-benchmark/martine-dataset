# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Shared helpers of the format writers: options, output folder, per-view image/mask export, manifest."""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from . import __version__
from .camera import Camera
from .imageio import read_exr, read_image, read_mask, tonemap, write_image, write_mask
from .sfm import Scene, Unit


class ConversionError(RuntimeError):
    pass


@dataclass
class TriOptions:
    backend: str = "auto"  # "auto" (pycolmap if installed, else the executable), "colmap", "pycolmap"
    colmap: str = "colmap"
    gpu: bool = False
    max_image_size: int = 3200  # COLMAP's default; keypoints are rescaled exactly by COLMAP
    threads: int = -1
    matcher: str = "exhaustive"
    min_track: int = 3
    erode_px: int = 3
    keep_work: bool = False
    # explicit SIFT settings, off by default (COLMAP's own values; the paper used the defaults)
    peak_threshold: Optional[float] = None  # SIFT peak threshold (None: COLMAP default 0.02/3)
    domain_size_pooling: bool = False  # DSP-SIFT (CPU extraction)


@dataclass
class Options:
    images: bool = True
    copy_mode: str = "copy"
    tonemap: Optional[str] = None
    background: Optional[str] = None
    lighting: Optional[str] = None
    workers: int = 4
    pixel_centre: Optional[float] = None
    points: str = "auto"  # "auto" (format default), "triangulate", "none"
    sphere_margin: float = 0.05
    image_format: Optional[str] = None
    overwrite: bool = False
    tri: TriOptions = field(default_factory=TriOptions)
    extra: dict = field(default_factory=dict)


MANIFEST = "martine_conversion.json"


def prepare_out(out: Path, overwrite: bool) -> Path:
    """Create the output folder. ``overwrite`` empties it first, but only if it is a previous conversion output
    (contains the manifest): loaders that glob their folders must not see files of an earlier run."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise ConversionError(f"{out} exists and is not empty (use --overwrite)")
        if not (out / MANIFEST).is_file():
            raise ConversionError(f"{out} is not empty and is not a conversion output ({MANIFEST} missing): "
                                  "refusing to empty it")
        for child in out.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).write_text(json.dumps({"status": "conversion in progress or failed"}) + "\n")
    return out


def source_image(unit: Unit, lighting: Optional[str]) -> Path:
    """The image of an MVS view, or the chosen LED image of an MVPS pose (default: first LED)."""
    if lighting is None:
        return unit.image()
    try:
        key = lighting if lighting.startswith("lighting_") else f"lighting_{int(lighting):02d}"
        return unit.image(key)
    except (KeyError, ValueError) as exc:
        raise ConversionError(f"{unit.name}: no lighting {lighting!r} (have {list(unit.image_keys)[:3]}...)") \
            from exc


def need_mask(unit: Unit) -> Path:
    if unit.mask is None:
        raise ConversionError(f"{unit.name}: mask not found (expected masks/{unit.name}.png next to sfm.json, "
                              "or pass --masks)")
    return unit.mask


def load_mask_checked(unit: Unit) -> np.ndarray:
    m = read_mask(need_mask(unit))
    cam = unit.camera
    if m.shape != (cam.height, cam.width):
        raise ConversionError(f"{unit.name}: mask {m.shape[1]}x{m.shape[0]} != camera {cam.width}x{cam.height}")
    return m


def load_linear(path: Path, cam: Camera) -> np.ndarray:
    img = read_exr(path) if path.suffix.lower() == ".exr" else read_image(path).astype(np.float32)
    if img.shape[:2] != (cam.height, cam.width):
        raise ConversionError(f"{path}: image {img.shape[1]}x{img.shape[0]} != camera {cam.width}x{cam.height}")
    return img


def export_ldr(unit: Unit, src: Path, dst: Path, tm: str, background: str, alpha: bool = False) -> None:
    """Tone-mapped 8-bit image (RGB, or RGBA with alpha = mask)."""
    img = load_linear(src, unit.camera)
    mask = load_mask_checked(unit) if (background == "black" or alpha or tm == "object99") else None
    out = tonemap(img, tm, mask=mask)
    if background == "black":
        out = np.where(mask[..., None], out, 0).astype(out.dtype)
    if alpha:
        out = np.concatenate([out, np.where(mask, np.iinfo(out.dtype).max, 0).astype(out.dtype)[..., None]], axis=2)
    write_image(dst, out)


def export_mask(unit: Unit, dst: Path) -> None:
    write_mask(dst, load_mask_checked(unit))


def run_parallel(fn: Callable, items: Sequence, workers: int) -> List:
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def write_manifest(out: Path, scene: Scene, fmt: str, offset: float, opts: Options, extra: dict) -> None:
    man = {
        "tool": f"martine converters {__version__}",
        "command": " ".join(sys.argv),
        "format": fmt,
        "source_sfm": str(scene.sfm_path),
        "frame": scene.frame,
        "pixel_centre_offset_added": offset,
        "note": "principal point = AliceVision (centre of the top-left pixel at (0, 0)) + offset; "
                "no resolution-dependent term",
        "modality": scene.modality,
        "units": [u.name for u in scene.units],
        "options": {k: v for k, v in asdict(opts).items() if k != "extra"},
        **extra,
    }
    (Path(out) / MANIFEST).write_text(json.dumps(man, indent=2, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def remove_tree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)
