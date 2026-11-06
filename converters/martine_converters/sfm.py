# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Reader of AliceVision ``sfm.json`` (SfMData JSON) files. Pure numpy + json.

Semantics checked against AliceVision v3.3.0 (``src/aliceVision/sfmDataIO/jsonIO.{hpp,cpp}``):

* ``pose.transform.rotation`` holds the 9 coefficients of the world-to-camera matrix in Eigen column-major order
  (``saveMatrix(pose.rotation())``), i.e. read row-major it is the camera-to-world rotation;
  ``center`` is the camera centre. Both in AliceVision's raw world.
* ``principalPoint`` is an offset from the image centre (versions >= 1.2.1): ``cx = width / 2 + pp[0]``, with the
  centre of the top-left pixel at (0, 0).
* focal: ``focalLength`` (mm) and ``sensorWidth`` (versions >= 1.2.2), ``pixelRatio`` handled as AliceVision does.

The returned cameras are in the evaluation (GT) frame ``X_gt = W X_raw`` with ``W = diag(1, -1, -1)`` unless
``frame="raw"``. The reader never looks at the resolution tier: every number comes from the file itself.
"""
from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .camera import W_CORR, Camera

PathLike = Union[str, "os.PathLike[str]"]


class SfmError(ValueError):
    pass


@dataclass(frozen=True)
class Unit:
    """One camera position: an MVS view (one image) or an MVPS pose (one image per LED)."""

    name: str
    camera: Camera
    images: Tuple[Tuple[str, Path], ...]
    mask: Optional[Path]
    pose_id: str
    view_ids: Tuple[str, ...]
    intrinsic_id: str

    def image(self, key: Optional[str] = None) -> Path:
        if key is None:
            return self.images[0][1]
        for k, p in self.images:
            if k == key:
                return p
        raise KeyError(f"{self.name}: no image {key!r} (have {[k for k, _ in self.images]})")

    @property
    def image_keys(self) -> Tuple[str, ...]:
        return tuple(k for k, _ in self.images)


@dataclass(frozen=True)
class Scene:
    sfm_path: Path
    modality: str  # "mvs" (one image per pose) or "mvps" (several images per pose)
    frame: str  # "gt" or "raw"
    version: Tuple[int, int, int]
    units: Tuple[Unit, ...]

    @property
    def names(self) -> List[str]:
        return [u.name for u in self.units]

    def unit(self, name: str) -> Unit:
        for u in self.units:
            if u.name == name:
                return u
        raise KeyError(name)

    def with_units(self, units: Sequence[Unit]) -> "Scene":
        return Scene(self.sfm_path, self.modality, self.frame, self.version, tuple(units))


def _version(data: dict) -> Tuple[int, int, int]:
    v = data.get("version", ["1", "2", "12"])
    try:
        return tuple(int(x) for x in v)[:3]  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise SfmError(f"bad sfm version {v!r}") from exc


def _floats(seq) -> np.ndarray:
    return np.array([float(x) for x in seq], dtype=np.float64)


def intrinsic_to_pixels(intr: dict, version: Tuple[int, int, int]) -> Tuple[int, int, float, float, float, float]:
    """``(width, height, fx, fy, cx, cy)`` of an AliceVision pinhole intrinsic, AliceVision pixel convention."""
    itype = intr.get("type", "pinhole")
    if itype != "pinhole":
        raise SfmError(f"intrinsic {intr.get('intrinsicId')}: type {itype!r} not supported (pinhole only)")
    dtype = intr.get("distortionType", "none")
    dparams = intr.get("distortionParams", "")
    if dtype not in ("none", "", None) or (isinstance(dparams, list) and any(float(p) != 0 for p in dparams)):
        raise SfmError(f"intrinsic {intr.get('intrinsicId')}: distortion {dtype!r} not supported "
                       "(the published images are undistorted)")
    w, h = int(intr["width"]), int(intr["height"])
    sensor_w = float(intr.get("sensorWidth", 36.0))
    ratio = float(intr.get("pixelRatio", 1.0))
    if version < (1, 2, 2):
        pxf = intr["pxFocalLength"]
        fx, fy = (float(pxf[0]), float(pxf[1])) if isinstance(pxf, list) else (float(pxf), float(pxf))
    elif version < (1, 2, 5):  # pixelRatio stored the focal ratio fx / fy (jsonIO.cpp:321-331)
        fx = float(intr["focalLength"]) / sensor_w * w
        fy = fx / ratio
    elif version < (1, 2, 11):  # compatibility mode (IntrinsicScaleOffset.cpp:220-226)
        fx = float(intr["focalLength"]) * w / sensor_w
        fy = float(intr["focalLength"]) * ratio * w / sensor_w
    else:  # IntrinsicScaleOffset.cpp:228-234
        fx = float(intr["focalLength"]) / ratio * w / sensor_w
        fy = float(intr["focalLength"]) * w / sensor_w
    pp = _floats(intr.get("principalPoint", ["0", "0"]))
    if version < (1, 2, 1):  # absolute principal point in old files (jsonIO.cpp:277-281)
        pp = pp - np.array([w / 2.0, h / 2.0])
    return w, h, fx, fy, w / 2.0 + pp[0], h / 2.0 + pp[1]


def _resolve(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def read_sfm(sfm_path: PathLike, frame: str = "gt", masks_dir: Optional[PathLike] = None) -> Scene:
    """Read an AliceVision ``sfm.json``. Units are sorted by name (capture order for the published data)."""
    if frame not in ("gt", "raw"):
        raise ValueError(f"frame must be 'gt' or 'raw', got {frame!r}")
    sfm_path = Path(sfm_path)
    if not sfm_path.is_file():
        raise SfmError(f"no sfm.json at {sfm_path}")
    with open(sfm_path) as f:
        data = json.load(f)
    version = _version(data)
    base = sfm_path.parent
    mdir = Path(masks_dir) if masks_dir is not None else base / "masks"
    M = W_CORR if frame == "gt" else np.eye(3)

    intrinsics = {str(i["intrinsicId"]): i for i in data.get("intrinsics", [])}
    poses = {str(p["poseId"]): p["pose"]["transform"] for p in data.get("poses", [])}

    groups: "OrderedDict[str, List[dict]]" = OrderedDict()
    skipped = 0
    for v in data.get("views", []):
        pid, iid = str(v.get("poseId", "")), str(v.get("intrinsicId", ""))
        if pid not in poses or iid not in intrinsics:
            skipped += 1
            continue
        groups.setdefault(pid, []).append(v)
    if not groups:
        raise SfmError(f"{sfm_path}: no view with both a pose and an intrinsic ({skipped} views without)")
    modality = "mvps" if any(len(vs) > 1 for vs in groups.values()) else "mvs"

    cache: Dict[str, Tuple] = {}
    units = []
    for pid, views in groups.items():
        iids = {str(v["intrinsicId"]) for v in views}
        if len(iids) != 1:
            raise SfmError(f"pose {pid}: views with different intrinsics {sorted(iids)}")
        iid = iids.pop()
        if iid not in cache:
            cache[iid] = intrinsic_to_pixels(intrinsics[iid], version)
        w, h, fx, fy, cx, cy = cache[iid]
        tr = poses[pid]
        R_c2w_raw = _floats(tr["rotation"]).reshape(3, 3)
        C_raw = _floats(tr["center"])
        cam = Camera(w, h, fx, fy, cx, cy, R_c2w_raw.T, C_raw).transformed(M)
        # (key, path, viewId) sorted by key: images[i] and view_ids[i] describe the same view
        entries = sorted((_resolve(v["path"], base).stem, _resolve(v["path"], base), str(v["viewId"]))
                         for v in views)
        if modality == "mvs":
            name = entries[0][0]
        else:
            parents = {e[1].parent for e in entries}
            if len(parents) != 1:
                raise SfmError(f"pose {pid}: images in several folders {sorted(map(str, parents))}")
            name = parents.pop().name
        mask = mdir / f"{name}.png"
        units.append(Unit(name=name, camera=cam, images=tuple((k, p) for k, p, _ in entries),
                          mask=mask if mask.is_file() else None, pose_id=pid,
                          view_ids=tuple(vid for _, _, vid in entries), intrinsic_id=iid))
    units.sort(key=lambda u: _natural_key(u.name))
    names = [u.name for u in units]
    if len(set(names)) != len(names):
        raise SfmError(f"{sfm_path}: duplicated unit names")
    return Scene(sfm_path=sfm_path, modality=modality, frame=frame, version=version, units=tuple(units))


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def select_units(scene: Scene, spec: Optional[str]) -> Scene:
    """Subset of units: comma-separated names, indices or python slices (``0:84:4``) of the sorted units."""
    if not spec:
        return scene
    units = list(scene.units)
    by_name = {u.name: u for u in units}
    chosen: List[Unit] = []
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        if item in by_name:
            chosen.append(by_name[item])
        elif re.fullmatch(r"-?\d*:-?\d*(:-?\d+)?", item):
            parts = [int(p) if p else None for p in item.split(":")]
            chosen.extend(units[slice(*parts)])
        elif re.fullmatch(r"-?\d+", item):
            if not -len(units) <= int(item) < len(units):
                raise SfmError(f"view index {item} out of range (0..{len(units) - 1})")
            chosen.append(units[int(item)])
        else:
            raise SfmError(f"unknown view {item!r}")
    seen, out = set(), []
    for u in chosen:
        if u.name not in seen:
            seen.add(u.name)
            out.append(u)
    if not out:
        raise SfmError(f"view selection {spec!r} is empty")
    return scene.with_units(out)


def locate_sfm(data_root: PathLike, obj: str, tier: str, modality: str = "mvs") -> Path:
    """Convenience: ``<root>/<object>/<modality>/<tier>/sfm.json`` (file location only; never used in the maths)."""
    p = Path(data_root) / obj / modality / tier / "sfm.json"
    if not p.is_file():
        raise SfmError(f"no sfm.json at {p}")
    return p
