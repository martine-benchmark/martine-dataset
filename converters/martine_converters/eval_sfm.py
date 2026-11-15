# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Evaluation camera set of a view subset (``martine-convert subsets ... --write-eval-sfm DIR``).

The evaluation cleans the GT and the reconstruction by visibility in the evaluation cameras (by default
``gt/eval_cameras.sfm`` and ``gt/masks_eval/``). For a sparse-view result, ``DIR`` holds the cameras of the subset
instead, in the same layout:

* ``DIR/eval_cameras.sfm``: AliceVision SfMData restricted to the selected views, their poses and intrinsics, in the
  raw AliceVision frame of the source (that of ``eval_cameras.sfm``, the MVS and the MVPS ``sfm.json``) and in its view
  order. The view paths are ``masks/<stem>.png``: the evaluation finds the mask of a view by the basename stem of its
  path.
* ``DIR/masks/<stem>.png``: copies of the source masks (``gt/masks_eval/NNN.png``; MVPS ``masks/pose_KK.png``).
* ``DIR/masks_watertight/`` (MVS, objects with ``gt/masks_watertight/``): the evaluation pairs these masks with the
  cameras by index, in sorted file-name order, so the copies keep the camera order of ``DIR/eval_cameras.sfm``.
* ``DIR/eval_sfm.json``: what was selected and from which files.

MVS: source ``<data>/<object>/gt/eval_cameras.sfm`` (full resolution). MVPS: source
``<data>/<object>/mvps/<tier>/sfm.json`` (the real MVPS cameras at that tier), one view per pose (``lighting_00``, else
the first image of the pose). A pose is identified by its ``images/pose_KK/`` folder, never by its poseId alone: some
published MVPS files repeat a poseId once per image. No watertight mask is published for the MVPS poses:
``--watertight-masks DIR`` gives them (``DIR/<stem>.png`` for every selected view, the size of its mask), copied to
``DIR/masks_watertight/<stem>.png`` and paired with the cameras by name. It replaces the published ones for MVS too.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .export import ConversionError
from .sfm import SfmError

SFM_NAME = "eval_cameras.sfm"
MANIFEST = "eval_sfm.json"
# kept top-level keys; featuresFolders and matchesFolders (cache paths of the SfM run) are dropped
KEYS = ("version", "views", "intrinsics", "poses")
MVPS_TIER = "d2"
LIGHTING = "lighting_00"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _transform(t: dict) -> np.ndarray:
    return np.array([float(x) for x in list(t["rotation"]) + list(t["center"])], dtype=np.float64)


def _cameras(sfm: dict) -> List[dict]:
    """Views in the order the evaluation loads them: with a pose and an intrinsic, the first view of each poseId."""
    poses = {str(p["poseId"]) for p in sfm.get("poses", [])}
    intr = {str(i["intrinsicId"]) for i in sfm.get("intrinsics", [])}
    seen, out = set(), []
    for v in sfm.get("views", []):
        pid = str(v.get("poseId", ""))
        if pid in poses and str(v.get("intrinsicId", "")) in intr and pid not in seen:
            seen.add(pid)
            out.append(v)
    return out


def _subset_sfm(sfm: dict, views: Sequence[dict], stems: Sequence[str]) -> dict:
    """``sfm`` restricted to ``views`` (paths renamed ``masks/<stem>.png``), their poses (first entry of each poseId)
    and the intrinsics they use (source order)."""
    first: Dict[str, dict] = {}
    for p in sfm["poses"]:
        first.setdefault(str(p["poseId"]), p)
    used = {str(v["intrinsicId"]) for v in views}
    parts = {"views": [dict(v, path=f"masks/{s}.png") for v, s in zip(views, stems)],
             "poses": [first[str(v["poseId"])] for v in views],
             "intrinsics": [i for i in sfm["intrinsics"] if str(i["intrinsicId"]) in used]}
    return {k: parts.get(k, sfm.get(k)) for k in sfm if k in KEYS}


def _mvs(data: Path, obj: str, names: Sequence[str]) -> dict:
    gt = data / obj / "gt"
    src = gt / SFM_NAME
    if not src.is_file():
        raise SfmError(f"no {src}")
    sfm = json.loads(src.read_text())
    cams = _cameras(sfm)
    stems = [_stem(v["path"]) for v in cams]
    missing = sorted(set(names) - set(stems))
    if missing:
        raise SfmError(f"{src}: views {missing} are not posed cameras of it")
    keep = set(names)
    idx = [i for i, s in enumerate(stems) if s in keep]
    sel = [stems[i] for i in idx]
    plan = {"source": src, "sfm": _subset_sfm(sfm, [cams[i] for i in idx], sel), "views": sel,
            "masks_source": gt / "masks_eval", "masks": {f"{s}.png": gt / "masks_eval" / f"{s}.png" for s in sel},
            "watertight": {}}
    wdir = gt / "masks_watertight"
    if wdir.is_dir():
        wt = sorted(wdir.glob("*.png"))  # paired with the cameras by index, as in the evaluation
        if len(wt) != len(cams):
            raise SfmError(f"{wdir}: {len(wt)} masks for {len(cams)} cameras in {src}")
        out_names = [f"{s}.png" for s in sel]
        if sorted(out_names) != out_names:  # keep the camera order in the sorted file names
            out_names = [f"{k:03d}_{s}.png" for k, s in enumerate(sel)]
        plan["watertight"] = {n: wt[i] for n, i in zip(out_names, idx)}
    return plan


def _mvps(data: Path, obj: str, names: Sequence[str], tier: str) -> dict:
    src = data / obj / "mvps" / tier / "sfm.json"
    if not src.is_file():
        raise SfmError(f"no {src}")
    sfm = json.loads(src.read_text())
    transforms = defaultdict(list)
    for p in sfm.get("poses", []):
        transforms[str(p["poseId"])].append(_transform(p["pose"]["transform"]))
    intr = {str(i["intrinsicId"]) for i in sfm.get("intrinsics", [])}
    folders: Dict[str, List[dict]] = {}
    for v in sfm.get("views", []):
        folders.setdefault(Path(v["path"]).parent.name, []).append(v)
    missing = sorted(set(names) - set(folders))
    if missing:
        raise SfmError(f"{src}: poses {missing} have no images/<pose>/ view in it")
    keep, chosen, sel = set(names), [], []
    for name, views in folders.items():
        if name not in keep:
            continue
        posed = [v for v in views if str(v.get("poseId")) in transforms and str(v.get("intrinsicId")) in intr]
        if not posed:
            raise SfmError(f"{src}: {name} has no view with a pose and an intrinsic")
        pids = sorted({str(v["poseId"]) for v in posed})
        trs = [t for pid in pids for t in transforms[pid]]
        if any(not np.allclose(t, trs[0], rtol=0, atol=1e-9) for t in trs[1:]):
            raise SfmError(f"{src}: {name} resolves to different transforms (poseIds {', '.join(pids)}, "
                           f"{len(trs)} pose entries): refusing to choose one")
        lit = [v for v in posed if _stem(v["path"]) == LIGHTING]
        chosen.append(lit[0] if lit else min(posed, key=lambda v: v["path"]))
        sel.append(name)
    owner: Dict[str, str] = {}
    for name, v in zip(sel, chosen):
        pid = str(v["poseId"])
        if pid in owner:
            raise SfmError(f"{src}: {owner[pid]} and {name} share the poseId {pid}: the evaluation would keep one "
                           "camera for both")
        owner[pid] = name
    mdir = src.parent / "masks"
    return {"source": src, "sfm": _subset_sfm(sfm, chosen, sel), "views": sel, "masks_source": mdir,
            "masks": {f"{s}.png": mdir / f"{s}.png" for s in sel}, "watertight": {}}


def _png_size(path: Path) -> tuple:
    head = Path(path).read_bytes()[:24]
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise SfmError(f"{path} is not a PNG")
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def _given_watertight(plan: dict, wdir: Path) -> dict:
    """``<stem>.png`` of ``wdir`` for every selected view, the size of its silhouette mask."""
    out = {}
    for dst, mask in plan["masks"].items():
        src = Path(wdir) / dst
        if not src.is_file():
            raise SfmError(f"--watertight-masks {wdir}: no {dst} for the view {Path(dst).stem}")
        if mask.is_file() and _png_size(src) != _png_size(mask):
            raise SfmError(f"{src}: size {_png_size(src)} differs from the mask {mask} {_png_size(mask)}")
        out[dst] = src
    return out


def _prepare(out: Path, data: Path, overwrite: bool) -> None:
    o, d = Path(os.path.realpath(out)), Path(os.path.realpath(data))
    if o == d or d in o.parents:
        raise ConversionError(f"{out} is under the data root {data}: refusing to write there")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise ConversionError(f"{out} exists and is not empty (use --overwrite)")
        if not (out / MANIFEST).is_file():
            raise ConversionError(f"{out} is not empty and is not a --write-eval-sfm output ({MANIFEST} missing): "
                                  "refusing to empty it")
        for child in out.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).write_text(json.dumps({"status": "in progress or failed"}) + "\n")


def write_eval_sfm(data: Path, obj: str, pairs: dict, out: Path, modality: str = "mvs", tier: Optional[str] = None,
                   overwrite: bool = False, selection: Optional[dict] = None,
                   watertight_dir: Optional[Path] = None) -> dict:
    """Write the evaluation camera set of ``pairs[modality]`` (``pairs``: ``{"mvs", "mvps", "distance_mm"}`` of a
    preset or subset) to ``out``; returns the manifest. ``selection``: ``{"preset": name}`` or ``{"n_views": n}``."""
    data, out = Path(data), Path(out)
    if modality == "mvs":
        plan = _mvs(data, obj, pairs["mvs"])
        tier = None
    elif modality == "mvps":
        tier = tier or MVPS_TIER
        plan = _mvps(data, obj, pairs["mvps"], tier)
    else:
        raise SfmError(f"unknown modality {modality!r}")
    if watertight_dir is not None:
        plan["watertight"] = _given_watertight(plan, Path(watertight_dir))
    for dst, src in {**plan["masks"], **plan["watertight"]}.items():
        if not src.is_file():
            raise SfmError(f"no mask {src} for {dst}")
    _prepare(out, data, overwrite)
    for sub, files in (("masks", plan["masks"]), ("masks_watertight", plan["watertight"])):
        if files:
            (out / sub).mkdir()
        for dst, src in files.items():
            shutil.copyfile(src, out / sub / dst)
    (out / SFM_NAME).write_text(json.dumps(plan["sfm"], indent=4) + "\n")
    manifest = {"object": obj, **(selection or {}), "modality": modality, "tier": tier, "views": plan["views"],
                "sfm": SFM_NAME, "source": str(plan["source"]), "source_md5": _md5(plan["source"]),
                "frame": "raw AliceVision frame of the source (unchanged)", "masks": "masks",
                "masks_source": str(plan["masks_source"])}
    if plan["watertight"]:
        manifest["watertight"] = [str(p) for p in plan["watertight"].values()]
        manifest["watertight_dir"] = "masks_watertight"
        manifest["watertight_note"] = "sorted file names in the camera order of " + SFM_NAME
        if watertight_dir is not None:
            manifest["watertight_source"] = str(watertight_dir)
            manifest["watertight_note"] = "given by --watertight-masks, one per view, paired with the cameras by name"
    else:
        manifest["watertight"] = "none"
        manifest["watertight_note"] = ("no watertight masks exist for the MVPS poses" if modality == "mvps" else
                                       "the object has no gt/masks_watertight")
    if modality == "mvps":
        pair = dict(zip(pairs["mvps"], zip(pairs["mvs"], pairs["distance_mm"])))
        manifest["paired_mvs_views"] = [pair[n][0] for n in plan["views"]]
        manifest["distance_mm"] = [pair[n][1] for n in plan["views"]]
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
