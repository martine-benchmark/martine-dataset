# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""View subsets of N views, comparable between MVS and MVPS (``--n-views``, ``martine-convert subsets``).

* MVS: k-means on the camera centres of the MVS ``sfm.json`` (scikit-learn ``KMeans``, ``random_state=42``,
  ``n_init=10``), then one view per cluster: the camera nearest to each centroid, without duplicate (optimal
  assignment, which is the plain nearest camera whenever those are all different).
* MVPS: for each selected MVS view, the nearest MVPS pose, without duplicate (optimal assignment on the distance
  between camera centres, scipy ``linear_sum_assignment``).

Only the camera centres are used, so the subsets do not depend on the resolution tier. The subsets of the published
objects for N = 5, 10, 20, 40 are precomputed in ``view_subsets.json`` (``martine-convert subsets --regenerate``):
for them, neither the MVS ``sfm.json`` nor scikit-learn and scipy are needed. Other subsets are computed, which needs
the MVS ``sfm.json`` of the object and scikit-learn and scipy (extra ``subsets``).
"""
from __future__ import annotations

import functools
import json
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .sfm import Scene, SfmError, read_sfm

PathLike = Union[str, "os.PathLike[str]"]
SEED = 42
TABLE = "view_subsets.json"
TABLE_N = (5, 10, 20, 40)


def _deps():
    try:
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ValueError("view subsets need scikit-learn and scipy: pip install 'martine-converters[subsets]' "
                         "(uv sync --extra subsets)") from exc
    return KMeans, linear_sum_assignment, cdist


def centres(scene: Scene) -> np.ndarray:
    return np.stack([np.asarray(u.camera.C, dtype=np.float64).reshape(3) for u in scene.units])


def kmeans_views(centers: np.ndarray, n: int, seed: int = SEED) -> List[int]:
    """Indices (sorted) of the ``n`` cameras nearest to the k-means centroids of ``centers``."""
    centers = np.asarray(centers, dtype=np.float64)
    if not 1 <= n <= len(centers):
        raise ValueError(f"--n-views {n}: between 1 and {len(centers)} views")
    if n == len(centers):
        return list(range(n))
    KMeans, linear_sum_assignment, cdist = _deps()
    km = KMeans(n_clusters=n, random_state=seed, n_init=10).fit(centers)
    _, cols = linear_sum_assignment(cdist(km.cluster_centers_, centers))
    return sorted(int(c) for c in cols)


def match_views(src: np.ndarray, dst: np.ndarray) -> Tuple[List[int], List[float]]:
    """For each ``src`` centre, a distinct ``dst`` centre minimising the summed distance; returns indices, distances."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if len(src) > len(dst):
        raise ValueError(f"--n-views {len(src)}: only {len(dst)} MVPS poses to pair with")
    _, linear_sum_assignment, cdist = _deps()
    cost = cdist(src, dst)
    rows, cols = linear_sum_assignment(cost)
    order = np.argsort(rows)
    return [int(c) for c in cols[order]], [float(cost[r, c]) for r, c in zip(rows[order], cols[order])]


def _read(path: PathLike, modality: str) -> Scene:
    scene = read_sfm(path, frame="gt")
    if scene.modality != modality:
        raise SfmError(f"{path}: expected an {modality.upper()} sfm.json, got {scene.modality.upper()}")
    return scene


def paired_subsets(mvs_sfm: PathLike, mvps_sfm: PathLike, n: int, seed: int = SEED) -> dict:
    """``{"mvs": [...], "mvps": [...], "distance_mm": [...]}``: the k-means MVS views (capture order) and, pair by
    pair, the MVPS pose matched to each, with the distance between the two camera centres (mm)."""
    mvs, mvps = _read(mvs_sfm, "mvs"), _read(mvps_sfm, "mvps")
    c_mvs = centres(mvs)
    sel = kmeans_views(c_mvs, n, seed)
    idx, dist = match_views(c_mvs[sel], centres(mvps))
    return {"mvs": [mvs.units[i].name for i in sel], "mvps": [mvps.units[i].name for i in idx], "distance_mm": dist}


@functools.lru_cache(maxsize=None)
def load_table() -> dict:
    """Precomputed subsets: ``{"n_views": [...], "objects": {object: {"N": {"mvs", "mvps", "distance_mm"}}}}``."""
    return json.loads(files("martine_converters").joinpath(TABLE).read_text())


def table_entry(obj: Optional[str], n: int) -> Optional[dict]:
    return load_table()["objects"].get(obj, {}).get(str(n)) if obj else None


def object_of(sfm_path: PathLike) -> Optional[str]:
    """Object name of an sfm.json in the published layout ``<root>/<object>/<modality>/<tier>/sfm.json``."""
    p = Path(os.path.abspath(sfm_path))
    return p.parents[2].name if p.parent.parent.name in ("mvs", "mvps") else None


def no_table(obj: Optional[str], n: int) -> str:
    return (f"no precomputed subset for {obj or 'this sfm.json'} with N={n} (precomputed: N = "
            f"{', '.join(map(str, TABLE_N))} for the published objects); computing it needs the MVS sfm.json of the "
            "object and the extra subsets (pip install 'martine-converters[subsets]')")


def _tier_key(p: Path):
    m = re.fullmatch(r"d(\d+)", p.parent.name)
    return (0, int(m.group(1))) if m else (1, p.parent.name)


def locate_pair(sfm_path: PathLike, modality: str) -> Path:
    """sfm.json of the other modality of the same object in the published layout
    ``<root>/<object>/<modality>/<tier>/sfm.json``: the same tier when present, else the smallest one."""
    p = Path(os.path.abspath(sfm_path))
    if p.parent.parent.name in ("mvs", "mvps"):
        base = p.parents[2] / modality
        same = base / p.parent.name / "sfm.json"
        if same.is_file():
            return same
        found = sorted(base.glob("*/sfm.json"), key=_tier_key)
        if found:
            return found[0]
    raise SfmError(f"view subsets of an MVPS sfm.json are paired with the MVS views of the same object: no "
                   f"<root>/<object>/{modality}/<tier>/sfm.json next to {p}")


def first_sfm(data: PathLike, obj: str, modality: str, tier: Optional[str] = None) -> Path:
    """``<data>/<obj>/<modality>/<tier>/sfm.json``, or that of the first tier when ``tier`` is absent."""
    base = Path(data) / obj / modality
    if tier and (base / tier / "sfm.json").is_file():
        return base / tier / "sfm.json"
    found = sorted(base.glob("*/sfm.json"), key=_tier_key)
    if not found:
        raise SfmError(f"no {base}/<tier>/sfm.json")
    return found[0]


def subset_names(sfm_path: PathLike, n: int, obj: Optional[str] = None, seed: int = SEED) -> List[str]:
    """Names of the ``n`` selected units of this sfm.json (MVS or MVPS), in its capture order: the precomputed
    subset of the object when there is one, else computed."""
    scene = read_sfm(sfm_path, frame="gt")
    obj = obj or object_of(sfm_path)
    entry = table_entry(obj, n)
    if entry is not None:
        chosen: Sequence[str] = entry[scene.modality]
        missing = sorted(set(chosen) - set(scene.names))
        if missing:
            raise SfmError(f"{sfm_path}: views {missing} of the precomputed subset of {obj} (N={n}) are not in it")
    elif scene.modality == "mvs":
        chosen = [scene.units[i].name for i in kmeans_views(centres(scene), n, seed)]
    else:
        try:
            mvs_sfm = locate_pair(sfm_path, "mvs")
        except SfmError as exc:
            raise SfmError(f"{no_table(obj, n)}. {exc}") from None
        chosen = paired_subsets(mvs_sfm, sfm_path, n, seed)["mvps"]
    keep = set(chosen)
    return [u.name for u in scene.units if u.name in keep]


def build_table(data: PathLike, n_views: Sequence[int] = TABLE_N, seed: int = SEED) -> dict:
    """Subsets of every object of ``data`` holding both MVS and MVPS, for each N (first tier of each modality)."""
    root = Path(data)
    objects = {}
    for obj in sorted(p.name for p in root.iterdir() if (p / "mvs").is_dir() and (p / "mvps").is_dir()):
        mvs, mvps = first_sfm(root, obj, "mvs"), first_sfm(root, obj, "mvps")
        objects[obj] = {}
        for n in n_views:
            res = paired_subsets(mvs, mvps, n, seed)
            objects[obj][str(n)] = {**res, "distance_mm": [round(d, 2) for d in res["distance_mm"]]}
    return {"description": "Paired MVS / MVPS view subsets of the published objects, in published view names. MVS: "
                           f"k-means on the MVS camera centres (scikit-learn KMeans, random_state={seed}, n_init=10), "
                           "camera nearest to each centroid; MVPS: pose nearest to each selected MVS view, without "
                           "duplicate (scipy linear_sum_assignment). distance_mm: distance between the paired camera "
                           "centres. Written by martine-convert subsets --regenerate.",
            "n_views": list(n_views), "objects": objects}


def dump_table(table: dict) -> str:
    """JSON with one line per (object, N) entry."""
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in table.items() if k != "objects"]
    lines.append(' "objects": {')
    objs = list(table["objects"].items())
    for i, (obj, by_n) in enumerate(objs):
        lines.append(f"  {json.dumps(obj)}: {{")
        items = list(by_n.items())
        lines += [f"   {json.dumps(n)}: {json.dumps(e)}{',' if j < len(items) - 1 else ''}" for j, (n, e) in enumerate(items)]
        lines.append("  }" + ("," if i < len(objs) - 1 else ""))
    return "\n".join(lines + [" }", "}"]) + "\n"
