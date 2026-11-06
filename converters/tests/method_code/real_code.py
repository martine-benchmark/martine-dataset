# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Import the ORIGINAL loader code of the methods when their repositories are available locally.

``MARTINE_METHOD_REPOS`` = folder containing the checkouts (PGSR, RNb-NeuS2, SuperNormal, neuralangelo). Each
helper returns None when the code (or its dependencies, e.g. torch) cannot be imported, and the tests then fall back
to the re-implementations of :mod:`methods`.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

REPOS = Path(os.environ.get("MARTINE_METHOD_REPOS") or "MARTINE_METHOD_REPOS-not-set")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def pgsr() -> Optional[dict]:
    """scene/colmap_loader.py (numpy only) + utils/graphics_utils.py (needs torch)."""
    root = REPOS / "PGSR"
    try:
        cl = _load(root / "scene" / "colmap_loader.py", "pgsr_colmap_loader")
    except Exception:
        return None
    out = {"read_extrinsics_text": cl.read_extrinsics_text, "read_intrinsics_text": cl.read_intrinsics_text,
           "qvec2rotmat": cl.qvec2rotmat, "has_torch": False}
    try:
        gu = _load(root / "utils" / "graphics_utils.py", "pgsr_graphics_utils")
    except Exception:  # torch missing: only the COLMAP reader is original code
        return out
    out.update(getWorld2View2=gu.getWorld2View2, focal2fov=gu.focal2fov, fov2focal=gu.fov2focal, has_torch=True)
    if hasattr(gu, "getProjectionMatrixCenterShift"):  # principal-point patch present
        out["getProjectionMatrixCenterShift"] = lambda *a: gu.getProjectionMatrixCenterShift(*a).numpy()
    return out


def rnb_loader():
    """rnb_neus2/dataloaders/rnb_loader.py: RnbDataLoader (needs cv2)."""
    root = REPOS / "RNb-NeuS2"
    if not (root / "rnb_neus2").is_dir():
        return None
    sys.path.insert(0, str(root))
    try:
        from rnb_neus2.dataloaders.rnb_loader import RnbDataLoader  # type: ignore
        return RnbDataLoader
    except Exception:
        return None
    finally:
        sys.path.remove(str(root))


def supernormal_load_K_Rt_from_P():
    """models/dataset_loader.py: load_K_Rt_from_P (module imports torch, pyexr, ...)."""
    root = REPOS / "SuperNormal"
    sys.path.insert(0, str(root))
    try:
        mod = _load(root / "models" / "dataset_loader.py", "supernormal_dataset_loader")
        return mod.load_K_Rt_from_P
    except Exception:
        return None
    finally:
        sys.path.remove(str(root))


def neuralangelo_camera():
    """projects/nerf/utils/camera.py (needs torch): get_center_and_ray, img2cam, cam2world, Pose."""
    root = REPOS / "neuralangelo"
    sys.path.insert(0, str(root))
    try:
        from projects.nerf.utils import camera  # type: ignore
        return camera
    except Exception:
        return None
    finally:
        sys.path.remove(str(root))


def unimsps_resize_with_padding():
    """Uni-MS-PS utils.py resize_with_padding (the module imports torch and the network: None without them)."""
    root = REPOS / "Uni-MS-PS"
    if not (root / "utils.py").is_file():
        return None
    sys.path.insert(0, str(root))
    try:
        return _load(root / "utils.py", "unimsps_utils").resize_with_padding
    except Exception:
        return None
    finally:
        sys.path.remove(str(root))
