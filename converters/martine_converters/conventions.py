# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Pixel-centre convention of every target format: the single source of truth.

AliceVision (the published ``sfm.json``) puts the centre of the top-left pixel at (0, 0): pixel index ``i`` has its
centre at image coordinate ``i``. A method whose pixel ``i`` has its centre at ``i + 0.5`` needs the principal point
``cx + 0.5``. The offset does NOT depend on the resolution: it is a property of the method's code only.

References point to the method commits listed in ``converters/README.md`` (Method versions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Convention:
    """``offset`` is added to the AliceVision principal point (pixels), ``verified`` tells whether it was checked in
    the method's code (False = documented assumption, switchable on the command line)."""

    offset: float
    method: str
    evidence: str
    verified: bool = True


CONVENTIONS: Dict[str, Convention] = {
    "alicevision": Convention(0.0, "AliceVision / Meshroom (input)",
                              "jsonIO.cpp:274-281, IntrinsicScaleOffset.cpp:36 (AliceVision v3.3.0); "
                              "measured on the published data"),
    "colmap": Convention(0.5, "COLMAP sparse (SfM, triangulation, 3DGS-family readers)",
                         "colmap@3.12.6 src/colmap/feature/sift.cc:239-240, 397-398 (keypoints + 0.5)"),
    "colmap-mvs": Convention(0.0, "COLMAP dense (patch_match_stereo, stereo_fusion)",
                             "colmap@3.12.6 src/colmap/mvs/patch_match_cuda.cu:108,195,490-491; "
                             "src/colmap/mvs/fusion.cc:426-429,509-511; src/colmap/image/undistortion.cc:785"),
    "pgsr": Convention(0.5, "PGSR (principal-point-aware fork github.com/bbrument/PGSR)",
                       "PGSR utils/graphics_utils.py:157-175 (getProjectionMatrixCenterShift) + "
                       "submodules/diff-plane-rasterization/cuda_rasterizer/auxiliary.h:41-44 (ndc2Pix)"),
    "meshroom": Convention(0.0, "Meshroom / AliceVision", "native AliceVision convention"),
    "realityscan": Convention(0.5, "RealityScan / RealityCapture (XMP)",
                              "not documented by Epic; continuous image plane as in nerfstudio "
                              "realitycapture_utils.py and Metashape; used for the paper",
                              verified=False),
    "neus2": Convention(0.5, "NeuS2 (instant-ngp testbed)",
                        "NeuS2 src/testbed_nerf.cu:1216-1228,1295,1348-1351; src/nerf_loader.cu:680-685; "
                        "include/neural-graphics-primitives/common_device.cuh:591-593"),
    "neuralangelo": Convention(0.5, "Neuralangelo",
                               "neuralangelo projects/nerf/utils/camera.py:297-298 (arange(W) + 0.5)"),
    "rnbneus2": Convention(0.5, "RNb-NeuS2 (testbed of NeuS2)",
                           "RNb-NeuS2 src/testbed_nerf.cu:1286-1291; src/nerf_loader.cu:680-685"),
    "supernormal": Convention(0.0, "SuperNormal",
                              "SuperNormal models/dataset_loader.py:121-122,167-168,184-185 "
                              "(linspace(0, W-1, W), upstream 09e2615)"),
}


def pixel_offset(fmt: str, override: Optional[float] = None) -> float:
    """Offset (pixels) added to the AliceVision principal point for ``fmt``; ``override`` replaces it (only allowed
    to be 0 or 0.5)."""
    if fmt not in CONVENTIONS:
        raise KeyError(f"unknown format {fmt!r}; known: {sorted(CONVENTIONS)}")
    if override is None:
        return CONVENTIONS[fmt].offset
    if override not in (0.0, 0.5):
        raise ValueError(f"pixel-centre override must be 0 or 0.5, got {override}")
    return float(override)


def describe() -> str:
    lines = [f"{'format':<13} {'offset':>6}  method: evidence"]
    for name, c in CONVENTIONS.items():
        flag = "" if c.verified else "  [closed source; used for the paper, switchable]"
        lines.append(f"{name:<13} {c.offset:>+6.1f}  {c.method}: {c.evidence}{flag}")
    return "\n".join(lines)
