# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Registry of the output formats of the three entry points."""
from __future__ import annotations

from pathlib import Path

from ..export import ConversionError, Options
from ..sfm import Scene

MVS_FORMATS = ("colmap", "colmap-mvs", "pgsr", "meshroom", "realityscan", "neus2", "neuralangelo")
PS_METHODS = ("unimsps", "sdmunips", "lino")
MVPS_FORMATS = ("rnbneus2", "supernormal")


def write_mvs(scene: Scene, out: Path, fmt: str, opts: Options) -> dict:
    if fmt in ("colmap", "colmap-mvs", "pgsr"):
        from .colmap import write
        return write(scene, out, fmt, opts)
    if fmt == "meshroom":
        from .alicevision import write_meshroom
        return write_meshroom(scene, out, opts)
    if fmt == "realityscan":
        from .alicevision import write_realityscan
        return write_realityscan(scene, out, opts)
    if fmt == "neus2":
        from .nerf import write_neus2
        return write_neus2(scene, out, opts)
    if fmt == "neuralangelo":
        from .nerf import write_neuralangelo
        return write_neuralangelo(scene, out, opts)
    raise ConversionError(f"unknown MVS format {fmt!r} ({MVS_FORMATS})")
