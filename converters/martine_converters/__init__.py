# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Conversion tools from the published Martine data (AliceVision sfm.json + EXR + PNG masks) to the input
formats of the reconstruction methods evaluated in the paper.

The camera maths is resolution-agnostic: a converter reads one AliceVision sfm.json (centre of the top-left pixel
at (0, 0), whatever its resolution) and only adds the pixel-centre offset of the target method
(see :mod:`martine_converters.conventions`).
"""

__version__ = "1.0.0"
