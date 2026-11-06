# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Re-implementations of how each method LOADS the converted files and PROJECTS a 3D point, written from the
methods' code (file:line references in each module, commits listed in converters/README.md, Method versions).

Every ``project_*`` function returns, for each view, the projection of world (GT-frame) points expressed in
**pixel indices** (the centre of pixel (i, j) is at (i, j)), using the method's own rule for where pixel centres
are. A correct conversion therefore reproduces the AliceVision projection of the published sfm.json exactly.

No third-party code is copied here (several method licences are not MIT); when the method repository is available
(``MARTINE_METHOD_REPOS``), :mod:`real_code` imports the original functions instead.
"""
