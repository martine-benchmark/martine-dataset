# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""COLMAP-format writers: ``colmap`` (sparse convention), ``colmap-mvs`` (dense convention) and ``pgsr``.

Layouts::

    colmap      out/images/<name>.png  out/masks/<name>.png.png  out/sparse/0/{cameras,images,points3D}.txt
    colmap-mvs  same layout, dense-pipeline camera (commands of the paper's run in the manifest)
    pgsr        out/images/<name>.png (RGBA, alpha = mask)  out/sparse/{cameras,images,points3D}.txt   (PGSR reads
                <source>/sparse)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..camera import rotmat_to_qvec
from ..colmap_model import ColmapCamera, ColmapImage, ColmapPoint, write_model
from ..conventions import pixel_offset
from ..export import (
    ConversionError,
    Options,
    export_ldr,
    export_mask,
    load_linear,
    load_mask_checked,
    remove_tree,
    run_parallel,
    source_image,
    write_manifest,
)
from ..imageio import tonemap, write_image, write_mask
from ..sfm import Scene
from ..triangulate import FeatureView, Triangulation, tier_name, triangulate, view_coverage

DEFAULTS = {  # tone map, background, points
    "colmap": ("paper", "black", "triangulate"),
    "colmap-mvs": ("paper", "black", "triangulate"),
    "pgsr": ("paper", "black", "triangulate"),
}


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def run_triangulation(scene: Scene, opts: Options, work: Path) -> Triangulation:
    """Known-pose triangulation of ``scene`` (the paper's strict protocol).

    One attempt with the images of the converted tier only and the configured SIFT (COLMAP's defaults unless
    ``--tri-peak-threshold``/``--tri-dsp`` are given): no retry at another tier or with another SIFT, and the result
    is kept whatever its number of points. An empty sparse model is written as such (with a warning); the method
    then fails, as in the paper's tables."""
    t = opts.tri
    if len(scene.units) < t.min_track:
        raise ConversionError(f"{len(scene.units)} views cannot hold a track of min_track={t.min_track} views "
                              "(--tri-min-track, or --points none)")
    tri = _triangulate_once(scene, opts, work)
    cov = view_coverage(tri, scene)
    tri.stats.update(cov, source_tier=tier_name(scene.sfm_path), peak_threshold=t.peak_threshold)
    if cov["views_without_points"]:
        print(f"warning: sparse triangulation gave {cov['n_points']} points, {cov['views_without_points']} of "
              f"{len(scene.units)} views without points (kept as is)", file=sys.stderr)
    return tri


def _triangulate_once(scene: Scene, opts: Options, work: Path) -> Triangulation:
    """Feature images (object 99.5 % tone map, eroded masks) + known-pose triangulation of ``scene`` itself."""
    img_dir, mask_dir = work / "images", work / "masks"

    def prep(u):
        img = load_linear(source_image(u, opts.lighting), u.camera)
        m = load_mask_checked(u)
        write_image(img_dir / f"{u.name}.png", tonemap(img, "object99", mask=m))
        write_mask(mask_dir / f"{u.name}.png.png", _erode(m, opts.tri.erode_px))

    run_parallel(prep, list(scene.units), opts.workers)
    views = [FeatureView(f"{u.name}.png", u.camera) for u in scene.units]
    return triangulate(views, img_dir, mask_dir, work / "colmap", opts.tri)


def write(scene: Scene, out: Path, fmt: str, opts: Options) -> dict:
    if fmt not in DEFAULTS:
        raise ConversionError(f"not a COLMAP format: {fmt}")
    delta = pixel_offset(fmt, opts.pixel_centre)
    tm, bg, pts_default = DEFAULTS[fmt]
    tm = opts.tonemap or tm
    bg = opts.background or bg
    points_mode = pts_default if opts.points == "auto" else opts.points
    out = Path(out)
    sparse = out / ("sparse" if fmt == "pgsr" else "sparse/0")

    tri: Optional[Triangulation] = None
    if points_mode == "triangulate":
        work = out / "_triangulation"
        tri = run_triangulation(scene, opts, work)
        if not opts.tri.keep_work:
            remove_tree(work)
    elif points_mode != "none":
        raise ConversionError(f"unknown --points {points_mode!r}")

    # cameras: one per distinct AliceVision intrinsic
    cam_ids: Dict[str, int] = {}
    cameras: Dict[int, ColmapCamera] = {}
    images: Dict[int, ColmapImage] = {}
    shift = pixel_offset("colmap") - delta  # keypoints are in the sparse convention
    for k, u in enumerate(scene.units, start=1):
        c = u.camera
        if u.intrinsic_id not in cam_ids:
            cid = len(cam_ids) + 1
            cam_ids[u.intrinsic_id] = cid
            cameras[cid] = ColmapCamera(cid, "PINHOLE", c.width, c.height, (c.fx, c.fy, c.cx + delta, c.cy + delta))
        name = f"{u.name}.png"
        im = ColmapImage(k, rotmat_to_qvec(c.R), c.t, cam_ids[u.intrinsic_id], name)
        if tri is not None:
            xy, ids = tri.observations[name]
            im.xys, im.point3D_ids = xy - shift, ids
        images[k] = im
    points: Dict[int, ColmapPoint] = tri.points if tri is not None else {}
    write_model(sparse, cameras, images, points)

    if opts.images:
        def one(u):
            # PGSR: RGBA with alpha = mask as in the paper's runs; its render.py zeroes the depth outside the mask
            # before the TSDF fusion (PGSR scene/cameras.py process_image, render.py)
            export_ldr(u, source_image(u, opts.lighting), out / "images" / f"{u.name}.png", tm, bg,
                       alpha=(fmt == "pgsr"))
            if fmt != "pgsr":
                export_mask(u, out / "masks" / f"{u.name}.png.png")

        run_parallel(one, list(scene.units), opts.workers)

    extra = {"pixel_centre_convention": f"{fmt}: cx = cx_AliceVision + {delta}",
             "layout": {"images": "images/", "sparse": str(sparse.relative_to(out)),
                        "masks": None if fmt == "pgsr" else "masks/<image name>.png"},
             "tonemap": tm, "background": bg, "points": points_mode,
             "triangulation": tri.stats if tri is not None else None}
    if fmt == "colmap-mvs":
        extra["run"] = ("colmap image_undistorter --image_path images --input_path sparse/0 --output_path dense "
                        "--output_type COLMAP; colmap patch_match_stereo --workspace_path dense --workspace_format "
                        "COLMAP --PatchMatchStereo.geom_consistency true; colmap stereo_fusion --workspace_path dense "
                        "--workspace_format COLMAP --input_type geometric --StereoFusion.mask_path masks "
                        "--output_path dense/fused.ply; colmap poisson_mesher --input_path dense/fused.ply "
                        "--output_path dense/meshed-poisson.ply --PoissonMeshing.trim 7  (in <out>)")
    if fmt == "pgsr":
        extra["run"] = ("python train.py -s <out> -m <model> -r 1  (principal-point-aware PGSR; official PGSR "
                        "ignores cx, cy)")
    write_manifest(out, scene, fmt, delta, opts, extra)
    return extra
