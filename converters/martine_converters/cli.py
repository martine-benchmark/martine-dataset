# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""`martine-convert`: published Martine data (AliceVision sfm.json) -> inputs of the reconstruction methods of the
paper.

Subcommands (see converters/README.md):

  mvs            MVS images + sfm.json            -> colmap | colmap-mvs | pgsr | meshroom | realityscan | neus2 |
                                                      neuralangelo
  ps             MVPS multi-light images          -> unimsps | sdmunips | lino inputs
  mvps           PS normals (+ reflectance)       -> rnbneus2 | supernormal
  check-normals  PS normals                       -> GT-free check of their camera-axes convention
  subsets        MVS + MVPS sfm.json              -> paired view subsets of N views (also --n-views)
  list           formats and pixel-centre conventions

The cameras are read from ONE AliceVision sfm.json; only the pixel-centre offset of the target method is added
(0 or 0.5 px, independent of the resolution). --data/--object/--tier/--modality only locate that file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from .conventions import describe
from .export import ConversionError, Options, TriOptions, prepare_out
from .formats import MVPS_FORMATS, MVS_FORMATS, PS_METHODS, write_mvs
from .sfm import SfmError, _natural_key, locate_sfm, read_sfm, select_units


def _source(p: argparse.ArgumentParser, modality: str) -> None:
    g = p.add_argument_group("input (give --sfm, or --data/--object/--tier)")
    g.add_argument("--sfm", type=Path, help="AliceVision sfm.json (its folder holds images/ and masks/)")
    g.add_argument("--data", type=Path, help="root of the published dataset")
    g.add_argument("--object", help="object folder name, e.g. rock")
    g.add_argument("--tier", help="tier folder, e.g. d1, d2, d4, d8 (file location only)")
    g.add_argument("--modality", default=modality, choices=("mvs", "mvps"), help=f"default {modality}")
    g.add_argument("--masks", type=Path, help="masks folder (default: <sfm folder>/masks)")
    s = g.add_mutually_exclusive_group()
    s.add_argument("--views", help="subset: names, indices or slices, e.g. '000,010' or '0:84:4'")
    s.add_argument("--n-views", type=int, metavar="N",
                   help="subset of N views: k-means on the MVS camera centres, MVPS poses paired to the nearest "
                        "selected MVS view (see `martine-convert subsets`)")
    g.add_argument("--frame", choices=("gt", "raw"),
                   help="input world frame: Meshroom defaults to raw so its OBJ exports in GT; other formats default to gt")


def _common(p: argparse.ArgumentParser, no_images: bool = True, copy_mode: bool = True,
            sphere: bool = True) -> None:
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--overwrite", action="store_true",
                   help="empty and reuse a previous output folder (one holding martine_conversion.json)")
    p.add_argument("--workers", type=int, default=4, help="parallel image conversions")
    if no_images:
        p.add_argument("--no-images", action="store_true", help="cameras (and points) only")
    if copy_mode:
        p.add_argument("--copy-mode", default="copy", choices=("copy", "symlink", "hardlink"),
                       help="how EXR images are placed when kept as EXR (meshroom, realityscan)")
    if sphere:
        p.add_argument("--sphere-margin", type=float, default=0.05,
                       help="relative margin of the bounding sphere (neural methods)")


def _locate(a) -> Path:
    if a.sfm is None:
        if not (a.data and a.object and a.tier):
            raise SfmError("give --sfm, or --data, --object and --tier")
        a.sfm = locate_sfm(a.data, a.object, a.tier, a.modality)
    return a.sfm


def _scene(a):
    frame = a.frame
    if frame is None:
        frame = "raw" if a.cmd == "mvs" and a.format == "meshroom" else "gt"
    views = a.views
    if a.n_views is not None:
        from .subsets import subset_names

        views = ",".join(subset_names(_locate(a), a.n_views, obj=a.object))
    return select_units(read_sfm(_locate(a), frame=frame, masks_dir=a.masks), views)


def _options(a, **kw) -> Options:
    tri = TriOptions(backend=getattr(a, "tri_backend", "auto"), colmap=getattr(a, "colmap", "colmap"),
                     gpu=getattr(a, "tri_gpu", False), max_image_size=getattr(a, "tri_max_image_size", 3200),
                     threads=getattr(a, "tri_threads", -1), matcher=getattr(a, "tri_matcher", "exhaustive"),
                     min_track=getattr(a, "tri_min_track", 3), erode_px=getattr(a, "tri_erode", 3),
                     keep_work=getattr(a, "tri_keep_work", False),
                     peak_threshold=getattr(a, "tri_peak_threshold", None),
                     domain_size_pooling=getattr(a, "tri_dsp", False))
    return Options(images=not getattr(a, "no_images", False), copy_mode=getattr(a, "copy_mode", "copy"),
                   workers=a.workers, overwrite=a.overwrite, sphere_margin=getattr(a, "sphere_margin", 0.05),
                   tri=tri, **kw)


def cmd_mvs(a) -> dict:
    scene = _scene(a)
    if scene.modality == "mvs" and a.lighting is not None:
        raise ConversionError("--lighting only applies to an MVPS sfm.json")
    if scene.modality == "mvps" and a.lighting is None:
        a.lighting = scene.units[0].image_keys[0]
        print(f"MVPS sfm.json: using {a.lighting} of every pose (--lighting to change)")
    opts = _options(a, tonemap=a.tonemap, background=a.background, lighting=a.lighting,
                    pixel_centre=a.pixel_centre, points=a.points, image_format=a.image_format,
                    extra={"rc_prior": a.rc_prior})
    prepare_out(a.out, a.overwrite)
    return write_mvs(scene, a.out, a.format, opts)


def cmd_ps(a) -> dict:
    from .formats.ps import write_ps

    scene = _scene(a)
    dilate = a.mask_dilate
    if a.mask_dilate_d1 is not None:
        from .formats.ps import tier_dilation

        tier = re.fullmatch(r"d(\d+)", a.tier or "")
        if not tier:
            raise ConversionError("--mask-dilate-d1 needs --tier dN (the downscale factor of the tier)")
        dilate = tier_dilation(a.mask_dilate_d1, int(tier.group(1)))
    opts = _options(a, extra={"mask_dilate": dilate, "mask_dilate_d1": a.mask_dilate_d1, "ps_scale": a.scale})
    prepare_out(a.out, a.overwrite)
    return write_ps(scene, a.out, a.method, opts)


def cmd_mvps(a) -> dict:
    from .formats.mvps import write_mvps

    scene = _scene(a)
    if a.normals is None and not a.no_images:
        raise ConversionError("--normals is required (or --no-images for cameras only)")
    opts = _options(a, pixel_centre=a.pixel_centre)
    prepare_out(a.out, a.overwrite)
    return write_mvps(scene, a.out, a.format, a.normals, opts, reflectance_dir=a.reflectance,
                      convention=a.normal_convention, check=not a.no_normal_check)


def cmd_check_normals(a) -> dict:
    from .export import load_mask_checked
    from .formats.mvps import find_map
    from .normals import check_convention, read_normal_map

    scene = _scene(a)
    res = {}
    for u in scene.units:
        res[u.name] = check_convention(read_normal_map(find_map(a.normals, u.name, "normal")), load_mask_checked(u))
        r = res[u.name]
        print(f"{u.name}: {r['convention']:<7} facing {r['facing_fraction']:.3f}  scores {r['scores']}")
    return res


def cmd_subsets(a) -> dict:
    from . import subsets as sb

    if a.regenerate:
        if not a.data:
            raise SfmError("--regenerate needs --data (the published dataset)")
        table = sb.build_table(a.data)
        a.regenerate.parent.mkdir(parents=True, exist_ok=True)
        a.regenerate.write_text(sb.dump_table(table))
        print(f"wrote {a.regenerate}: {len(table['objects'])} objects, N = {table['n_views']}")
        return {}
    if not (a.object and a.n_views):
        raise SfmError("give --object and --n-views (or --regenerate)")
    entry = None if a.recompute else sb.table_entry(a.object, a.n_views)
    if entry is not None:
        res, source = {k: entry[k] for k in ("mvs", "mvps", "distance_mm")}, "precomputed"
    else:
        if not a.data:
            raise SfmError(f"{sb.no_table(a.object, a.n_views)}: give --data")
        mvs, mvps = (sb.first_sfm(a.data, a.object, m, a.tier) for m in ("mvs", "mvps"))
        res, source = sb.paired_subsets(mvs, mvps, a.n_views), "computed"
    d = np.array(res["distance_mm"])
    print(f"{a.object}: {a.n_views} views, {source} (k-means on the MVS camera centres; MVPS pose nearest to each "
          "view)")
    print(f"  {'mvs':<8} {'mvps':<10} distance")
    for s, p, x in zip(res["mvs"], res["mvps"], res["distance_mm"]):
        print(f"  {s:<8} {p:<10} {x:8.2f} mm")
    print(f"paired distance: min {d.min():.2f}, median {np.median(d):.2f}, max {d.max():.2f} mm")
    print(f"MVS  --views {','.join(res['mvs'])}")
    print(f"MVPS --views {','.join(sorted(res['mvps'], key=_natural_key))}")
    out = {**res, "object": a.object, "n_views": a.n_views, "source": source}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mvs", help="MVS formats")
    _source(p, "mvs")
    _common(p)
    p.add_argument("--format", required=True, choices=MVS_FORMATS)
    p.add_argument("--lighting", help="MVPS input: LED used for every pose (e.g. 00 or lighting_00)")
    p.add_argument("--tonemap", choices=("paper", "object99", "linear8"),
                   help="tone map of the 8-bit outputs (default: paper); EXR outputs are copied untouched")
    p.add_argument("--background", choices=("keep", "black"),
                   help="pixels outside the mask in 8-bit outputs (default: per format, as in the paper)")
    p.add_argument("--pixel-centre", type=float, choices=(0.0, 0.5),
                   help="override the pixel-centre offset of the format (meant for realityscan)")
    p.add_argument("--points", default="auto", choices=("auto", "triangulate", "none"),
                   help="sparse points for colmap/colmap-mvs/pgsr (default: triangulate)")
    p.add_argument("--image-format", choices=("exr", "png8", "png16", "jpg", "tif16"), help="realityscan images")
    p.add_argument("--rc-prior", default="locked", choices=("locked", "exact", "initial"),
                   help="realityscan PosePrior / CalibrationPrior")
    g = p.add_argument_group("triangulation (COLMAP, cameras held fixed; one attempt with the images of the converted "
                                          "tier, kept whatever its number of points)")
    g.add_argument("--tri-backend", default="auto", choices=("auto", "colmap", "pycolmap"),
                   help="auto: pycolmap when installed, else the COLMAP executable")
    g.add_argument("--colmap", default="colmap", help="COLMAP executable")
    g.add_argument("--tri-gpu", action="store_true", help="GPU SIFT extraction/matching")
    g.add_argument("--tri-max-image-size", type=int, default=3200, help="SIFT max image size (COLMAP default)")
    g.add_argument("--tri-threads", type=int, default=-1)
    g.add_argument("--tri-matcher", default="exhaustive", choices=("exhaustive", "sequential"))
    g.add_argument("--tri-min-track", type=int, default=3)
    g.add_argument("--tri-erode", type=int, default=3, help="mask erosion (px) for the features")
    g.add_argument("--tri-keep-work", action="store_true", help="keep the COLMAP database and feature images")
    g.add_argument("--tri-peak-threshold", type=float,
                   help="SIFT peak threshold (COLMAP default 0.0067); not used for the paper")
    g.add_argument("--tri-dsp", action="store_true",
                   help="DSP-SIFT (domain size pooling) features; not used for the paper")
    p.set_defaults(func=cmd_mvs)

    p = sub.add_parser("ps", help="universal photometric stereo inputs")
    _source(p, "mvps")
    _common(p, no_images=False, copy_mode=False, sphere=False)
    p.add_argument("--method", default="unimsps", choices=PS_METHODS)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--mask-dilate", type=int, default=0, help="dilate the masks given to PS (px at the tier)")
    g.add_argument("--mask-dilate-d1", type=int, default=None,
                   help="dilate the masks given to PS by this many d1 pixels (round(N / k) px at tier dk, at least "
                        "1; the paper's Uni-MS-PS runs: 5)")
    p.add_argument("--scale", default="pose", choices=("pose", "image"),
                   help="linear scaling: one per pose (radiometry kept, default) or per image")
    p.set_defaults(func=cmd_ps)

    p = sub.add_parser("mvps", help="MVPS formats (after PS)")
    _source(p, "mvps")
    _common(p, copy_mode=False)
    p.add_argument("--format", required=True, choices=MVPS_FORMATS)
    p.add_argument("--normals", type=Path, help="folder of pose_KK.png|exr camera-frame normal maps")
    p.add_argument("--reflectance", type=Path, help="folder of pose_KK reflectance maps (rnbneus2 albedo)")
    p.add_argument("--normal-convention", default="opengl", choices=("opengl", "opencv"),
                   help="camera axes of the normal maps (Uni-MS-PS / SDM-UniPS: opengl)")
    p.add_argument("--no-normal-check", action="store_true", help="skip the silhouette check of the axes")
    p.add_argument("--pixel-centre", type=float, choices=(0.0, 0.5), help="override the pixel-centre offset")
    p.set_defaults(func=cmd_mvps)

    p = sub.add_parser("check-normals", help="GT-free check of the axes of PS normal maps")
    _source(p, "mvps")
    p.add_argument("--normals", type=Path, required=True)
    p.set_defaults(func=cmd_check_normals)

    p = sub.add_parser("subsets", help="paired MVS / MVPS view subsets of N views")
    p.add_argument("--object", help="object folder name, e.g. rock")
    p.add_argument("--n-views", type=int, metavar="N")
    p.add_argument("--data", type=Path, help="root of the published dataset (needed to compute a subset that is not "
                                             "precomputed, and for --regenerate)")
    p.add_argument("--tier", help="tier folder read for the camera centres (default, or when a modality lacks it: "
                                  "the first one; the subsets do not depend on it)")
    p.add_argument("--recompute", action="store_true", help="compute the subset even when it is precomputed")
    p.add_argument("--out", type=Path, help="also write the subsets to this JSON file")
    p.add_argument("--regenerate", type=Path, metavar="FILE",
                   help="compute the subsets of every object of --data for N = 5, 10, 20, 40 and write them to FILE "
                        "(the precomputed table is martine_converters/view_subsets.json)")
    p.set_defaults(func=cmd_subsets)

    p = sub.add_parser("list", help="formats and pixel-centre conventions")
    p.set_defaults(func=lambda a: print(describe()) or {})
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        res = a.func(a)
    except (ConversionError, SfmError, KeyError, ValueError, IndexError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if a.cmd in ("mvs", "ps", "mvps"):
        print(json.dumps({k: v for k, v in (res or {}).items() if k in ("run", "next", "sfm", "triangulation",
                                                                        "pixel_centre_convention")},
                         indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
