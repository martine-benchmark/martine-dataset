# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Meshroom / AliceVision (``meshroom``) and RealityScan / RealityCapture (``realityscan``) writers.

``meshroom``::

    out/sfm.json            AliceVision SfMData, absolute image paths, poses in the chosen frame (CLI default raw:
                            AliceVision exports the OBJ in W times its SfM world, W=diag(1,-1,-1))
    out/images/<name>.exr   (copy / symlink / hardlink of the source EXR)
    out/masks/<name>.png    (PrepareDenseScene --masksFolders, looked up by image stem)

    The sfm.json carries triangulated landmarks (``structure``) by default: AliceVision's DepthMap picks the neighbour
    cameras and the depth range from the landmarks observed in each view (MultiViewParams.cpp:519-560,
    SgmDepthList.cpp:59), the published sfm.json has none.

``realityscan``::

    out/images/<name>.<ext>            image (EXR by default, as the paper's runs)
    out/images/<name>.xmp              xcr camera (locked pose and calibration)
    out/images/<name>.<ext>.mask.png   RealityScan mask layer
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

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
from ..imageio import place_file, tonemap, write_image
from ..sfm import Scene


def _fmt(x: float) -> str:
    return format(float(x), ".17g")


# --- Meshroom ------------------------------------------------------------------------------------------------------
def write_meshroom(scene: Scene, out: Path, opts: Options) -> dict:
    delta = pixel_offset("meshroom", opts.pixel_centre)
    if delta != 0.0:
        raise ConversionError("meshroom is the native AliceVision convention: no pixel-centre override")
    out = Path(out)
    data = json.loads(Path(scene.sfm_path).read_text())
    keep_views = {vid for u in scene.units for vid in u.view_ids}
    keep_poses = {u.pose_id for u in scene.units}
    by_view = {}
    for u in scene.units:
        for vid, (key, path) in zip(u.view_ids, u.images):
            by_view[vid] = (u, key, path)
    # one image per pose for MVS methods
    chosen = {}
    for u in scene.units:
        src = source_image(u, opts.lighting)
        chosen[u.pose_id] = (u, src)
    views = []
    for v in data.get("views", []):
        vid = str(v["viewId"])
        if vid not in keep_views:
            continue
        u, key, path = by_view[vid]
        if path != chosen[u.pose_id][1]:
            continue  # other LEDs of an MVPS pose
        nv = copy.deepcopy(v)
        dst = out / "images" / f"{u.name}{path.suffix}"
        nv["path"] = str(dst.resolve())
        views.append(nv)
    poses = []
    for p in data.get("poses", []):
        pid = str(p["poseId"])
        if pid not in keep_poses:
            continue
        u = chosen[pid][0]
        np_ = copy.deepcopy(p)
        cam = u.camera  # already in the requested frame
        np_["pose"]["transform"]["rotation"] = [_fmt(x) for x in cam.R.T.reshape(-1)]  # camera-to-world, row-major
        np_["pose"]["transform"]["center"] = [_fmt(x) for x in cam.C]
        poses.append(np_)
    used_intr = {str(v["intrinsicId"]) for v in views}
    data["views"] = views
    data["poses"] = poses
    data["intrinsics"] = [i for i in data.get("intrinsics", []) if str(i["intrinsicId"]) in used_intr]
    for k in ("structure", "featuresFolders", "matchesFolders"):
        data.pop(k, None)
    points_mode = "triangulate" if opts.points == "auto" else opts.points
    tri = None
    if points_mode == "triangulate":
        from .colmap import run_triangulation

        work = out / "_triangulation"
        tri = run_triangulation(scene, opts, work)
        if not opts.tri.keep_work:
            remove_tree(work)
        data["structure"] = _landmarks(scene, tri, {u.pose_id: vid for u in scene.units
                                                    for vid, (_, path) in zip(u.view_ids, u.images)
                                                    if path == chosen[u.pose_id][1]})
    elif points_mode != "none":
        raise ConversionError(f"unknown --points {points_mode!r}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sfm.json").write_text(json.dumps(data, indent=4) + "\n")

    if opts.images:
        def one(u):
            src = chosen[u.pose_id][1]
            place_file(src, out / "images" / f"{u.name}{src.suffix}", opts.copy_mode)
            export_mask(u, out / "masks" / f"{u.name}.png")

        run_parallel(one, list(scene.units), opts.workers)
    extra = {"frame_note": f"poses and landmarks in {scene.frame} frame; Meshroom OBJ exports in "
                           "diag(1,-1,-1) times this frame (raw input yields GT OBJ)",
             "points": points_mode, "triangulation": tri.stats if tri is not None else None,
             "run": "Meshroom: start from this sfm.json (PrepareDenseScene --masksFolders masks, DepthMap "
                    "--downscale 1: the resolution is the tier of the input sfm.json)"
                    + ("" if tri is not None else "; WITHOUT landmarks DepthMap has no neighbour cameras / depth "
                                                  "range: triangulate in Meshroom first")}
    write_manifest(out, scene, "meshroom", delta, opts, extra)
    return extra


def _landmarks(scene: Scene, tri, view_of_pose: dict) -> list:
    """AliceVision ``structure`` from the triangulation (jsonIO.cpp saveLandmark): X in the file's world,
    observations keyed by viewId with pixel coordinates in the AliceVision convention (COLMAP keypoint - 0.5)."""
    from ..conventions import pixel_offset as _off

    shift = _off("colmap") - _off("meshroom")
    units = list(scene.units)
    out = []
    for pid in sorted(tri.points):
        p = tri.points[pid]
        obs = []
        for img_id, k in p.track:
            u = units[img_id - 1]
            xy = tri.observations[f"{u.name}.png"][0][k] - shift
            obs.append({"observationId": view_of_pose[u.pose_id], "featureId": str(k),
                        "x": [_fmt(xy[0]), _fmt(xy[1])], "scale": "1"})
        out.append({"landmarkId": str(pid - 1), "descType": "sift", "color": [str(int(c)) for c in p.rgb],
                    "X": [_fmt(x) for x in p.xyz], "observations": obs})
    return out


# --- RealityScan ---------------------------------------------------------------------------------------------------
XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
    xcr:Version="3"
    xcr:PosePrior="{pose_prior}"
    xcr:Coordinates="absolute"
    xcr:DistortionModel="none"
    xcr:FocalLength35mm="{f35}"
    xcr:Skew="0"
    xcr:AspectRatio="{aspect}"
    xcr:PrincipalPointU="{ppu}"
    xcr:PrincipalPointV="{ppv}"
    xcr:CalibrationPrior="{calib_prior}"
    xcr:CalibrationGroup="-1"
    xcr:DistortionGroup="-1"
    xcr:InTexturing="1"
    xcr:InMeshing="1">
   <xcr:Rotation>{rotation}</xcr:Rotation>
   <xcr:Position>{position}</xcr:Position>
   <xcr:DistortionCoeficients>0 0 0 0 0 0</xcr:DistortionCoeficients>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def rc_parameters(cam, delta: float) -> dict:
    """XMP numbers of one camera: rotation world-to-camera (row-major), position, 35 mm focal length and principal
    point normalised by max(W, H) relative to the image centre, in the convention ``cx_rc = cx_AV + delta``."""
    s = float(max(cam.width, cam.height))
    return {"rotation": cam.R.reshape(-1), "position": cam.C, "f35": cam.fx * 36.0 / s,
            "aspect": cam.fy / cam.fx,
            "ppu": (cam.cx + delta - cam.width / 2.0) / s, "ppv": (cam.cy + delta - cam.height / 2.0) / s}


def rc_to_pixels(width: int, height: int, f35: float, ppu: float, ppv: float, aspect: float = 1.0):
    """Inverse of :func:`rc_parameters` (fx, fy, cx_rc, cy_rc)."""
    s = float(max(width, height))
    fx = f35 * s / 36.0
    return fx, fx * aspect, ppu * s + width / 2.0, ppv * s + height / 2.0


def write_realityscan(scene: Scene, out: Path, opts: Options) -> dict:
    delta = pixel_offset("realityscan", opts.pixel_centre)
    out = Path(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ext = {"exr": ".exr", "png8": ".png", "png16": ".png", "jpg": ".jpg", "tif16": ".tif"}
    fmt_img = opts.image_format or "exr"
    if fmt_img not in ext:
        raise ConversionError(f"unknown --image-format {fmt_img!r} for realityscan ({sorted(ext)})")
    lock = opts.extra.get("rc_prior", "locked")
    scale16 = None
    if opts.images and fmt_img in ("png16", "tif16"):
        # one linear scale for the whole scene (consistent exposure across views, no clipping on the object)
        def peak(u):
            img = load_linear(source_image(u, opts.lighting), u.camera)
            return float(img[load_mask_checked(u)].max())

        scale16 = 1.0 / max(max(run_parallel(peak, list(scene.units), opts.workers)), 1e-12)

    def one(u):
        cam = u.camera
        src = source_image(u, opts.lighting)
        fname = f"{u.name}{ext[fmt_img]}"
        p = rc_parameters(cam, delta)
        (img_dir / f"{u.name}.xmp").write_text(XMP.format(
            pose_prior=lock, calib_prior=lock, f35=_fmt(p["f35"]), aspect=_fmt(p["aspect"]), ppu=_fmt(p["ppu"]),
            ppv=_fmt(p["ppv"]), rotation=" ".join(_fmt(x) for x in p["rotation"]),
            position=" ".join(_fmt(x) for x in p["position"])))
        if not opts.images:
            return
        if fmt_img == "exr":
            place_file(src, img_dir / fname, opts.copy_mode)
        elif fmt_img in ("png8", "jpg"):
            export_ldr(u, src, img_dir / fname, opts.tonemap or "paper", opts.background or "keep")
        else:
            write_image(img_dir / fname, tonemap(load_linear(src, cam), "linear16", scale=scale16))
        export_mask(u, img_dir / f"{fname}.mask.png")

    run_parallel(one, list(scene.units), opts.workers)
    extra = {"pixel_centre_convention": f"realityscan: PrincipalPointU = (cx_AV + {delta} - W/2) / max(W, H) "
                                        "(used for the paper; switch with --pixel-centre)",
             "image_format": fmt_img, "prior": lock, "linear16_scale": scale16,
             "run": "RealityScan: import the images folder (XMPs are read automatically), align with the locked "
                    "priors, then reconstruct"}
    write_manifest(out, scene, "realityscan", delta, opts, extra)
    return extra
