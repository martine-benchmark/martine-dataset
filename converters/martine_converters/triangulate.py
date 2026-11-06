# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Known-pose triangulation of sparse 3D points with COLMAP (executable) or pycolmap.

Recipe of the dataset staging (led_reorder/scripts/triangulate_sparse.py): features on the object only (eroded
masks), SIFT, exhaustive matching, ``point_triangulator`` with the cameras and poses held fixed (no intrinsics
refinement), check that they did not move. The cameras given to COLMAP are in COLMAP's own (sparse) pixel
convention, i.e. AliceVision principal point + 0.5 (``conventions.CONVENTIONS['colmap']``): SIFT keypoints are
stored at ``vl_feat + 0.5``.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import Camera, rotmat_to_qvec
from .colmap_model import ColmapCamera, ColmapImage, ColmapPoint, read_model, write_model
from .conventions import pixel_offset
from .export import ConversionError, TriOptions


@dataclass(frozen=True)
class FeatureView:
    name: str  # image file name inside the work folder, e.g. "000.png"
    camera: Camera  # AliceVision-convention camera (GT or raw frame, the triangulated points follow it)


@dataclass
class Triangulation:
    points: Dict[int, ColmapPoint]  # ids 1..n, tracks use the image ids 1..N of ``views`` order
    observations: Dict[str, Tuple[np.ndarray, np.ndarray]]  # name -> (xy in COLMAP convention, point ids)
    stats: dict


def _colmap_env(exe: str) -> dict:
    env = dict(os.environ)
    path = shutil.which(exe) or exe
    lib = Path(path).resolve().parent.parent / "lib"
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def _run(cmd: List[str], env: dict, log: Path) -> None:
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        tail = "\n".join(Path(log).read_text().splitlines()[-15:])
        raise ConversionError(f"{cmd[0]} {cmd[1]} failed (rc {r.returncode}), log {log}:\n{tail}")


def _option_prefixes(exe: str, env: dict) -> Tuple[str, str]:
    """COLMAP <= 3.12: SiftExtraction/SiftMatching.{use_gpu,...}; >= 3.13/4.x: FeatureExtraction/FeatureMatching."""
    r = subprocess.run([exe, "feature_extractor", "-h"], env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr  # COLMAP prints its help through glog (stderr) in some versions
    if "FeatureExtraction.use_gpu" in out:
        return "FeatureExtraction", "FeatureMatching"
    if "SiftExtraction.use_gpu" in out:
        return "SiftExtraction", "SiftMatching"
    raise ConversionError(f"cannot read the options of {exe} feature_extractor (is COLMAP installed?)")


def colmap_camera_params(cam: Camera) -> Tuple[float, float, float, float]:
    d = pixel_offset("colmap")
    return cam.fx, cam.fy, cam.cx + d, cam.cy + d


def _db_images(db: Path) -> Dict[str, Tuple[int, int]]:
    con = sqlite3.connect(str(db))
    try:
        return {n: (int(i), int(c)) for i, n, c in con.execute("SELECT image_id, name, camera_id FROM images")}
    finally:
        con.close()


def _db_set_camera(db: Path, cam_id: int, width: int, height: int, params: Sequence[float]) -> None:
    """Force a database camera to PINHOLE (model id 1) with the given parameters (keypoints do not depend on it)."""
    con = sqlite3.connect(str(db))
    try:
        con.execute("UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                    (1, int(width), int(height), np.asarray(params, np.float64).tobytes(), int(cam_id)))
        con.commit()
    finally:
        con.close()


def _db_counts(db: Path) -> dict:
    con = sqlite3.connect(str(db))
    try:
        kp = [r[0] for r in con.execute("SELECT rows FROM keypoints")]
        tv = con.execute("SELECT COUNT(*), COALESCE(SUM(rows), 0) FROM two_view_geometries WHERE rows > 0").fetchone()
        return {"keypoints_median": float(np.median(kp)) if kp else 0.0, "verified_pairs": int(tv[0]),
                "inlier_matches": int(tv[1])}
    finally:
        con.close()


def _input_model(views: Sequence[FeatureView], name2db: Dict[str, Tuple[int, int]], path: Path) -> None:
    cams: Dict[int, ColmapCamera] = {}
    imgs: Dict[int, ColmapImage] = {}
    for v in views:
        if v.name not in name2db:
            raise ConversionError(f"image {v.name} missing from the COLMAP database")
        iid, cid = name2db[v.name]
        p = colmap_camera_params(v.camera)
        if cid in cams and not np.allclose(cams[cid].params, p, rtol=0, atol=0):
            raise ConversionError(f"database camera {cid} shared by images with different intrinsics")
        cams[cid] = ColmapCamera(cid, "PINHOLE", v.camera.width, v.camera.height, p)
        imgs[iid] = ColmapImage(iid, rotmat_to_qvec(v.camera.R), v.camera.t, cid, v.name)
    write_model(path, cams, imgs, {})


def _check_fixed(model_in: Path, model_out: Path) -> Tuple[float, float]:
    cin, iin, _ = read_model(model_in)
    cout, iout, _ = read_model(model_out)
    for k, c in cin.items():
        if k not in cout or not np.allclose(cout[k].params, c.params, rtol=1e-12, atol=0):
            raise ConversionError(f"COLMAP changed camera {k}: {c.params} -> {cout.get(k)}")
    a = {i.name: i for i in iin.values()}
    b = {i.name: i for i in iout.values()}
    if set(a) != set(b):
        raise ConversionError(f"COLMAP dropped images: {sorted(set(a) ^ set(b))[:5]}")
    dq = max(min(np.abs(a[n].qvec - b[n].qvec).max(), np.abs(a[n].qvec + b[n].qvec).max()) for n in a)
    dt = max(np.abs(a[n].tvec - b[n].tvec).max() for n in a)
    if dq > 1e-9 or dt > 1e-6 * max(1.0, max(np.abs(a[n].tvec).max() for n in a)):
        raise ConversionError(f"COLMAP moved the cameras (quaternion {dq:.2e}, translation {dt:.2e})")
    return float(dq), float(dt)


def _collect(views: Sequence[FeatureView], model_out: Path, min_track: int) -> Triangulation:
    _, imgs, pts = read_model(model_out)
    by_id = {i.id: i for i in imgs.values()}
    order = {v.name: k + 1 for k, v in enumerate(views)}  # our image ids
    keep: Dict[int, ColmapPoint] = {}
    obs: Dict[str, List[Tuple[float, float, int]]] = {v.name: [] for v in views}
    dropped = 0
    for pid in sorted(pts):
        p = pts[pid]
        names = [by_id[i].name for i, _ in p.track]
        if len(set(names)) != len(names) or len(names) < min_track:
            dropped += 1
            continue
        new_id = len(keep) + 1
        track = []
        for (iid, k), name in zip(p.track, names):
            x, y = by_id[iid].xys[k]
            obs[name].append((float(x), float(y), new_id))
            track.append((order[name], len(obs[name]) - 1))
        keep[new_id] = ColmapPoint(new_id, p.xyz, p.rgb, p.error, track)
    observations = {n: (np.array([(x, y) for x, y, _ in o], float).reshape(-1, 2),
                        np.array([i for _, _, i in o], np.int64)) for n, o in obs.items()}
    errs = np.array([p.error for p in keep.values()]) if keep else np.zeros(0)
    tl = np.array([len(p.track) for p in keep.values()]) if keep else np.zeros(0)
    stats = {"n_points_colmap": len(pts), "n_points": len(keep), "dropped": dropped, "min_track": min_track,
             "reproj_error_mean_px": float(errs.mean()) if errs.size else float("nan"),
             "track_length_median": float(np.median(tl)) if tl.size else float("nan")}
    return Triangulation(keep, observations, stats)


def triangulate(views: Sequence[FeatureView], image_dir: Path, mask_dir: Optional[Path], work: Path,
                opts: TriOptions) -> Triangulation:
    """Triangulate with fixed cameras. ``image_dir`` holds the 8-bit feature images named ``view.name``,
    ``mask_dir`` the COLMAP masks ``<name>.png`` (white = keep features)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    backend = opts.backend
    if backend == "auto":
        try:
            import pycolmap  # noqa: F401
            backend = "pycolmap" if hasattr(pycolmap, "FeatureExtractionOptions") else "colmap"
        except ImportError:
            backend = "colmap"
    if backend == "pycolmap":
        return _triangulate_pycolmap(views, image_dir, mask_dir, work, opts)
    if backend != "colmap":
        raise ConversionError(f"unknown triangulation backend {opts.backend!r}")
    exe = opts.colmap
    if shutil.which(exe) is None and not Path(exe).is_file():
        raise ConversionError(f"COLMAP executable {exe!r} not found and pycolmap not installed (uv sync --extra "
                              "tri, or --colmap /path/to/colmap, or --points none)")
    env = _colmap_env(exe)
    log = work / "colmap.log"
    fe, fm = _option_prefixes(exe, env)
    db = work / "database.db"
    db.unlink(missing_ok=True)
    cams = [v.camera for v in views]
    single = all(c.intrinsics_equal(cams[0]) for c in cams)
    cmd = [exe, "feature_extractor", "--database_path", db, "--image_path", image_dir,
           "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", int(single),
           f"--{fe}.use_gpu", int(opts.gpu), f"--{fe}.max_image_size", int(opts.max_image_size),
           f"--{fe}.num_threads", int(opts.threads)]
    if single:
        cmd += ["--ImageReader.camera_params", ",".join(format(p, ".17g") for p in colmap_camera_params(cams[0]))]
    if opts.peak_threshold is not None:
        cmd += ["--SiftExtraction.peak_threshold", format(float(opts.peak_threshold), ".17g")]
    if opts.domain_size_pooling:
        cmd += ["--SiftExtraction.domain_size_pooling", 1]
    if mask_dir is not None:
        cmd += ["--ImageReader.mask_path", mask_dir]
    _run(cmd, env, log)
    name2db = _db_images(db)
    for v in views:  # make the database cameras exactly the converted ones
        iid, cid = name2db[v.name]
        _db_set_camera(db, cid, v.camera.width, v.camera.height, colmap_camera_params(v.camera))
    if opts.matcher == "exhaustive":
        _run([exe, "exhaustive_matcher", "--database_path", db, f"--{fm}.use_gpu", int(opts.gpu),
              f"--{fm}.num_threads", int(opts.threads)], env, log)
    elif opts.matcher == "sequential":
        _run([exe, "sequential_matcher", "--database_path", db, f"--{fm}.use_gpu", int(opts.gpu),
              f"--{fm}.num_threads", int(opts.threads)], env, log)
    else:
        raise ConversionError(f"unknown matcher {opts.matcher!r}")
    mi, mo, mt = work / "model_in", work / "model_out", work / "model_txt"
    for d in (mi, mo, mt):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    _input_model(views, name2db, mi)
    _run([exe, "point_triangulator", "--database_path", db, "--image_path", image_dir, "--input_path", mi,
          "--output_path", mo, "--clear_points", 1, "--refine_intrinsics", 0,
          "--Mapper.num_threads", int(opts.threads)], env, log)
    _run([exe, "model_converter", "--input_path", mo, "--output_path", mt, "--output_type", "TXT"], env, log)
    dq, dt = _check_fixed(mi, mt)
    tri = _collect(views, mt, opts.min_track)
    tri.stats.update(_db_counts(db), backend="colmap", colmap_version=_version(exe, env), pose_check_dq=dq,
                     pose_check_dt=dt)
    return tri


def _version(exe: str, env: dict) -> str:
    r = subprocess.run([exe, "-h"], env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"COLMAP (\S+)", out)
    return m.group(1) if m else "unknown"


def _triangulate_pycolmap(views: Sequence[FeatureView], image_dir: Path, mask_dir: Optional[Path], work: Path,
                          opts: TriOptions) -> Triangulation:
    try:
        import pycolmap  # type: ignore
    except ImportError as exc:
        raise ConversionError("pycolmap is not installed (uv sync --extra tri) - or use --tri-backend colmap")\
            from exc
    if not hasattr(pycolmap, "FeatureExtractionOptions"):
        raise ConversionError(f"pycolmap {pycolmap.__version__} is too old (FeatureExtractionOptions API needed, "
                              "tested with 4.2.1) - or use --tri-backend colmap")
    db = work / "database.db"
    db.unlink(missing_ok=True)
    cams = [v.camera for v in views]
    single = all(c.intrinsics_equal(cams[0]) for c in cams)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    if single:
        reader.camera_params = ",".join(format(p, ".17g") for p in colmap_camera_params(cams[0]))
    if mask_dir is not None:
        reader.mask_path = str(mask_dir)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = int(opts.max_image_size)
    extraction.num_threads = int(opts.threads)
    extraction.use_gpu = bool(opts.gpu)
    if opts.peak_threshold is not None:
        extraction.sift.peak_threshold = float(opts.peak_threshold)
    if opts.domain_size_pooling:
        extraction.sift.domain_size_pooling = True
    mode = pycolmap.CameraMode.SINGLE if single else pycolmap.CameraMode.PER_IMAGE
    pycolmap.extract_features(str(db), str(image_dir), image_names=[v.name for v in views], camera_mode=mode,
                              reader_options=reader, extraction_options=extraction)
    name2db = _db_images(db)
    for v in views:
        iid, cid = name2db[v.name]
        _db_set_camera(db, cid, v.camera.width, v.camera.height, colmap_camera_params(v.camera))
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = int(opts.threads)
    matching.use_gpu = bool(opts.gpu)
    if opts.matcher == "exhaustive":
        pycolmap.match_exhaustive(str(db), matching_options=matching)
    else:
        pycolmap.match_sequential(str(db), matching_options=matching)
    mi, mo, mt = work / "model_in", work / "model_out", work / "model_txt"
    for d in (mi, mo, mt):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    _input_model(views, name2db, mi)
    rec = pycolmap.Reconstruction(str(mi))
    pipeline = pycolmap.IncrementalPipelineOptions()
    pipeline.num_threads = int(opts.threads)
    rec = pycolmap.triangulate_points(rec, str(db), str(image_dir), str(mo), clear_points=True,
                                      options=pipeline, refine_intrinsics=False)
    rec.write_text(str(mt))
    dq, dt = _check_fixed(mi, mt)
    tri = _collect(views, mt, opts.min_track)
    tri.stats.update(_db_counts(db), backend="pycolmap", pycolmap_version=pycolmap.__version__,
                     pose_check_dq=dq, pose_check_dt=dt)
    return tri


def tier_name(sfm_path: Path) -> str:
    """Folder name of the tier of ``sfm_path`` (``d8`` for ``<object>/mvs/d8/sfm.json``)."""
    return Path(sfm_path).resolve().parent.name


def view_coverage(tri: Triangulation, scene) -> dict:
    """Points per view of ``scene`` (a :class:`..sfm.Scene`), recorded in the manifest: COLMAP patch match and
    AliceVision DepthMap take the depth range and the neighbour views of each image from the points it observes."""
    counts = [len(tri.observations.get(f"{u.name}.png", (np.zeros((0, 2)),))[0]) for u in scene.units]
    return {"n_points": len(tri.points), "min_view_points": int(min(counts)) if counts else 0,
            "views_without_points": int(sum(c == 0 for c in counts))}
