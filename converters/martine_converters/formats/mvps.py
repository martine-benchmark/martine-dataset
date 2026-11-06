# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""MVPS writers (entry point ``mvps``, after photometric stereo): ``rnbneus2`` and ``supernormal``.

Input: one camera-frame normal map per pose (``<normals>/pose_KK.png|exr``, default OpenGL axes as written by
Uni-MS-PS / SDM-UniPS), optional reflectance (``<reflectance>/pose_KK.*``), the MVPS ``sfm.json`` and masks.

``rnbneus2`` (RNb-NeuS2 ``run_pipeline.py --input out``, IDR folder)::

    out/normal/NNN.png   16-bit, camera frame, OpenGL axes (the testbed flips y and z itself)
    out/mask/NNN.png
    out/albedo/NNN.png   16-bit reflectance (only with --reflectance)
    out/cameras.npz      world_mat_i = K [R | t] (cx + 0.5, world = GT frame), scale_mat_i = normalised -> world

``supernormal`` (SuperNormal ``exp_runner.py --conf out/supernormal.conf``)::

    out/normal_world_space/NNN.exr  float32, world (GT) frame, R G B = x y z, 0 outside the mask
    out/mask/NNN.png
    out/cameras_sphere.npz          world_mat_i (cx + 0), scale_mat_i
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..conventions import pixel_offset
from ..export import ConversionError, Options, load_mask_checked, run_parallel, write_manifest
from ..imageio import read_image, write_exr_rgb, write_image, write_mask
from ..normals import camera_to_world, check_convention, encode16, from_opencv, read_normal_map, to_opencv, unit
from ..sfm import Scene
from .nerf import scene_sphere

FORMATS = ("rnbneus2", "supernormal")


def find_map(folder: Path, name: str, what: str) -> Path:
    for ext in (".exr", ".png", ".tif", ".tiff"):  # float maps first (8-bit PNG loses ~0.7 deg)
        p = Path(folder) / f"{name}{ext}"
        if p.is_file():
            return p
    raise ConversionError(f"no {what} map for {name} in {folder} (expected {name}.exr or .png)")


def load_normals_cv(path: Path, cam, convention: str, mask: np.ndarray) -> np.ndarray:
    n = read_normal_map(path)
    if n.shape[:2] != (cam.height, cam.width):
        raise ConversionError(f"{path}: normal map {n.shape[1]}x{n.shape[0]} != camera {cam.width}x{cam.height} "
                              "(run PS on the images of this same sfm.json)")
    n_cv, ok = unit(to_opencv(n, convention), mask)
    return n_cv


def write_mvps(scene: Scene, out: Path, fmt: str, normals_dir: Path, opts: Options,
               reflectance_dir: Optional[Path] = None, convention: str = "opengl", check: bool = True) -> dict:
    if fmt not in FORMATS:
        raise ConversionError(f"unknown MVPS format {fmt!r} ({FORMATS})")
    delta = pixel_offset(fmt, opts.pixel_centre)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    units = list(scene.units)
    digits = max(3, len(str(len(units) - 1)))
    sphere = scene_sphere(scene, opts)
    S = sphere.scale_mat()

    checks = {}
    if check and opts.images:  # GT-free sanity check of the normal axes on a few poses
        for u in units[:: max(1, len(units) // 4)][:4]:
            n = read_normal_map(find_map(normals_dir, u.name, "normal"))
            try:
                checks[u.name] = check_convention(n, load_mask_checked(u))
            except ValueError as exc:
                checks[u.name] = {"error": str(exc)}
        found = {c.get("convention") for c in checks.values() if "convention" in c}
        if found and found != {convention}:
            raise ConversionError(f"normal maps look like {sorted(found)} while --normal-convention is "
                                  f"{convention!r} (scores: {json.dumps(checks, default=str)}); "
                                  "fix the convention or pass --no-normal-check")

    cams = {}
    for i, u in enumerate(units):
        cam = u.camera
        P = np.eye(4)
        P[:3, :4] = cam.projection(delta)
        cams[f"world_mat_{i}"] = P
        cams[f"scale_mat_{i}"] = S
    np.savez(out / ("cameras.npz" if fmt == "rnbneus2" else "cameras_sphere.npz"), **cams)

    def one(iu):
        i, u = iu
        stem = f"{i:0{digits}d}"
        mask = load_mask_checked(u)
        n_cv = load_normals_cv(find_map(normals_dir, u.name, "normal"), u.camera, convention, mask)
        write_mask(out / "mask" / f"{stem}.png", mask)
        if fmt == "rnbneus2":
            n_gl = from_opencv(n_cv, "opengl")
            enc = encode16(n_gl)
            enc[~mask] = 0
            write_image(out / "normal" / f"{stem}.png", enc)
            if reflectance_dir is not None:
                a = read_image(find_map(reflectance_dir, u.name, "reflectance"))
                if a.shape[:2] != (u.camera.height, u.camera.width):
                    raise ConversionError(f"{u.name}: reflectance size != camera size")
                a = a[..., :3]
                if a.dtype == np.uint8:
                    a = a.astype(np.float32) / 255.0
                elif a.dtype == np.uint16:
                    a = a.astype(np.float32) / 65535.0
                a16 = np.round(np.clip(a, 0, 1) * 65535).astype(np.uint16)
                a16[~mask] = 0
                write_image(out / "albedo" / f"{stem}.png", a16)
        else:
            n_w = camera_to_world(n_cv, u.camera)
            n_w[~mask] = 0
            write_exr_rgb(out / "normal_world_space" / f"{stem}.exr", n_w)
        return stem

    if opts.images:
        stems = run_parallel(one, list(enumerate(units)), max(1, min(opts.workers, 4)))
    else:
        stems = [f"{i:0{digits}d}" for i in range(len(units))]
    (out / "views.json").write_text(json.dumps({s: u.name for s, u in zip(stems, units)}, indent=1) + "\n")
    (out / "normalization.json").write_text(json.dumps(sphere.to_dict(), indent=2) + "\n")
    if fmt == "supernormal":
        _supernormal_conf(out)
    extra = {"normal_input_convention": convention, "normal_check": checks, "normalization": sphere.to_dict(),
             "reflectance": str(reflectance_dir) if reflectance_dir else None,
             "run": ("python run_pipeline.py --input <out> --testbed build/testbed --output <res> [--has-albedo] "
                     "[--scaling-mode none]" if fmt == "rnbneus2" else
                     "python exp_runner.py --conf <out>/supernormal.conf --obj_name <name>")}
    write_manifest(out, scene, fmt, delta, opts, extra)
    return extra


def _supernormal_conf(out: Path) -> None:
    """Configuration of the paper's SuperNormal runs (dataset_temp/run_supernormal.py), data paths filled in, with
    ``otype = HashGrid`` added: without it tiny-cuda-nn falls back to a OneBlob encoding
    (third_parties/tiny-cuda-nn/src/encoding.cu:133) and ignores the hash-grid parameters, which the paper's
    configuration did (SuperNormal's own configs, e.g. config/own_objects.conf, set it)."""
    (out / "supernormal.conf").write_text(f"""general {{
    dataset_class = models.dataset_loader.Dataset
    renderer_class = models.renderer.NeuSRenderer
    base_exp_dir = {(out / 'exp').resolve()}
    recording = [./]
}}
dataset {{
    data_dir = {out.resolve()}
    normal_dir = normal_world_space
    cameras_name = cameras_sphere.npz
    exclude_views = []
    upsample_factor = 1
}}
train {{
    learning_rate = 5e-4
    learning_rate_alpha = 0.05
    end_iter = 100000
    increase_bindwidth_every = 2000
    gradient_method = dfd
    batch_size = 2048
    patch_size = 3
    warm_up_end = 500
    use_white_bkgd = False
    loss_type = l2
    normal_weight = 1
    eikonal_weight = 1
    mask_weight = 1
}}
val {{
    save_freq = 10000
    val_normal_freq = 100000
    val_normal_resolution_level = 2
    gradient_method = dfd
    val_mesh_freq = 100000
    val_mesh_res = 1024
    report_freq = 100
    eval_metric_freq = 100000
}}
model {{
    sdf_network {{
        d_out = 1
        d_in = 3
        d_hidden = 64
        n_layers = 1
        skip_in = [-1]
        bias = 0.8
        geometric_init = True
        weight_norm = True
        input_concat = True
    }}
    variance_network {{
        init_val = 0.5
    }}
    ray_marching {{
        start_step_size = 1e-2
        end_step_size = 1e-3
        occ_threshold = 0.1
        occ_sigmoid_k = 80.0
        occ_resolution = 128
        occ_update_freq = 8
    }}
    encoding {{
        otype = HashGrid
        n_levels = 16
        n_features_per_level = 2
        log2_hashmap_size = 19
        base_resolution = 16
        per_level_scale = 1.447
    }}
}}
""")
