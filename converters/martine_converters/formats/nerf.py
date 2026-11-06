# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""``neus2`` and ``neuralangelo`` writers (``transforms.json`` family).

NeuS2 (instant-ngp testbed, ``from_na`` mode)::

    out/transforms.json   per-frame ``transform_matrix`` = camera-to-world with OpenCV axes in the normalised frame
                          (``(X - c) / r``), per-frame ``intrinsic_matrix`` (cx + 0.5), ``scale`` 0.5 / ``offset``
                          0.5 (unit sphere -> [0, 1]^3), ``n2w`` = normalised -> world (applied at mesh export)
    out/images/<name>.png RGBA, alpha = mask (paper)

Neuralangelo::

    out/transforms.json   shared intrinsics (cx + 0.5), ``transform_matrix`` = camera-to-world with OpenGL axes in
                          the world (GT) frame, ``sphere_center`` / ``sphere_radius`` (normalisation done by the loader
                          and undone by extract_mesh.py), ``aabb_range``
    out/images/<name>.png RGB, background black (paper)
    out/neuralangelo.yaml config to pass to train.py (parent: projects/neuralangelo/configs/base.yaml)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..camera import CV_TO_GL
from ..conventions import pixel_offset
from ..export import ConversionError, Options, export_ldr, load_mask_checked, run_parallel, source_image, write_manifest
from ..normalize import Sphere, bounding_sphere
from ..sfm import Scene


def scene_sphere(scene: Scene, opts: Options) -> Sphere:
    units = list(scene.units)
    return bounding_sphere([u.camera for u in units], lambda i: load_mask_checked(units[i]),
                           margin=opts.sphere_margin)


def _write_normalization(out: Path, sphere: Sphere, extra: dict) -> None:
    (out / "normalization.json").write_text(json.dumps({**sphere.to_dict(), **extra}, indent=2) + "\n")


def _same_size(scene: Scene):
    sizes = {(u.camera.width, u.camera.height) for u in scene.units}
    if len(sizes) != 1:
        raise ConversionError(f"all views must have the same size, got {sorted(sizes)}")
    return sizes.pop()


def write_neus2(scene: Scene, out: Path, opts: Options) -> dict:
    delta = pixel_offset("neus2", opts.pixel_centre)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = _same_size(scene)
    sphere = scene_sphere(scene, opts)
    c, r = sphere.center, sphere.radius
    frames = []
    for u in scene.units:
        cam = u.camera
        c2w = cam.c2w()
        c2w[:3, 3] = (cam.C - c) / r
        K = np.eye(4)
        K[:3, :3] = cam.K(delta)
        frames.append({"file_path": f"images/{u.name}.png", "transform_matrix": c2w.tolist(),
                       "intrinsic_matrix": K.tolist()})
    n2w = sphere.scale_mat()
    meta = {"w": w, "h": h, "aabb_scale": 1.0, "scale": 0.5, "offset": [0.5, 0.5, 0.5], "from_na": True,
            "n2w": n2w.tolist(), "frames": frames}
    (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
    _write_normalization(out, sphere, {"neus2_ngp": "x_ngp = 0.5 * (X - center) / radius + 0.5"})
    tm = opts.tonemap or "paper"
    if opts.images:
        run_parallel(lambda u: export_ldr(u, source_image(u, opts.lighting), out / "images" / f"{u.name}.png", tm,
                                          opts.background or "keep", alpha=True), list(scene.units), opts.workers)
    extra = {"normalization": sphere.to_dict(), "tonemap": tm,
             "run": "python scripts/run.py --scene <out>/transforms.json --name <run> --network dtu.json "
                    "--n_steps 50000 --save_mesh   (mesh exported in the world frame through n2w)"}
    write_manifest(out, scene, "neus2", delta, opts, extra)
    return extra


def write_neuralangelo(scene: Scene, out: Path, opts: Options) -> dict:
    delta = pixel_offset("neuralangelo", opts.pixel_centre)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = _same_size(scene)
    cams = [u.camera for u in scene.units]
    if not all(cm.intrinsics_equal(cams[0], tol=1e-9) for cm in cams):
        raise ConversionError("Neuralangelo supports one shared intrinsics only (views have different cameras)")
    k0 = cams[0].K(delta)
    sphere = scene_sphere(scene, opts)
    c, r = sphere.center, sphere.radius
    frames = []
    for u in scene.units:
        c2w_gl = u.camera.c2w()
        c2w_gl[:3, :3] = c2w_gl[:3, :3] @ CV_TO_GL  # negate camera y and z columns (CV -> GL)
        frames.append({"file_path": f"images/{u.name}.png", "transform_matrix": c2w_gl.tolist()})
    meta = {
        "camera_angle_x": float(2 * np.arctan(w / (2 * k0[0, 0]))),
        "camera_angle_y": float(2 * np.arctan(h / (2 * k0[1, 1]))),
        "fl_x": float(k0[0, 0]), "fl_y": float(k0[1, 1]), "cx": float(k0[0, 2]), "cy": float(k0[1, 2]),
        "sk_x": 0.0, "sk_y": 0.0, "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0,
        "is_fisheye": False, "w": w, "h": h,
        "aabb_scale": float(np.exp2(np.rint(np.log2(r)))),
        "aabb_range": [[float(c[i] - r), float(c[i] + r)] for i in range(3)],
        "sphere_center": [float(x) for x in c], "sphere_radius": float(r),
        "frames": frames,
    }
    (out / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
    _write_normalization(out, sphere, {})
    cfg = [
        "_parent_: projects/neuralangelo/configs/base.yaml",
        "max_iter: 500000",
        "model:",
        "    object:",
        "        sdf:",
        "            mlp:",
        "                inside_out: false",
        "            encoding:",
        "                coarse2fine:",
        "                    init_active_level: 4",
        "    appear_embed:",
        "        enabled: false",
        "data:",
        "    type: projects.neuralangelo.data",
        f"    root: {out.resolve()}",
        f"    num_images: {len(frames)}",
        "    train:",
        f"        image_size: [{h}, {w}]",
        "        batch_size: 1",
        "    val:",
        f"        image_size: [{max(1, h // 4)}, {max(1, w // 4)}]",
        "        batch_size: 1",
        "        subset: 1",
        "    readjust:",
        "        center: [0.0, 0.0, 0.0]",
        "        scale: 1.0",
    ]
    (out / "neuralangelo.yaml").write_text("\n".join(cfg) + "\n")
    tm = opts.tonemap or "paper"
    if opts.images:
        run_parallel(lambda u: export_ldr(u, source_image(u, opts.lighting), out / "images" / f"{u.name}.png", tm,
                                          opts.background or "black"), list(scene.units), opts.workers)
    extra = {"normalization": sphere.to_dict(), "tonemap": tm,
             "run": "torchrun --nproc_per_node=1 train.py --config=<out>/neuralangelo.yaml --logdir=<logs>; "
                    "projects/neuralangelo/scripts/extract_mesh.py (mesh in the world frame through sphere_*)"}
    write_manifest(out, scene, "neuralangelo", delta, opts, extra)
    return extra
