# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Universal photometric stereo inputs (entry point ``ps``): one folder per MVPS pose, linear images (one scale per
pose: radiometry kept across LEDs; 8-bit for Uni-MS-PS, 16-bit for the others) and the pose mask. The method runs
with its own code (command in the manifest); the ``mvps`` entry point then reads its normal maps (camera frame, OpenGL
axes) as ``out/normals/pose_KK.(exr|png)``.

``unimsps`` (Uni-MS-PS, the paper's choice; ``inference_file.py --path_obj <pose folder>``)::

    out/inputs/pose_KK/lighting_LL.png (8-bit), mask.png    -> out/normals/pose_KK.png (16-bit)

``sdmunips`` (SDM-UniPS, ``sdm_unips/main.py --test_dir out/inputs``, images prefixed ``L``)::

    out/inputs/pose_KK.data/L_LL.png, mask.png      -> out/normals/pose_KK.(exr|png), out/reflectance/pose_KK.*

``lino`` (LINO-UniPS, ``eval.py --task_name Real --data_root out/Real``; its loader picks the "Real" branch from the
path and the images by the prefix ``L``, src/data/data_module.py:204-206)::

    out/Real/pose_KK/L_LL.png, mask.png              -> out/normals/pose_KK.(exr|png)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..export import (
    ConversionError,
    Options,
    load_linear,
    load_mask_checked,
    run_parallel,
    write_manifest,
)
from ..imageio import tonemap, write_image, write_mask
from ..sfm import Scene

METHODS = ("unimsps", "sdmunips", "lino")
_LAYOUT = {  # method -> (inputs subfolder, pose folder suffix, image name pattern)
    "unimsps": ("inputs", "", "lighting_{ll}.png"),
    "sdmunips": ("inputs", ".data", "L_{ll}.png"),
    "lino": ("Real", "", "L_{ll}.png"),
}
# Tone map (bit depth) of the images. Uni-MS-PS reads them with cv2.IMREAD_UNCHANGED and its resize_with_padding
# casts uint16 to uint8 with astype (utils.py:22-23): the low byte is kept, the images become noise and the normals
# flat. It gets 8-bit linear images, which reproduce the paper's normals (median 2.4 deg on rock d4, pilot job 123954;
# the paper fed the float EXR through its own loader). SDM-UniPS and LINO-UniPS read 16-bit PNGs (bit depth from the
# dtype: SDM-UniPS realdata.py:219-223, LINO-UniPS src/data/data_module.py:51).
_TONEMAP = {"unimsps": "linear8", "sdmunips": "linear16", "lino": "linear16"}


def tier_dilation(d1_px: int, factor: int) -> int:
    """Mask dilation at tier d<factor> equivalent to ``d1_px`` pixels at d1 (round half up, at least 1 px when
    d1_px > 0). The paper dilated the d1 mask by 5 px before downscaling it (dataset_temp/launch_unimsps_all.sh
    DILATE=5): 1 px at d4, whereas 5 px at d4 is 20 px at d1."""
    if d1_px <= 0:
        return 0
    return max(1, int(np.floor(d1_px / factor + 0.5)))


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _check_lino_path(out: Path) -> None:
    """LINO-UniPS chooses its data branch by substring of the object path (data_module.py:180-206): the path must
    contain "Real" and none of the benchmark names tested before it."""
    p = str(Path(out).resolve())
    clash = [k for k in ("DiLiGenT", "DIR_pms", "LUCES") if k in p]
    if clash:
        raise ConversionError(f"LINO-UniPS would read {p} as the {clash[0]} benchmark (its loader tests substrings "
                              "of the path): choose an output folder whose path does not contain "
                              "DiLiGenT, DIR_pms or LUCES")


def _run_hint(method: str, n_images: int) -> str:
    """Command of the method on these inputs (the converters write inputs only)."""
    if method == "unimsps":
        return ("Uni-MS-PS (https://github.com/Clement-Hardy/Uni-MS-PS), for each pose: python inference_file.py "
                "--path_obj <out>/inputs/pose_KK --folder_save <out>/normals --nb_img -1 --cuda")
    if method == "sdmunips":
        return ("SDM-UniPS (https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023): python sdm_unips/main.py "
                f"--session_name <session> --test_dir <out>/inputs --checkpoint <ckpt> --scalable --max_image_num "
                f"{n_images} --target normal_and_brdf; copy <session>/results/pose_KK.data/normal.(exr|png) to "
                "<out>/normals/pose_KK.(exr|png) (baseColor to <out>/reflectance/)")
    return ("LINO-UniPS (https://github.com/houyuanchen111/LINO_UniPS; EXR output and --ckpt_path in the fork "
            f"https://github.com/bbrument/LINO_UniPS): python eval.py --task_name Real --data_root <out>/Real "
            f"--num_images {n_images} [--ckpt_path <lino.pth>]; copy output/<date>/Real/results/<n>/pose_KK/"
            "nml_predict.(exr|png) to <out>/normals/pose_KK.(exr|png)")


def write_ps(scene: Scene, out: Path, method: str, opts: Options) -> dict:
    if scene.modality != "mvps":
        raise ConversionError("the ps entry point needs an MVPS sfm.json (several lightings per pose)")
    if method not in METHODS:
        raise ConversionError(f"unknown PS method {method!r} ({METHODS})")
    out = Path(out)
    if method == "lino":
        _check_lino_path(out)
    out.mkdir(parents=True, exist_ok=True)
    dilate = int(opts.extra.get("mask_dilate", 0))
    scale_mode = opts.extra.get("ps_scale", "pose")
    sub, suffix, pattern = _LAYOUT[method]

    def folder(u):
        return out / sub / f"{u.name}{suffix}"

    def peak(im, m):
        return float(np.max(im[m])) if m.any() else float(np.max(im))

    def one(u):
        # two passes (max, then write) so that only one image of the pose is in memory at a time
        m = _dilate(load_mask_checked(u), dilate)
        top = max(peak(load_linear(p, u.camera), m) for _, p in u.images) if scale_mode == "pose" else None
        dropped = []
        for k, p in u.images:
            im = load_linear(p, u.camera)
            s = 1.0 / max(top if top is not None else peak(im, m), 1e-12)
            name = pattern.format(ll=k.split("_")[-1])
            img = tonemap(im, _TONEMAP[method], scale=s)
            # Uni-MS-PS divides each image by its mean inside the mask (utils.py:236-240): an 8-bit image that is 0
            # there gives a NaN (empty) normal map (whisk, metal: 21/252 poses with one scale per pose). It carries
            # no shading and Uni-MS-PS takes any number of images: it is left out.
            if method == "unimsps" and m.any() and not img[m].any():
                dropped.append(name)
                continue
            write_image(folder(u) / name, img)
        if len(dropped) == len(u.images):
            raise ConversionError(f"{u.name}: every image is black inside the mask")
        write_mask(folder(u) / "mask.png", m)
        return len(u.images) - len(dropped), dropped

    res = run_parallel(one, list(scene.units), max(1, min(opts.workers, 4)))
    counts = [n for n, _ in res]
    dropped = {u.name: d for u, (_, d) in zip(scene.units, res) if d}
    extra = {"method": method, "n_poses": len(scene.units), "images_per_pose": sorted(set(counts)),
             "scale": scale_mode, "bit_depth": 8 if _TONEMAP[method] == "linear8" else 16,
             "mask_dilate_px": dilate, "dropped_images": dropped,
             "outputs": "normals/pose_KK.(exr|png): camera-frame normals, OpenGL axes (x right, y up, z to the "
                        "camera)",
             "run": _run_hint(method, max(counts)),
             "next": "martine-convert mvps --sfm <same sfm.json> --normals <out>/normals --format "
                     "rnbneus2|supernormal --out <dir>"}
    if opts.extra.get("mask_dilate_d1") is not None:
        extra["mask_dilate_d1_px"] = int(opts.extra["mask_dilate_d1"])
    write_manifest(out, scene, f"ps-{method}", 0.0, opts, extra)
    return extra
