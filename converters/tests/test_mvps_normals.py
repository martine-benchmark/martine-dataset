# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""PS and MVPS entry points on a synthetic sphere: the normal frames each method expects, the GT-free convention
check, the PS input folders, and the GT-free normalisation."""
from __future__ import annotations

import json

import numpy as np
import pytest
import synthetic
from martine_converters.export import ConversionError, Options
from martine_converters.formats.mvps import write_mvps
from martine_converters.formats.ps import tier_dilation, write_ps
from martine_converters.imageio import read_exr, read_image, read_mask
from martine_converters.normalize import bounding_sphere
from martine_converters.normals import check_convention, decode, read_normal_map
from martine_converters.sfm import read_sfm

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    cams, nmaps = synthetic.build(root)
    return root, cams, nmaps


def test_reader_matches_synthetic_cameras(scene_dir):
    root, cams, _ = scene_dir
    s = read_sfm(root / "sfm.json")
    assert s.modality == "mvps" and len(s.units) == len(cams)
    for u, c in zip(s.units, cams):
        assert np.allclose(u.camera.R, c.R, atol=1e-12) and np.allclose(u.camera.C, c.C, atol=1e-9)
        assert (u.camera.cx, u.camera.cy) == (c.cx, c.cy)


def test_supernormal_world_normals(scene_dir, tmp_path):
    root, cams, nmaps = scene_dir
    s = read_sfm(root / "sfm.json")
    out = tmp_path / "sn"
    write_mvps(s, out, "supernormal", root / "normals", Options(workers=2))
    names = json.loads((out / "views.json").read_text())
    for stem, name in names.items():
        mask, n_w = nmaps[name]
        n = read_exr(out / "normal_world_space" / f"{stem}.exr")
        err = np.abs(n[mask] - n_w[mask]).max()
        assert err < 2e-4, f"{name}: {err}"
        assert np.all(n[~mask] == 0)
        assert np.array_equal(read_mask(out / "mask" / f"{stem}.png"), mask)
    conf = (out / "supernormal.conf").read_text()
    assert "otype = HashGrid" in conf  # tiny-cuda-nn falls back to OneBlob without it
    cams_npz = np.load(out / "cameras_sphere.npz")
    S = cams_npz["scale_mat_0"]
    assert S[0, 0] == S[1, 1] == S[2, 2] and np.allclose(np.linalg.norm(S[:3, 3] - synthetic.CENTER), 0, atol=3.0)


def test_rnbneus2_testbed_decoding(scene_dir, tmp_path):
    """RNb-NeuS2 testbed (src/testbed_nerf.cu:1507-1510): n = 2 v - 1, n.y *= -1, n.z *= -1 (camera, OpenCV axes),
    world = c2w rotation of the frame (decomposed from world_mat @ scale_mat by RnbDataLoader)."""
    root, cams, nmaps = scene_dir
    s = read_sfm(root / "sfm.json")
    out = tmp_path / "rnb"
    write_mvps(s, out, "rnbneus2", root / "normals", Options(workers=2))
    names = json.loads((out / "views.json").read_text())
    cams_npz = np.load(out / "cameras.npz")
    for stem, name in names.items():
        i = int(stem)
        P = (cams_npz[f"world_mat_{i}"] @ cams_npz[f"scale_mat_{i}"])[:3, :4]
        K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
        c2w_rot = R.T
        v = read_image(out / "normal" / f"{stem}.png")
        assert v.dtype == np.uint16 and v.shape[2] == 3
        n = v.astype(np.float64) / 65535 * 2 - 1
        n[..., 1] *= -1
        n[..., 2] *= -1
        n_w = n @ c2w_rot.T
        mask, ref = nmaps[name]
        assert np.abs(n_w[mask] - ref[mask]).max() < 2e-4
        assert np.allclose(K / K[2, 2], cams[i].K(0.5), atol=1e-6)  # pixel centre + 0.5 for the testbed


def test_reflectance_to_albedo(scene_dir, tmp_path):
    root, _, nmaps = scene_dir
    from martine_converters.imageio import write_image

    refl = tmp_path / "refl"
    for name, (mask, _) in nmaps.items():
        a = np.zeros(mask.shape + (3,), np.float32)
        a[mask] = [0.2, 0.5, 0.8]
        write_image(refl / f"{name}.exr", a)
    s = read_sfm(root / "sfm.json")
    out = tmp_path / "rnb_albedo"
    write_mvps(s, out, "rnbneus2", root / "normals", Options(workers=2), reflectance_dir=refl)
    a16 = read_image(out / "albedo" / "000.png")
    mask = nmaps["pose_00"][0]
    assert np.allclose(a16[mask] / 65535, [0.2, 0.5, 0.8], atol=1e-4) and np.all(a16[~mask] == 0)


def test_convention_check(scene_dir):
    root, _, nmaps = scene_dir
    mask, _ = nmaps["pose_01"]
    n = read_normal_map(root / "normals" / "pose_01.png")
    r = check_convention(n, mask)
    assert r["convention"] == "opengl" and r["facing_fraction"] > 0.99 and r["scores"]["+x-y"] > 0.9
    r = check_convention(n * np.array([1, -1, -1]), mask)
    assert r["convention"] == "opencv" and r["facing_fraction"] > 0.99
    r = check_convention(n * np.array([-1, 1, 1]), mask)
    assert r["convention"] not in ("opengl", "opencv")


def test_wrong_convention_is_refused(scene_dir, tmp_path):
    root, _, _ = scene_dir
    with pytest.raises(ConversionError, match="look like"):
        write_mvps(read_sfm(root / "sfm.json"), tmp_path / "x", "supernormal", root / "normals",
                   Options(workers=1), convention="opencv")


def test_normal_size_mismatch(scene_dir, tmp_path):
    root, _, nmaps = scene_dir
    small = tmp_path / "small"
    small.mkdir()
    for name in nmaps:
        v = read_image(root / "normals" / f"{name}.png")
        cv2.imwrite(str(small / f"{name}.png"), v[::2, ::2, ::-1])
    with pytest.raises(ConversionError, match="!= camera"):
        write_mvps(read_sfm(root / "sfm.json"), tmp_path / "y", "rnbneus2", small, Options(workers=1),
                   check=False)


@pytest.mark.parametrize("method", ["unimsps", "sdmunips", "lino"])
def test_ps_inputs(scene_dir, tmp_path, method):
    root, _, nmaps = scene_dir
    out = tmp_path / method
    write_ps(read_sfm(root / "sfm.json"), out, method, Options(workers=2))
    d = out / {"unimsps": "inputs/pose_00", "sdmunips": "inputs/pose_00.data", "lino": "Real/pose_00"}[method]
    pattern = "lighting_{:02d}.png" if method == "unimsps" else "L_{:02d}.png"
    names = sorted(p.name for p in d.iterdir())
    assert names == sorted([pattern.format(li) for li in range(3)] + ["mask.png"])
    imgs = [read_image(d / pattern.format(li)) for li in range(3)]
    # Uni-MS-PS casts 16-bit images to uint8 (low byte kept): it gets 8-bit images, the others 16-bit
    dtype, full = (np.uint8, 255.0) if method == "unimsps" else (np.uint16, 65535.0)
    assert all(i.dtype == dtype for i in imgs)
    src = [read_exr(root / "images" / "pose_00" / f"lighting_{li:02d}.exr") for li in range(3)]
    top = max(s[nmaps["pose_00"][0]].max() for s in src)
    for i, s in zip(imgs, src):  # one linear scale per pose: radiometry between LEDs kept
        assert np.abs(i / full - s / top).max() <= 0.5 / full + 1e-6
    assert np.array_equal(read_mask(d / "mask.png"), nmaps["pose_00"][0])
    assert [f.name for f in out.rglob("*.sh")] == []  # inputs only: no script that runs the method
    manifest = json.loads((out / "martine_conversion.json").read_text())
    assert manifest["images_per_pose"] == [3]  # LINO-UniPS needs --num_images


def test_unimsps_inputs_are_read_unchanged_by_its_loader(scene_dir, tmp_path):
    """Uni-MS-PS reads the PNGs with cv2.IMREAD_UNCHANGED (utils.py:202-203) and its resize_with_padding casts a uint16
    image with astype(np.uint8) (utils.py:22-23), keeping the low byte: 16-bit inputs became noise and the normals flat
    (pilot jobs 123860-123865). The written images must reach the network unchanged; 8-bit linear inputs with one scale
    per pose reproduce the paper's normals (median 2.4 deg, job 123954)."""
    import cv2
    from method_code import methods, real_code

    root, _, _ = scene_dir
    out = tmp_path / "unimsps"
    write_ps(read_sfm(root / "sfm.json"), out, "unimsps", Options(workers=1))
    real = real_code.unimsps_resize_with_padding()  # None without torch: the re-implementation is used
    files = sorted((out / "inputs" / "pose_00").glob("lighting_*.png"))
    assert files
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        assert img.max() > 0
        assert np.array_equal(methods.unimsps_as_read(img), img)
        if real is not None:  # original code: a square crop at its own size is not resized, only cast
            n = min(img.shape[:2])
            got, _ = real(img[:n, :n], (n, n))
            assert np.array_equal(got, img[:n, :n].astype(got.dtype))


def _black_led_copy(scene_dir, tmp_path, leds):
    """Copy of the synthetic scene whose pose_00 images of the given LEDs are black inside the mask."""
    import shutil

    root, _, nmaps = scene_dir
    dst = tmp_path / "scene"
    shutil.copytree(root, dst)
    m = nmaps["pose_00"][0]
    for li in leds:
        f = dst / "images" / "pose_00" / f"lighting_{li:02d}.exr"
        img = np.where(m[..., None], 0.0, read_exr(f))
        cv2.imwrite(str(f), img[..., ::-1].astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    return dst


def test_unimsps_drops_black_images(scene_dir, tmp_path):
    """whisk (metal), 26/10: with one scale per pose, some 8-bit LED images were 0 inside the mask; Uni-MS-PS divides
    each image by its masked mean (utils.py:236-240), hence an empty (NaN) normal map on 21/252 poses. Such an image
    carries no shading: it is left out of the pose (Uni-MS-PS takes any number of images). The pose scale itself is
    kept: it was the closest to the GT normals on whisk, rock and warrior_head (whisk_rerun, jobs 138183-138184)."""
    dst = _black_led_copy(scene_dir, tmp_path, [1])
    out = tmp_path / "unimsps"
    write_ps(read_sfm(dst / "sfm.json"), out, "unimsps", Options(workers=1))
    assert sorted(f.name for f in (out / "inputs" / "pose_00").glob("lighting_*.png")) == \
        ["lighting_00.png", "lighting_02.png"]
    man = json.loads((out / "martine_conversion.json").read_text())
    assert man["dropped_images"] == {"pose_00": ["lighting_01.png"]}
    assert man["images_per_pose"] == [2, 3] and man["scale"] == "pose"


def test_unimsps_refuses_a_pose_without_images(scene_dir, tmp_path):
    dst = _black_led_copy(scene_dir, tmp_path, [0, 1, 2])
    with pytest.raises(ConversionError, match="pose_00"):
        write_ps(read_sfm(dst / "sfm.json"), tmp_path / "unimsps", "unimsps", Options(workers=1))


@pytest.mark.parametrize("d1_px,factor,want", [(5, 1, 5), (5, 2, 3), (5, 4, 1), (5, 8, 1), (0, 4, 0)])
def test_mask_dilation_given_at_full_resolution(d1_px, factor, want):
    """The paper dilated the d1 mask by 5 px before downscaling it (dataset_temp/launch_unimsps_all.sh DILATE=5);
    5 px at tier d4 is 20 px at d1. Closer to the GT normals on whisk by 1.5-2 deg, neutral on rock and warrior_head
    (jobs 138183-138184)."""
    assert tier_dilation(d1_px, factor) == want


def test_lino_refuses_ambiguous_paths(scene_dir, tmp_path):
    root, _, _ = scene_dir
    with pytest.raises(ConversionError, match="LUCES"):
        write_ps(read_sfm(root / "sfm.json"), tmp_path / "LUCES_tests" / "x", "lino", Options(workers=1))


def test_bounding_sphere_synthetic(scene_dir):
    root, cams, _ = scene_dir
    s = read_sfm(root / "sfm.json")
    sph = bounding_sphere([u.camera for u in s.units], lambda i: read_mask(s.units[i].mask), margin=0.05)
    d = np.linalg.norm(sph.center - synthetic.CENTER)
    assert sph.radius >= d + synthetic.RADIUS  # encloses the object
    assert sph.radius <= 1.35 * synthetic.RADIUS  # but stays tight (visual hull of 6 views + 5 %)
    assert d < 0.1 * synthetic.RADIUS


def test_decode_png_and_exr_identity():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(5, 5, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    from martine_converters.normals import encode16
    assert np.abs(decode(encode16(n)) - n).max() < 2e-5
    assert np.array_equal(decode(n.astype(np.float32)), n.astype(np.float32))


def test_float_maps_are_preferred(scene_dir, tmp_path):
    from martine_converters.formats.mvps import find_map
    from martine_converters.imageio import write_image

    write_image(tmp_path / "pose_00.png", np.zeros((4, 4, 3), np.uint16))
    write_image(tmp_path / "pose_00.exr", np.zeros((4, 4, 3), np.float32))
    assert find_map(tmp_path, "pose_00", "normal").suffix == ".exr"


def test_overwrite_only_empties_conversion_outputs(tmp_path):
    from martine_converters.export import MANIFEST, prepare_out

    other = tmp_path / "other"
    other.mkdir()
    (other / "precious.txt").write_text("x")
    with pytest.raises(ConversionError):
        prepare_out(other, overwrite=True)
    assert (other / "precious.txt").exists()
    conv = prepare_out(tmp_path / "conv", overwrite=False)
    (conv / "old").mkdir()
    (conv / "old" / "f").write_text("x")
    with pytest.raises(ConversionError):
        prepare_out(conv, overwrite=False)
    prepare_out(conv, overwrite=True)
    assert sorted(p.name for p in conv.iterdir()) == [MANIFEST]
