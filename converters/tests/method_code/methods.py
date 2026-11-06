# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""How each method reads the converted files and where it puts a 3D point, in pixel indices.

Each loader returns ``{name: project(X) -> (uv_index (N, 2), depth (N,))}`` where ``name`` is the view/pose name of
the conversion (image stem). References are to the commits of converters/README.md (Method versions).
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

Proj = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _pinhole(K: np.ndarray, R: np.ndarray, t: np.ndarray, index_shift: float) -> Proj:
    def f(X):
        xc = X @ R.T + t
        uvw = xc @ K.T
        return uvw[:, :2] / uvw[:, 2:] - index_shift, xc[:, 2]
    return f


# --- COLMAP text model ------------------------------------------------------------------------------------------
def qvec2rotmat(q):
    # COLMAP scripts/python/read_write_model.py qvec2rotmat (BSD); same formula in PGSR scene/colmap_loader.py:43-54
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_text(sparse: Path):
    cams = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            x = line.split()
            cams[int(x[0])] = (x[1], int(x[2]), int(x[3]), np.array(x[4:], float))
    imgs = {}
    rows = [ln for ln in (sparse / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    for head in rows[0::2]:
        x = head.split()
        imgs[x[9]] = (np.array(x[1:5], float), np.array(x[5:8], float), int(x[8]))
    return cams, imgs


def colmap(sparse: Path, dense: bool) -> Dict[str, Proj]:
    """COLMAP ``Camera::ImgFromCam`` for PINHOLE: u = fx x/z + cx (src/colmap/sensor/models.h). Pixel index i has
    its centre at u = i + 0.5 for the sparse pipeline (sift.cc:239-240 adds 0.5 to VLFeat keypoints) and at u = i
    for the dense pipeline (patch_match_cuda.cu:108,195 cast the ray of pixel (col, row) through K^-1 (col, row, 1);
    fusion.cc:426-429 compares projections with integer col/row)."""
    cams, imgs = read_colmap_text(sparse)
    out = {}
    for name, (q, t, cid) in imgs.items():
        model, w, h, p = cams[cid]
        assert model == "PINHOLE"
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
        out[Path(name).stem] = _pinhole(K, qvec2rotmat(q), t, 0.0 if dense else 0.5)
    return out


# --- PGSR -------------------------------------------------------------------------------------------------------
def pgsr(source: Path, real=None) -> Dict[str, Proj]:
    """PGSR: ``readColmapSceneInfo`` reads ``<source>/sparse/{cameras,images}.txt`` (scene/dataset_readers.py:156-168);
    ``readColmapCameras``: R = qvec2rotmat(q)^T, T = t, FovX = focal2fov(fx, W), cx = params[2] (principal-point patch,
    :104-116); ``Camera``: Fx = fov2focal(FovX, W) (scene/cameras.py:74-75), full projection
    ``getWorld2View2(R, T)^T @ getProjectionMatrixCenterShift(znear, zfar, Cx, Cy, Fx, Fy, W, H)^T``
    (graphics_utils.py:122-133, 157-175); rasteriser ``ndc2Pix(v, S) = ((v + 1) S - 1) / 2`` (auxiliary.h:41-44) gives
    the pixel index directly. ``real`` = dict of the original functions when PGSR is importable."""
    import math

    def w2v(R, t):
        Rt = np.eye(4)
        Rt[:3, :3] = R.T
        Rt[:3, 3] = t
        return Rt.astype(np.float64)

    def pmc(znear, zfar, cx, cy, fx, fy, w, h):
        top, bottom = cy / fy * znear, -(h - cy) / fy * znear
        left, right = -(w - cx) / fx * znear, cx / fx * znear
        P = np.zeros((4, 4))
        P[0, 0] = 2 * znear / (right - left)
        P[1, 1] = 2 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = 1.0
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    fns = {"qvec2rotmat": qvec2rotmat, "getWorld2View2": w2v, "getProjectionMatrixCenterShift": pmc,
           "focal2fov": lambda f, n: 2 * math.atan(n / (2 * f)), "fov2focal": lambda fov, n: n / (2 * math.tan(fov / 2))}
    if real is not None:
        fns.update({k: v for k, v in real.items() if k in fns})
    if real is not None and "read_extrinsics_text" in real:
        extr = real["read_extrinsics_text"](str(source / "sparse" / "images.txt"))
        intr = real["read_intrinsics_text"](str(source / "sparse" / "cameras.txt"))
        items = [(e.name, e.qvec, e.tvec, intr[e.camera_id].width, intr[e.camera_id].height, intr[e.camera_id].params)
                 for e in extr.values()]
    else:
        cams, imgs = read_colmap_text(source / "sparse")
        items = [(n, q, t, cams[c][1], cams[c][2], cams[c][3]) for n, (q, t, c) in imgs.items()]
    q2r, w2v, pmc = fns["qvec2rotmat"], fns["getWorld2View2"], fns["getProjectionMatrixCenterShift"]
    f2fov, fov2f = fns["focal2fov"], fns["fov2focal"]

    out = {}
    for name, q, t, w, h, p in items:
        R = np.transpose(q2r(q))
        fx = fov2f(f2fov(p[0], w), w)
        fy = fov2f(f2fov(p[1], h), h)
        WV = np.asarray(w2v(R, np.asarray(t)), dtype=np.float64).T
        PM = np.asarray(pmc(0.01, 100.0, float(p[2]), float(p[3]), fx, fy, w, h), dtype=np.float64).T
        full = WV @ PM

        def f(X, full=full, WV=WV, w=w, h=h):
            Xh = np.concatenate([X, np.ones((len(X), 1))], axis=1)
            ph = Xh @ full
            ndc = ph[:, :3] / ph[:, 3:]
            u = ((ndc[:, 0] + 1.0) * w - 1.0) * 0.5
            v = ((ndc[:, 1] + 1.0) * h - 1.0) * 0.5
            return np.stack([u, v], axis=1), (Xh @ WV)[:, 2]
        out[Path(name).stem] = f
    return out


# --- AliceVision / Meshroom (our output sfm.json, world = converter frame) ---------------------------------------
def meshroom(sfm_path: Path) -> Dict[str, Proj]:
    """AliceVision reading of the written sfm.json (rotation row-major = camera-to-world, centre, cx = w/2 + pp),
    without any world correction: the written file's world is the converter frame. Pixel index = coordinate."""
    d = json.loads(Path(sfm_path).read_text())
    intr = {str(i["intrinsicId"]): i for i in d["intrinsics"]}
    poses = {str(p["poseId"]): p["pose"]["transform"] for p in d["poses"]}
    out = {}
    for v in d["views"]:
        i = intr[str(v["intrinsicId"])]
        w = int(i["width"])
        f = float(i["focalLength"]) * w / float(i["sensorWidth"])
        K = np.array([[f, 0, w / 2 + float(i["principalPoint"][0])],
                      [0, f, int(i["height"]) / 2 + float(i["principalPoint"][1])], [0, 0, 1]])
        tr = poses[str(v["poseId"])]
        R = np.array([float(x) for x in tr["rotation"]]).reshape(3, 3).T
        C = np.array([float(x) for x in tr["center"]])
        out[Path(v["path"]).stem] = _pinhole(K, R, -R @ C, 0.0)
    return out


# --- RealityScan XMP --------------------------------------------------------------------------------------------
def realityscan(images: Path, delta: float, sizes: Dict[str, Tuple[int, int]]) -> Dict[str, Proj]:
    """XMP (xcr 1.1): Rotation world-to-camera row-major, Position = centre, f_px = FocalLength35mm max(W,H)/36,
    c = PrincipalPoint max(W,H) + W/2 (nerfstudio process_data/realitycapture_utils.py:81-85). ``delta`` is the
    ASSUMED pixel centre of RealityScan (index = coordinate - delta)."""
    ns = "{http://www.capturingreality.com/ns/xcr/1.1#}"
    out = {}
    for xmp in sorted(images.glob("*.xmp")):
        desc = ET.parse(xmp).getroot().find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
        R = np.array(desc.find(ns + "Rotation").text.split(), float).reshape(3, 3)
        C = np.array(desc.find(ns + "Position").text.split(), float)
        w, h = sizes[xmp.stem]
        s = max(w, h)
        fx = float(desc.get(ns + "FocalLength35mm")) * s / 36.0
        fy = fx * float(desc.get(ns + "AspectRatio"))
        cx = float(desc.get(ns + "PrincipalPointU")) * s + w / 2.0
        cy = float(desc.get(ns + "PrincipalPointV")) * s + h / 2.0
        out[xmp.stem] = _pinhole(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), R, -R @ C, delta)
    return out


# --- NeuS2 / RNb-NeuS2 testbed (instant-ngp) ----------------------------------------------------------------------
def ngp(transforms: dict) -> Dict[str, Proj]:
    """instant-ngp testbed of NeuS2 / RNb-NeuS2 with ``from_na``:

    * per frame ``focal = intrinsic[0][0], [1][1]``, ``principal_point = intrinsic[0][2] / w, [1][2] / h``
      (src/nerf_loader.cu:680-685), ``xform = nerf_matrix_to_ngp(transform_matrix)``: columns 1, 2 negated, translation
      ``* scale + offset``, ``from_na`` negates columns 1, 2 back (include/.../nerf_loader.h:114-122);
    * training ray of the normalised image position ``xy`` (pixel = floor(xy * res), common_device.cuh:591-593):
      ``dir = R ((xy - pp) res / f, 1)`` (src/testbed_nerf.cu:1348-1351), so pixel i has its centre at
      ``xy res = i + 0.5``;
    * world <-> testbed: mesh export ``p_world = n2w_s p + n2w_t`` with ``p = (p_ngp - offset) / scale``.
    """
    w, h = transforms["w"], transforms["h"]
    scale = float(transforms.get("scale", 1.0))
    off = np.array(transforms.get("offset", [0.5, 0.5, 0.5]), float)
    n2w = np.array(transforms["n2w"], float)
    n2w_s, n2w_t = n2w[0, 0], n2w[:3, 3]
    assert transforms.get("from_na"), "conversions use from_na"
    out = {}
    for fr in transforms["frames"]:
        K = np.array(fr["intrinsic_matrix"], float)
        f = np.array([K[0, 0], K[1, 1]])
        pp = np.array([K[0, 2] / w, K[1, 2] / h])
        M = np.array(fr["transform_matrix"], float)[:3, :4].copy()
        M[:, 1] *= -1
        M[:, 2] *= -1
        M[:, 3] = M[:, 3] * scale + off
        M[:, 1] *= -1
        M[:, 2] *= -1
        R, o = M[:, :3], M[:, 3]

        def proj(X, R=R, o=o, f=f, pp=pp):
            Xn = (X - n2w_t) / n2w_s
            Xngp = Xn * scale + off
            d = (Xngp - o) @ np.linalg.inv(R).T  # camera-frame direction R^-1 (X - o)
            xy_res = np.stack([d[:, 0] / d[:, 2] * f[0] + pp[0] * w, d[:, 1] / d[:, 2] * f[1] + pp[1] * h], 1)
            return xy_res - 0.5, d[:, 2]
        out[Path(fr.get("file_path", fr.get("normal_path"))).stem] = proj
    return out


def rnbneus2(folder: Path, size: Tuple[int, int], loader=None, dtype=np.float64) -> Dict[str, Proj]:
    """RNb-NeuS2 ``run_pipeline.py --input folder``: ``RnbDataLoader`` (rnb_neus2/dataloaders/rnb_loader.py:52-108)
    decomposes ``P = world_mat_i @ scale_mat_i`` with ``cv2.decomposeProjectionMatrix`` (K normalised, c2w = R^T,
    centre) and returns ``scale_mat_0``; ``prepare_testbed_data`` (rnb_neus2/prepare.py:139-230) writes
    ``transform_matrix = c2w`` (translation rescaled by its own normalisation, identity with ``--scaling-mode none``),
    ``intrinsic_matrix = K`` and ``n2w = scale_mat_0 @ inv(scale_matrix)``; then the NeuS2 testbed model applies."""
    if loader is not None:
        data = loader(str(folder)).load()
        views = [(v["K"], v["c2w"]) for v in data["views"]]
        S = np.asarray(data["scale_mat"], float)
    else:
        import cv2

        cams = np.load(folder / "cameras.npz")
        n = max(int(k.split("_")[-1]) for k in cams.files) + 1
        views = []
        for i in range(n):
            P = (cams[f"world_mat_{i}"].astype(dtype) @ cams[f"scale_mat_{i}"].astype(dtype))[:3, :4]
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K4 = np.eye(4)
            K4[:3, :3] = K / K[2, 2]
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = (t[:3] / t[3])[:, 0]
            views.append((K4, c2w))
        S = cams["scale_mat_0"].astype(dtype)
    names = json.loads((folder / "views.json").read_text())
    stems = sorted(names)
    w, h = size
    tf = {"w": w, "h": h, "scale": 0.5, "offset": [0.5, 0.5, 0.5], "from_na": True, "n2w": np.asarray(S).tolist(),
          "frames": [{"file_path": names[s] + ".png", "transform_matrix": np.asarray(c2w).tolist(),
                      "intrinsic_matrix": np.asarray(K).tolist()} for s, (K, c2w) in zip(stems, views)]}
    return ngp(tf)


# --- Neuralangelo -----------------------------------------------------------------------------------------------
def neuralangelo(folder: Path) -> Dict[str, Proj]:
    """projects/neuralangelo/data.py:101-131: intr from fl_x, sk_x, cx / sk_y, fl_y, cy (shared); c2w =
    ``_gl_to_cv(transform_matrix)`` (columns 1, 2 negated); centre -= sphere_center (+ readjust 0); /= sphere_radius
    (x readjust 1); intrinsics scaled by W / raw_W (1 here). projects/nerf/utils/camera.py:297-298: pixel grid
    ``arange(W) + 0.5`` -> pixel i has its centre at u = i + 0.5."""
    m = json.loads((folder / "transforms.json").read_text())
    K = np.array([[m["fl_x"], m["sk_x"], m["cx"]], [m["sk_y"], m["fl_y"], m["cy"]], [0, 0, 1]], float)
    c, r = np.array(m["sphere_center"], float), float(m["sphere_radius"])
    out = {}
    for fr in m["frames"]:
        c2w = np.array(fr["transform_matrix"], float) * np.array([1, -1, -1, 1])
        c2w[:3, 3] = (c2w[:3, 3] - c) / r
        R = c2w[:3, :3].T
        t = -R @ c2w[:3, 3]

        def proj(X, R=R, t=t):
            return _pinhole(K, R, t, 0.5)((X - c) / r)
        out[Path(fr["file_path"]).stem] = proj
    return out


# --- SuperNormal --------------------------------------------------------------------------------------------------
def supernormal(folder: Path, load_K_Rt_from_P=None, dtype=np.float32) -> Dict[str, Proj]:
    """models/dataset_loader.py (upstream 09e2615): ``P = world_mat @ scale_mat`` (:106), ``load_K_Rt_from_P`` =
    ``cv2.decomposeProjectionMatrix`` (:14-36), rays through the integer pixel grid ``linspace(0, W - 1, W)``
    (:121-128) -> pixel index = coordinate. Normal maps ``normal_world_space/*.exr`` sorted, view index = file name
    (:61-62)."""
    import cv2

    cams = np.load(folder / "cameras_sphere.npz")
    names = json.loads((folder / "views.json").read_text())
    out = {}
    for stem in sorted(names):
        i = int(stem)
        S = cams[f"scale_mat_{i}"].astype(dtype)  # dataset_loader.py:94-98 cast to float32
        P = (cams[f"world_mat_{i}"].astype(dtype) @ S)[:3, :4]
        if load_K_Rt_from_P is not None:
            K4, C2W = load_K_Rt_from_P(None, P)
            K = np.asarray(K4)[:3, :3]
        else:
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K = K / K[2, 2]
            C2W = np.eye(4)
            C2W[:3, :3] = R.T
            C2W[:3, 3] = (t[:3] / t[3])[:, 0]
        R = np.asarray(C2W)[:3, :3].T
        tt = -R @ np.asarray(C2W)[:3, 3]
        Sinv = np.linalg.inv(S.astype(np.float64))

        def proj(X, K=K, R=R, tt=tt, Sinv=Sinv):
            Xn = X @ Sinv[:3, :3].T + Sinv[:3, 3]
            return _pinhole(K, R, tt, 0.0)(Xn)
        out[names[stem]] = proj
    return out


def stem_index(name: str) -> int:
    return int(re.findall(r"\d+", name)[-1])


# --- Uni-MS-PS image reading ---------------------------------------------------------------------------------------
def unimsps_as_read(img: np.ndarray) -> np.ndarray:
    """Uni-MS-PS utils.resize_with_padding (utils.py:20-24): a uint16 image is cast with ``astype(np.uint8)`` (the
    low byte is kept, i.e. modulo 256) before the resize; other dtypes pass unchanged."""
    return img.astype(np.uint8) if img.dtype == np.uint16 else img
