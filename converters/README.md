# Martine converters

`martine-convert` turns the published Martine data (AliceVision `sfm.json`, EXR images, PNG masks, any tier
d1/d2/d4/d8) into the inputs of each reconstruction method of the paper, with the **pixel-centre convention of each
method handled explicitly** and tested against the method's own loader.

## Method versions

| Method | Family | Command | Pixel centre | Version used for the paper |
| :-- | :-- | :-- | :-- | :-- |
| COLMAP MVS | classical MVS | `mvs --format colmap-mvs` | +0 | [colmap/colmap](https://github.com/colmap/colmap) `3.12.6` |
| PGSR | Gaussian splatting | `mvs --format pgsr` | +0.5 | [bbrument/PGSR](https://github.com/bbrument/PGSR) `5b9ab11` (`main`: upstream `de24f1a` + principal-point patch) |
| Meshroom | photogrammetry | `mvs --format meshroom` | +0 | [Meshroom](https://github.com/alicevision/Meshroom) release 2025.1.0 (AliceVision `v3.3.0`) |
| RealityScan | photogrammetry | `mvs --format realityscan` | +0.5 | RealityScan 2.1.0, Linux (closed source) |
| NeuS2 | neural implicit | `mvs --format neus2` | +0.5 | [19reborn/NeuS2](https://github.com/19reborn/NeuS2) `8a5284c` |
| Neuralangelo | neural implicit | `mvs --format neuralangelo` | +0.5 | [NVlabs/neuralangelo](https://github.com/NVlabs/neuralangelo) `94390b6` |
| Uni-MS-PS | photometric stereo | `ps --method unimsps` | n/a | [meshroomHubWarehouse/Uni-MS-PS](https://github.com/meshroomHubWarehouse/Uni-MS-PS) `6076929` (`speedup`) |
| SDM-UniPS | photometric stereo | `ps --method sdmunips` | n/a | [meshroomHubWarehouse/SDM-UniPS](https://github.com/meshroomHubWarehouse/SDM-UniPS) `e264f07` (`feat/float32-deterministic`); not used for the paper |
| LINO-UniPS | photometric stereo | `ps --method lino` | n/a | [meshroomHubWarehouse/LINO_UniPS](https://github.com/meshroomHubWarehouse/LINO_UniPS) `c04460d` (`feat/float32-deterministic`); not used for the paper |
| RNb-NeuS2 | MVPS | `mvps --format rnbneus2` | +0.5 | [bbrument/RNb-NeuS2](https://github.com/bbrument/RNb-NeuS2) `72ee0f2` |
| SuperNormal | MVPS | `mvps --format supernormal` | +0 | [CyberAgentAILab/SuperNormal](https://github.com/CyberAgentAILab/SuperNormal) `09e2615` |

This table is the reference for the versions of the paper's runs. `mvs --format colmap` writes a plain COLMAP sparse
model (+0.5) for any other tool that reads one (3DGS family). The file:line references of the code and the tests
point to these versions.

## Quick start

```bash
uv sync                                    # from the repository root; without uv: pip install -e ".[exr,tri]"
DATA=/path/to/martine
uv run martine-convert mvs --data $DATA --object rock --tier d8 --format neus2 --out work/rock_neus2
uv run martine-convert ps --data $DATA --object rock --tier d8 --method unimsps --out work/rock_ps
# run Uni-MS-PS on each work/rock_ps/inputs/pose_KK, its normals in work/rock_ps/normals/pose_KK.png
uv run martine-convert mvps --data $DATA --object rock --tier d8 --format rnbneus2 \
    --normals work/rock_ps/normals --out work/rock_rnb
```

The converters write the inputs of a method only: each method runs with its own code. Each output folder holds
`martine_conversion.json`: source, options, offset applied, triangulation statistics, and a hint of the method's
command. `martine-convert list` prints the conventions with their code references.

## Common options

Every command reads **one** `sfm.json`: `--sfm .../mvs/d2/sfm.json`, or `--data <root> --object <name> --tier
<d1|d2|d4|d8> [--modality mvs|mvps]`. The tier only locates the file; it never enters the maths.

| Option | Effect |
| :-- | :-- |
| `--views 0:84:4` | subset of views (names, indices or slices) |
| `--n-views N` | subset of N views, paired between MVS and MVPS (see [View subsets](#view-subsets)) |
| `--no-images` | cameras and points only (needs numpy only) |
| `--frame gt\|raw` | input world frame (default: `raw` for Meshroom, `gt` otherwise; see below) |
| `--lighting LL` | with an MVPS `sfm.json`, `mvs` uses this LED for every pose (default: the first) |
| `--pixel-centre 0\|0.5` | override the method's offset |
| `--workers N`, `--overwrite` | parallel image writing; empty a previous output |

Optional extras: `exr` (OpenEXR, faster EXR I/O), `tri` (pycolmap, triangulation without a COLMAP executable) and
`subsets` (scikit-learn and scipy, to compute view subsets that are not precomputed): `uv sync --extra exr --extra
tri --extra subsets`.

## View subsets

`--n-views N` (any command, instead of `--views`) keeps N camera positions, chosen so that the MVS and MVPS subsets
of an object are comparable:

1. MVS: k-means with N clusters on the camera centres of the MVS `sfm.json` (scikit-learn `KMeans`,
   `random_state=42`, `n_init=10`); each cluster gives the camera nearest to its centroid, without duplicate.
2. MVPS: each selected MVS view is paired with the nearest MVPS pose, without duplicate (optimal assignment on the
   distance between camera centres, scipy `linear_sum_assignment`).

Only the camera centres are used, so the subsets are the same at every tier.

The subsets of the published objects for N = 5, 10, 20 and 40 are precomputed in
[`martine_converters/view_subsets.json`](martine_converters/view_subsets.json) (published view names, and the
distance between the paired camera centres). `--n-views` reads them from there, so the MVPS images alone suffice and
no extra is needed. Another N is computed: this needs the MVS `sfm.json` of the object (found in the published layout
`<root>/<object>/mvs/<tier>/sfm.json`) and the extra `subsets`. `martine-convert subsets --data <root> --regenerate
<file>` recomputes the whole file.

`martine-convert subsets` prints the pairs and the two `--views` lists (`--data` is only needed for a subset that is
not precomputed, or with `--recompute`):

```bash
uv run martine-convert subsets --object rock --n-views 5 --out work/rock_v5.json
uv run martine-convert mvs --data $DATA --object rock --tier d4 --n-views 5 --format neus2 --out work/rock_neus2_v5
```

```text
rock: 5 views, precomputed (k-means on the MVS camera centres; MVPS pose nearest to each view)
  mvs      mvps       distance
  005      pose_05        2.92 mm
  031      pose_31        8.65 mm
  038      pose_38       11.32 mm
  058      pose_58       14.58 mm
  073      pose_73       21.08 mm
paired distance: min 2.92, median 11.32, max 21.08 mm
MVS  --views 005,031,038,058,073
MVPS --views pose_05,pose_31,pose_38,pose_58,pose_73
```

`--out` writes the same lists as JSON (`mvs`, `mvps`, `distance_mm`, pair by pair).

## Conventions

- **Pixel centre.** AliceVision (the published `sfm.json`) puts the centre of the top-left pixel at (0, 0):
  `cx = W/2 + principalPoint[0]`. Methods that put it at (0.5, 0.5) need `cx + 0.5`. The offset is a property of the
  method's code, not of the resolution (`martine_converters/conventions.py`).
- **World frame.** Meshes target the frame of `gt/`: `X_gt = diag(1, −1, −1) X_raw`, where `X_raw` is the world of
  the published `sfm.json`. Cameras are written in that GT frame, except for Meshroom: its OBJ export applies the
  rotation itself, so the converter writes raw poses and the OBJ comes out in GT.
- **Normalisation** (NeuS2, Neuralangelo, RNb-NeuS2, SuperNormal): bounding sphere of the visual hull of the masks,
  enlarged by 5 %; cameras and masks only, no GT, the same at every tier (`normalization.json`).
- **Images.** 8-bit formats get the paper's tone mapping (gamma 1/2.2; auto exposure and Reinhard only above 1).
  Photometric stereo gets linear images (8-bit for Uni-MS-PS, 16-bit otherwise).
- **Sparse points (strict protocol of the paper).** COLMAP, PGSR and Meshroom get points triangulated once, with
  the images of the converted tier only (the `--views` subset), COLMAP's default SIFT and the cameras held fixed.
  There is no retry at another tier or with another SIFT, and no threshold on the number of points. A weakly
  textured object can give few points or none (vase d8); the method may then fail, as in the paper's tables. The
  manifest records `n_points`, `min_view_points` and `views_without_points`. `--tri-peak-threshold` and `--tri-dsp`
  change the SIFT explicitly; they are off by default and were not used for the paper. COLMAP: the `colmap`
  executable (`--colmap`), or pycolmap when installed (`--tri-backend`).

## Method details

<details><summary><b>COLMAP</b> (<code>colmap</code>, <code>colmap-mvs</code>)</summary>

- `colmap`: `images/<name>.png` (8-bit, black background), `masks/<name>.png.png`, `sparse/0/*.txt`. Offset +0.5:
  COLMAP stores SIFT keypoints at VLFeat + 0.5 (`src/colmap/feature/sift.cc:239-240, 397-398`). Methods derived
  from 3DGS often ignore `cx, cy`: check that yours honours them.
- `colmap-mvs`: the same layout with the dense-pipeline camera. Offset +0: `patch_match_stereo`
  casts pixel (col, row) through `K⁻¹ (col, row, 1)` (`src/colmap/mvs/patch_match_cuda.cu:108,195`), `stereo_fusion`
  compares integer indices (`src/colmap/mvs/fusion.cc:426-429, 509-511`). The stored keypoints are shifted by −0.5,
  so the model stays self-consistent.
- The paper's dense pipeline: `image_undistorter`, `patch_match_stereo` with geometric consistency, `stereo_fusion`
  with `--StereoFusion.mask_path masks`, `poisson_mesher` with `--PoissonMeshing.trim 7` (CUDA needed; full commands
  in the manifest). Patch match takes its depth range from the sparse points: keep the default `--points triangulate`.
</details>

<details><summary><b>PGSR</b> (<code>pgsr</code>)</summary>

- Produces `images/<name>.png` (RGBA, alpha = mask: PGSR's `render.py` zeroes the depth outside it) and
  `sparse/*.txt` with the triangulated points. Offset +0.5 (`getProjectionMatrixCenterShift` + `ndc2Pix`).
- Use the fork [bbrument/PGSR](https://github.com/bbrument/PGSR) (default branch `main`, the patched version): the
  official PGSR ignores `cx, cy`, about a 4 mm
  rigid bias on Martine. The fork passes the principal point to the CUDA rasteriser (rebuild
  `diff-plane-rasterization`) and stores the TSDF depth as float32; its README lists the changes.
- Run: `python train.py -s <out> -m <out>/model -r 1`, then `render.py`. Always `-r 1` on the converted tier: PGSR's
  own `-r 2/4/8` rescaling is off by up to 0.5 px, and without `-r` images wider than 1600 px are downscaled.
</details>

<details><summary><b>Meshroom / AliceVision</b> (<code>meshroom</code>)</summary>

- Produces `sfm.json` (absolute image paths, raw-frame poses, **landmarks**), `images/<name>.exr`, `masks/<name>.png`.
  Offset +0.
- DepthMap takes its neighbour views and depth range from the landmarks (`depthMap/SgmDepthList.cpp:59`), and the
  published `sfm.json` has none: the converter triangulates them (sparse points above), observations in the
  AliceVision convention. `--points none` leaves this to AliceVision's own SfM nodes.
- Run: PrepareDenseScene (`--masksFolders <out>/masks`) → DepthMap (`--downscale 1`: convert the tier you want, as
  `--downscale k` drops the half-pixel term) → DepthMapFilter → Meshing → MeshFiltering.
</details>

<details><summary><b>RealityScan</b> (<code>realityscan</code>)</summary>

- Produces `images/<name>.exr` (`--image-format png8|png16|jpg|tif16` possible), `<name>.xmp` (xcr 1.1, priors
  `locked` by default, `--rc-prior exact|initial`) and `<name>.exr.mask.png`.
- Offset +0.5, used for the paper (continuous image plane, as in nerfstudio `realitycapture_utils.py` and
  Metashape). The XMP stores `Rotation` (world-to-camera), `Position`, `FocalLength35mm = f·36/max(W, H)` and
  `PrincipalPointU/V = (c − W/2 or H/2)/max(W, H)`. RealityScan is closed source, so this cannot be checked in its
  code; a mesh shifted by about half a pixel against `gt/` would mean `--pixel-centre 0`.
- Run: add the `images/` folder (XMPs are read with the images), align with the locked priors, reconstruct.
</details>

<details><summary><b>NeuS2</b> (<code>neus2</code>) and <b>Neuralangelo</b> (<code>neuralangelo</code>)</summary>

- NeuS2: `transforms.json` (camera-to-world, OpenCV axes, normalised, per-frame intrinsics, `n2w`), RGBA
  `images/<name>.png`. Offset +0.5 (`src/testbed_nerf.cu:1216-1228`). Run `python scripts/run.py --scene
  <out>/transforms.json --name <run> --network dtu.json --n_steps 50000 --save_mesh`. All views must share one size.
- Neuralangelo: `transforms.json` (one shared intrinsic, OpenGL axes, `sphere_center`, `sphere_radius`),
  `images/<name>.png`, `neuralangelo.yaml`. Offset +0.5 (`projects/nerf/utils/camera.py:297-298`). Run
  `torchrun --nproc_per_node=1 train.py --config=<out>/neuralangelo.yaml --logdir=<logs>`, then
  `projects/neuralangelo/scripts/extract_mesh.py`.
</details>

<details><summary><b>Photometric stereo</b> (<code>ps</code>: <code>unimsps</code>, <code>sdmunips</code>, <code>lino</code>)</summary>

- The three methods are universal: they need no light calibration (the per-LED calibration is published in
  `calibration/lights/`). One folder per pose with **linear** images (one scale per pose, so the radiometry between
  LEDs is kept; `--scale image` for one scale per image) and the pose mask (`--mask-dilate` px at the tier, or
  `--mask-dilate-d1 N` px at d1, i.e. round(N/k) at tier dk; the paper's Uni-MS-PS runs used `--mask-dilate-d1 5`).
- `mvps` reads the normals as `normals/pose_KK.(exr|png)` (camera frame, OpenGL axes: x right, y up, z towards the
  camera): put each pose's normal map there. `check-normals` verifies the axes from the silhouettes alone.
- Uni-MS-PS: `inputs/pose_KK/lighting_LL.png` + `mask.png`, **8-bit** (it casts 16-bit images to `uint8` by keeping
  the low byte, `utils.py:22-23`). An LED image that is 0 inside the mask is dropped (`dropped_images` in the
  manifest). Run `inference_file.py --path_obj <pose folder> --folder_save normals --nb_img -1` on each pose (all
  30 LEDs, as in the paper).
- SDM-UniPS: `inputs/pose_KK.data/L_LL.png` + `mask.png`; run `sdm_unips/main.py --test_dir inputs --target
  normal_and_brdf` (normals and baseColor, usable with `mvps --reflectance`). The official code writes 8-bit PNG only; the fork in
  the table also writes float EXR.
- LINO-UniPS: `Real/pose_KK/L_LL.png` + `mask.png` (LINO picks its data branch from the path: `Real`, checked by
  the converter); run `eval.py --task_name Real --data_root Real --num_images N` (N in the manifest). The official code writes an 8-bit PNG; the
  fork in the table also writes float EXR.
</details>

<details><summary><b>MVPS</b> (<code>mvps</code>: <code>rnbneus2</code>, <code>supernormal</code>)</summary>

- Input: the PS normals of the **same** `sfm.json` (same size, checked), `--normal-convention opengl` (default) or
  `opencv`; the axes are verified on the silhouettes (`--no-normal-check` to skip). Optional `--reflectance`.
- RNb-NeuS2: `normal/NNN.png` (16-bit, camera frame), `mask/NNN.png`, `cameras.npz`, `views.json`. Offset +0.5
  (NeuS2 testbed). Run `python run_pipeline.py --input <out> --testbed build/testbed --output <res>` (`--scaling-mode
  none` keeps the converter's normalisation). The paper trained with normals only (no albedo).
- SuperNormal: `normal_world_space/NNN.exr` (float32, world frame), `mask/NNN.png`, `cameras_sphere.npz`,
  `supernormal.conf`. Offset +0 (`linspace(0, W − 1, W)`, `models/dataset_loader.py:121-122`). Run `python
  exp_runner.py --conf <out>/supernormal.conf --obj_name <name>`. The config adds `otype = HashGrid`: without it,
  tiny-cuda-nn silently falls back to OneBlob, which is what the paper's runs used.
</details>

## Tests

```bash
uv run pytest                    # from the repository root; set MARTINE_DATA=/path/to/martine for the data tests
uv run pytest -m "not slow"      # skip image conversions and triangulation
```

`tests/test_projection.py` converts the cameras of the test object at every published tier, loads the files the way
each method does (`tests/method_code/`; the original code is imported when `MARTINE_METHOD_REPOS` points to the
checkouts), and checks that 2 000 GT points land within 1e-3 px of the AliceVision projection. `MARTINE_COLMAP`
points to a COLMAP executable for the triangulation tests.

## License

MIT (see [`LICENSE-CODE`](../LICENSE-CODE)).
