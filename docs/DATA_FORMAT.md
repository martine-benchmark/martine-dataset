# Martine data format

Layout of the published dataset, file naming and acquisition order, and the camera conventions. This file is the
single reference for the conventions; the converters and the evaluation follow it.

## Layout

```text
martine_dataset/
├── <object>/
│   ├── gt/                                # absent for pillow and coin (no ground truth)
│   │   ├── gt.ply                         # scanned GT mesh (mm)
│   │   ├── gt_pcd.npy, gt_pcd_provenance.json   # GT point sampling used by the evaluation
│   │   ├── attributes/                    # per-point curvature and visibility count
│   │   ├── challenges/                    # <property>.npy boolean masks over gt_pcd.npy, excluded.npy
│   │   ├── challenges_raw/                # painted sources of the partial masks (when any)
│   │   ├── eval_cameras.sfm, masks_eval/  # posed full-resolution MVS views of the evaluation (84; 81 for grid)
│   │   └── masks_watertight/              # hole-culling masks (sneakers, wood)
│   ├── mvs/<tier>/                        # single light; tiers d1, d2, d4, d8
│   │   ├── images/NNN.exr                 # views in capture order
│   │   ├── masks/NNN.png
│   │   └── sfm.json                       # AliceVision cameras of the tier
│   └── mvps/<tier>/                       # 30 LEDs; tiers d2, d4, d8
│       ├── images/pose_KK/lighting_LL.exr # pose KK lit by LED LL
│       ├── dark/pose_KK.exr               # black frame of pose KK (all LEDs off)
│       ├── masks/pose_KK.png
│       └── sfm.json
└── calibration/                           # each folder has its README.md
    ├── lights/                            # per-LED calibration (camera frame, mm)
    ├── spheres/{lambertian,specular}/     # captures behind the light calibration, d1 only
    │   ├── d1/images/view_XX/lighting_LL.exr  # turntable view XX (20, 18° steps) lit by LED LL
    │   ├── d1/dark/view_XX.exr            # black frame of view XX (all LEDs off)
    │   ├── d1/masks/view_XX.png           # sphere mask
    │   ├── d1/sfm.json                    # AliceVision cameras of the capture (SfM units, not mm)
    │   └── sphere.json                    # fitted sphere (centre, radius) in the frame of d1/sfm.json
    └── color_checker/                     # ColorChecker Classic, one viewpoint
        ├── mvs.png                        # under the MVS lighting
        └── mvps.png                       # under the MVPS lighting (one LED)
```

The calibration spheres (Lambertian and specular) were captured with the same camera, lens and LEDs as the objects,
and processed like the MVPS images (undistorted linear EXR, background blacked out beyond 150 px), at d1 only; LED
`LL` is the same physical LED as in the objects. The ColorChecker images are the developed images as provided (8-bit
sRGB, 4784 x 3188, the size of the d2 tier): no raw capture, mask or camera pose.

Images are 16-bit (half) EXR, background blacked out beyond a safety margin of 150 px at d1 (75 / 37 / 18 px at
d2 / d4 / d8). Masks of every tier are derived from the full-resolution mask (`INTER_AREA`, thresholded at 128).

## Naming and acquisition order

Each object was captured in two sweeps of the same camera trajectory: once under a single light (**MVS**), once
under the 30 LEDs of the light dome (**MVPS**). Both are calibrated in the same metric frame, so MVS view *k* and
MVPS pose *k* are close viewpoints, but distinct shots with their own poses.

- **MVS views**: `images/NNN.exr` are numbered in capture order (EXIF `DateTimeOriginal` + `SubsecTimeOriginal`),
  so consecutive indices are neighbouring viewpoints. The index is the same in every tier, in `masks/NNN.png`, in
  the `sfm.json` view paths, and in `gt/eval_cameras.sfm` / `gt/masks_eval/NNN.png`.
- **MVPS poses**: `pose_KK/` are numbered in capture order as well.
- **MVPS lights**: the LEDs fire in the same fixed sequence at every pose, so `lighting_LL.exr` is always lit by
  LED `LL`, for every pose and object; `calibration/lights/` gives LED `LL` at index `LL`. LED 24 gives much darker
  images (grazing light): they are neither missing nor corrupt.
- **MVPS black frames**: `dark/pose_KK.exr` is the shot taken with every LED off at pose `KK` (the first of the 31
  shots of the pose), developed exactly like its `lighting_LL.exr` (same camera, undistortion, resolution and mask),
  e.g. to subtract the ambient light and the sensor offset. It has no view in `sfm.json`: use the camera of pose
  `KK`. Available in every MVPS tier for the 28 objects.
- **Identifiers**: `viewId` / `poseId` in `sfm.json` are AliceVision identifiers and carry no order: use the file
  names. Each view keeps the `metadata` of its RAW shot (capture time, camera body and lens, RAW conversion).
- **Views without a pose**: three MVS views of `grid` (`019`, `027`, `065`) could not be registered by the
  structure from motion (rerunning it from the RAW files fails on the same three views). They are kept so that the
  numbering stays continuous, but their image and mask are entirely black, their `poseId` has no entry in the
  `poses` of `sfm.json`, and they are absent from `gt/eval_cameras.sfm` and `gt/masks_eval/` (81 evaluation views
  for `grid`). Skip views without a pose when loading the cameras. Every other view and every MVPS pose of every
  object has a pose.

## Camera conventions

The images are undistorted; the cameras are pinhole, given only as AliceVision `sfm.json` files, in one pixel
convention for every tier.

- **Pixels**: the centre of the top-left pixel is at (0, 0). `principalPoint` is an offset from the image centre:
  `cx = width/2 + principalPoint[0]`, `cy = height/2 + principalPoint[1]`; the focal length is in mm,
  `fx = focalLength * width / sensorWidth`.
- **Tiers**: `d2`, `d4` and `d8` are pixel-area (OpenCV `INTER_AREA`) downsamplings of `d1` by `s = 1/2, 1/4, 1/8`
  (MVPS `d4` and `d8` are 1/2 and 1/4 of `d2`). The resampling is centre-aligned, so
  `cx_dk = s * (cx_d1 + 0.5) - 0.5`, i.e. `principalPoint_dk = s * principalPoint_d1 - (1 - s)/2`.
- **Poses**: `rotation` is camera-to-world (row-major; its columns are the camera axes x right, y down, z forward)
  and `center` the camera centre, in mm: `X_cam = Rᵀ (X − C)`.
- **Frame**: the `sfm.json` poses are in the AliceVision frame. For objects with ground truth, the GT mesh
  (`gt.ply`, `gt_pcd.npy`) is in the same frame up to `W = diag(1, −1, −1)`: `x_gt = W x_sfm`. To use the cameras
  with the mesh, move them with `W`: `C' = W C`, `R_c2w' = W R_c2w` (the evaluation code does the same). Without it
  the object still projects in front of the cameras and inside the images, but seen from another angle, so check
  the overlap with the masks.
- **pillow and coin** (no GT) are in a canonical turntable frame, in mm: origin on the turntable axis at the fabric
  height, −Y up in `sfm.json` coordinates (+Y after `W`), +Z horizontal towards the camera of MVS view 000. Their
  metric scale is approximate: about 0.2 % for `pillow`, about 1 % for `coin` (shot at 58 mm instead of 42 mm,
  scale from the turntable fabric texture).
- **Other formats** (COLMAP, NeRF-style `transforms.json`, ...) are not shipped: `martine-convert` writes each method's
  format in the pixel convention it expects ([converters](../converters/README.md)).

## Challenge masks

`gt/challenges/<property>.npy` (`lambertian`, `specular`, `transparent`, `not_textured`, `fine_structure`) are
boolean masks over `gt/gt_pcd.npy`: a property counts for an object if and only if its `.npy` exists. A property
covering the whole object ships as an all-ones mask; a partial one is painted on the GT mesh (sources in
`gt/challenges_raw/`). `gt/challenges/excluded.npy` marks unreliable GT zones, never evaluated. *High curvature* and
*low visibility* are continuous per-point properties (`gt/attributes/`), binned by the evaluation.
