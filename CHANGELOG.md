# Changelog

Versions of the dataset on Harvard Dataverse ([doi:10.7910/DVN/2BEPEH](https://doi.org/10.7910/DVN/2BEPEH)).
Images, image masks and cameras have not changed since v1.0.

## v3.0 (submitted for review on 2026-11-24)

Only `gt/challenges/` and one file of `gt/challenges_raw/` change; `gt.ply`, `gt_pcd.npy`, `attributes/`,
`masks_eval/` and `eval_cameras.sfm` are the files of v2.0, byte for byte.

- **Painted property masks recomputed** (`gt/challenges/<property>.npy`, 15 objects). A GT point used to take the
  label of the painted mesh only when the nearest painted *vertex* was within 0.1 mm, which lost the points lying
  inside the triangles of a mesh coarser than the GT point cloud. The label is now transferred from the nearest
  painted *triangle* within the same 0.1 mm (eval-3d
  [#41](https://github.com/martine-benchmark/eval-3d/pull/41)). Masks only gain points. Largest changes:
  - `grid` / `excluded`: 22.2 % → 75.5 % of the GT points;
  - `paw_pad` / `lambertian`: 27.7 % → 50.0 %, `paw_pad` / `transparent`: 27.3 % → 50.0 %;
  - `surgery_saw` / `specular`: +59 918 points (26.3 %).

  Smaller changes (+4 to +6 072 points per mask) on `can`, `candle`, `colander`, `jam_pot`, `knife`, `lego`,
  `petanque_ball`, `sneakers`, `spikes_ball`, `straw_hat`, `surgery_saw` and `whisk`.
- **`warrior_head` / `not_textured` repainted** on the published `gt.ply`: the previous painted mesh was an earlier
  version of the scan, with a hole at the base of the neck. `gt/challenges_raw/not_textured.ply` is replaced and the
  mask goes from 20.5 % to 23.5 % of the GT points.
- **`wood` is no longer textureless**: `wood/gt/challenges/not_textured.npy` is removed.
- Whole-object masks (all ones) and the masks of `moka_pot` and `surgery_scissors` are unchanged.

Effect on the evaluation: `excluded.npy` marks GT zones that are never evaluated, so the scores of the eight objects
with such a zone change (`candle`, `colander`, `grid`, `petanque_ball`, `sneakers`, `spikes_ball`, `straw_hat`,
`whisk`), `grid` most of all, and so do the per-property scores of the masks above. Use eval-3d at `559e136` or later.

Effect on the scores of the methods of the paper: the overall means move by at most 0.5 point of F-score at 0.5 mm;
the fine-structure scores change strongly with the new excluded zone of `grid` (22 % → 75 % of its GT); the
textureless scores, now without `wood`, drop by 4 to 8 points depending on the method. Detailed figures: paper and
website.

## v2.0 (2026-11-07)

- `croissant.json` added (Croissant 1.1 and RAI metadata, one `FileObject` per archive with its checksums).

## v1.0 (2026-11-06)

- First release: 28 objects, MVS (`d1` to `d8`) and MVPS (`d2` to `d8`) images, masks and cameras, MVPS dark frames,
  `calibration/` (lights, spheres, color checker), `gt/` for 26 objects, as TAR archives.
