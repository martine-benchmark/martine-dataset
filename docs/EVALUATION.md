# Evaluating a reconstruction on Martine

From a mesh reconstructed by your method to the paper's Chamfer, F-score and per-property scores, on the published
dataset layout. The evaluation code is a separate repository: [eval-3d](https://github.com/martine-benchmark/eval-3d).

The evaluation never writes into the dataset: it reads the evaluation kit under `<DATA>/<object>/gt/` and writes
into a separate workspace `<RUNS>`.

## 1. Install

```bash
git clone https://github.com/martine-benchmark/eval-3d.git
cd eval-3d
uv sync                                  # provides the `eval-pipeline` command
uv run eval-pipeline --help
# optional, for `visualize` only (needs OSMesa): uv sync --extra visualization
```

Without uv: `pip install -e .` in a Python ≥ 3.10 environment, then `eval-pipeline` directly.

## 2. Dataset and workspace

```
<DATA>/<object>/                         # the download (read-only)
├── mvs/, mvps/                          # reconstruction inputs, any tier
└── gt/
    ├── gt.ply, gt_pcd.npy               # GT mesh and its point sampling
    ├── attributes/, challenges/         # per-point properties (visibility, curvature, challenges)
    ├── eval_cameras.sfm, masks_eval/    # 84 full-resolution evaluation views (81 for grid)
    └── masks_watertight/                # hole culling, objects whose GT has holes (sneakers, wood)

<RUNS>/<object>/<method>/                # your workspace (written by the pipeline)
├── results_raw/mesh.ply                 # YOUR mesh (.obj also accepted): the only input you supply
├── results_cleaned/mesh.ply             # written by cleanup
└── eval_results/                        # metrics.json, curves/, distances/, zones/
```

## 3. Configure

Copy the Martine config to a local file (`*.local.yaml` is ignored by git) and edit its `paths:` block:

```bash
cp config/martine.yaml config/martine.local.yaml
CFG=config/martine.local.yaml
```

```yaml
paths:
  data_root:         "/path/to/martine/{object}/mvs/d1"   # not read by the evaluation
  gt_root:           "/path/to/martine/{object}/gt"
  gt_artifacts_root: "/path/to/martine/{object}/gt"
  eval_cameras:      "/path/to/martine/{object}/gt/eval_cameras.sfm"
  eval_masks:        "/path/to/martine/{object}/gt/masks_eval"
  watertight_masks:  "/path/to/martine/{object}/gt/masks_watertight"
  eval_root:         "/path/to/workspace/runs/{object}"   # written
```

`{object}` is filled from `dataset.objects` (the 26 objects with a GT; `pillow` and `coin` have none). Set
`dataset.methods` to your method folder names, or scope each command with `-m`. Scoring always uses the
full-resolution kit under `gt/`, whatever tier (`d1`/`d2`/`d4`/`d8`) the mesh was computed from.

## 4. Score a mesh

The mesh must be in the dataset frame, in millimetres, with **no alignment step**: reconstructing from the published
`sfm.json` puts it there (see the [converters](../converters/README.md)). Do not resample `gt.ply` either: every
method is scored against the shipped `gt_pcd.npy`.

```bash
uv run eval-pipeline -c $CFG run -o vase -m mymethod      # cleanup (silhouette carving) + evaluate
uv run eval-pipeline -c $CFG apply-masks -o vase -m mymethod   # per-property distances (section 6)
uv run eval-pipeline -c $CFG aggregate                    # tables over all evaluated cells
```

Each stage also runs alone (`cleanup`, `evaluate`, `curves -n 100`); `-o/--object`, `-m/--method` and `-f/--force`
scope them. Without `-o`/`-m`, a command covers every object × method of the config.

## 5. Read the metrics

`<RUNS>/<object>/<method>/eval_results/metrics.json` (distances in mm):

| field | meaning |
| :-- | :-- |
| `chamfer` | mean of both directions; `NaN` = nothing within `max_dist` (a result, not a crash) |
| `chamfer_a2b` / `chamfer_b2a` | accuracy (reconstruction → GT) / completeness (GT → reconstruction) |
| `coverage` | minimum of the two coverages: read it before the Chamfer (a fragment can be very accurate) |
| `precision` / `recall` / `fscore` | arrays aligned with `thresholds`: 0.5, 1, 1.5, 2, 5 mm (the paper reports F@0.5 mm) |
| `max_dist`, `seed` | echoed settings |

Settings of the Martine config, in mm: resampling step `downsample_density` 0.05, cutoff `max_dist` 4.0.

## 6. Per-property scores

The paper also scores each mesh on subsets of the GT points: visibility (number of views that see a point),
curvature (concave, flat, convex) and the object challenges (specular, transparent, etc.). `apply-masks` writes
these distances under `eval_results/zones/`. `aggregate` then pools them over the objects into
`<RUNS>/aggregated/{global,visibility,curvature,challenge}_metrics.json`.

```bash
uv run eval-pipeline -c $CFG apply-masks                  # every object × method; --no-visibility,
                                                          # --no-curvature, --no-challenges skip a family
uv run eval-pipeline -c $CFG aggregate                    # bins from the config (aggregation:)
uv run eval-pipeline -c $CFG latex                        # LaTeX tables of the aggregated metrics
```

The curvature bins (|k| > 0.75 mm⁻¹) come from the config's `aggregation:` block. `aggregate --visibility-groups`
and `--curvature-thresholds` override the bins, but the paper's numbers use the config's defaults.

## 7. Running on a cluster

Set `execution.mode: "slurm"` in your local config and fill `execution.slurm` (account, partition and resources per
stage). Then `uv run eval-pipeline -c $CFG run --slurm` submits one job per object × method cell. Run `aggregate`
once the jobs have finished.

## 8. Gotchas

- **Units are millimetres**: do not rescale the mesh.
- **No alignment**: the mesh is scored as it is. A mesh in another frame scores badly instead of failing.
- **Cameras**: cleanup uses `gt/eval_cameras.sfm` (the 84 MVS viewpoints, 81 for grid), whatever the
  reconstruction's modality or tier. `gt/masks_eval/` intentionally differ from the reconstruction masks.
- **`pillow` and `coin`** have no GT: imagery only, absent from the config's object list.
