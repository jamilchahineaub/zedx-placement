# ZED-X Dual-Camera Placement Experiment

Isaac Sim 5.1 + ZED SDK 5.3.1.

Finds the best placement (height, radius, angle) for two ZED-X cameras around a person.
Isaac Sim renders the scene and streams two virtual cameras; the ZED SDK tracks and fuses
the skeleton; we score it against Isaac's ground-truth joints (MPJPE, PCK, coverage).

## Run

From the repo root (`/home/jimmy/zedx-placement`).

```bash
# one layout, end to end
python3 scripts/run_pipeline.py --h 1.5 --r 2.5 --rel-az 90 \
    --subject-name center --layout-id my_run --machine laptop --mode both

# full sweep over the grid in config/experiment.yaml
python3 sweep.py --machine laptop
python3 sweep.py --machine laptop --limit 3      # quick 3-layout test

# clean up stale runs / free the ports
python3 scripts/preflight.py --machine laptop

# tests
python3 -m pytest analysis/tests/ -v
```

`run_pipeline.py` flags: `--h` height m, `--r` radius m, `--rel-az` cam B angle from cam A,
`--subject-name` (from experiment.yaml), `--layout-id` output label, `--machine`,
`--mode single|fusion|both`, `--model accurate|fast`, `--conf`, `--cams both|a|b`.

## Ports

Cameras stream on localhost ports 30000 (cam A) and 30002 (cam B). A crashed run leaves
them bound and the next run fails.

```bash
ss -tulnp | grep ':3000[0-9]'        # check
python3 scripts/preflight.py --machine laptop   # kill + free (easiest)

# manual
pkill -f zed_single; pkill -f zed_fusion; pkill -f run_episode
rm -f /dev/shm/sl_local_*
```

## Layout

```
config/machine.laptop.yaml   per-machine paths (change this on a new machine)
config/experiment.yaml       grid, room, cameras, occluders, metric params
isaac/                       runs under isaacsim/python.sh (omni/pxr/isaacsim)
  scene_builder.py           builds room, loads character, places + streams cameras
  run_episode.py             one Isaac episode
  gt_logger.py               writes ground-truth joints per frame
  camera_rig.py              camera math + evaluate_layout() (sweep boundary)
zed/                         runs under python3 (pyzed)
  zed_single.py              single-camera tracking
  zed_fusion.py              dual-camera fusion (the predictor)
  make_fusion_config.py      per-layout fusion config (Isaac->ZED pose conversion)
analysis/                    plain python3, unit-tested
  joint_map.py               ZED joints <-> Isaac rig joints
  metrics.py                 MPJPE, PCK, coverage, etc.
  geo_prescreener.py         geometric pre-filter before running Isaac
scripts/run_pipeline.py      one-command single layout
scripts/preflight.py         cleanup / free ports
sweep.py                     loops the grid, calls evaluate_layout per layout
results/                     output (append-only): CSVs, per-layout files, logs
```

Two Python runtimes, never mixed: `isaac/` uses `isaacsim/python.sh`; everything else uses
system `python3`.

## New machine

Edit/create `config/machine.<name>.yaml` (the only place with absolute paths), then pass
`--machine <name>`:

```yaml
isaac_python:    /path/to/isaacsim/python.sh
zed_python:      python3
zed_ext_path:    /path/to/zed-isaac-sim/exts
reference_scene: assets/test.usd         # bundled in repo (relative = resolved against repo root)
headless:        true                    # false = show viewport
```

`test.usd` ships in the repo under `assets/`, so it travels with a clone — no need to point
at an external path. The character mesh itself is still pulled from S3 at runtime, so the
machine needs internet. To change the scene, re-save into `assets/test.usd` and commit it.

## Changing things

- **Scene** — `config/experiment.yaml` `scene_type`: `warehouse` (default, Isaac
  Simple_Warehouse) or `simple_room` (procedural). Warehouse loads from the Isaac assets
  root (network) or a `warehouse_usd` override; loader in `isaac/scene_builder.py`
  `_load_warehouse`. The procedural room (`room_size_m`, `occluders`) is `_build_room` and
  applies only to `simple_room`.
- **Character** — `reference_scene` in the machine config + `character_prim` in
  experiment.yaml. Subject positions: `subject_positions` in experiment.yaml.
- **Motion** — `config/experiment.yaml` `character_motion`: `inplace` (default; deterministic
  baked in-place articulation) or `none` (static). Amplitudes/period in the `motion:` block;
  the animation is authored in `isaac/character_motion.py` (a UsdSkel clip, same motion every
  layout so placements compare fairly). `jitter_variance`/`id_drops` are live under motion,
  NaN when `none`.
- **Metrics** — compute in `analysis/metrics.py` `compute_metrics`, add the column to
  `RESULTS_COLUMNS`; joint pairing in `analysis/joint_map.py`. Run the tests after.
- **Sweep** — grid in `experiment.yaml`; search logic in `sweep.py` (`layouts()` + gates).
  It only calls `camera_rig.evaluate_layout`.
- **Third camera** — add `cam_c` (port 30004/serial 1003) in experiment.yaml; add a third
  annotator + ZED_X prim in `scene_builder.py`; a third pose in `make_fusion_config.py` +
  `zed360_template.json`; open/subscribe it in `zed_fusion.py`.

## Metrics (results.csv)

| Column | Meaning |
|---|---|
| `h_m`, `r_m`, `rel_az_deg` | layout inputs |
| `tilt_deg` | camera downward tilt |
| `convergence_angle_deg` | triangulation angle between the two cameras at the subject |
| `subject_pos_name` | subject position used |
| `mpjpe_mm` | absolute mean per-joint error, mm (includes the global fused↔GT frame offset; lower better) |
| `mpjpe_aligned_mm` | **true pose accuracy** — MPJPE after removing the per-frame global translation (registration offset). This is what ranking scores accuracy on, since the absolute offset is a fusion-calibration artifact, not a placement property |
| `registration_offset_mm` | the global fused↔GT frame translation (diagnostic). `zed_fusion` subscribes with `override_gravity=True` so camera poses are applied verbatim — this should stay small and uniform across tilts; large/tilt-growing values indicate a pose/calibration problem. `mpjpe_aligned_mm` factors it out regardless |
| `pck30`, `pck50` | fraction of joints within 30 / 50 mm (computed on the absolute error) |
| `detection_coverage` | fraction of frames a body was detected |
| `joint_visibility_cam_a/b/either` | fraction of joints in each camera's FOV+range |
| `joint_visibility_both` | fraction visible to both with a good triangulation angle |
| `unique_contribution_cam_b` | fraction cam B adds that cam A misses |
| `jitter_variance` | mean over joints of the temporal variance (mm²) of the prediction error — how much tracking wobbles around truth (live under motion; NaN if static) |
| `id_drops` | (#distinct tracked body-ids − 1) + (#tracked→untracked gaps) over the window (live under motion; NaN if static) |

Accuracy columns (`mpjpe`, `pck`, `coverage`) compare ZED prediction vs ground truth;
the rest are geometry from camera + joint positions. Computed in `analysis/metrics.py` and
`analysis/geo_prescreener.py`. Under motion, `mpjpe`/`pck`/`jitter`/`id_drops` use a
per-frame pass (GT↔prediction frames matched by `wall_clock`); a static character uses a
time-averaged pose. `jitter`/`id_drops` only differentiate layouts once the character moves.

## Ranking

`sweep.py` only fills `results.csv`; it does not rank. `analysis/rank.py` reads that CSV
(read-only) and ranks the camera positions, ending with **physical mounting instructions**
for the best layouts.

```bash
python3 analysis/rank.py                              # rank results/results.csv, print top 15
python3 analysis/rank.py --preset pose --top 10
python3 analysis/rank.py --weights pose_fidelity=0.6,absolute_placement=0.1,stability=0.3
python3 analysis/rank.py --out reports/ranking.csv    # also write a CSV (never into results/)
```

### Score
Layouts are grouped by `(h, r, rel_az)` (averaged across subjects), then scored on **three
axes** (all weights/bands in `config/experiment.yaml → ranking:`):

```
score = 0.50·pose_fidelity + 0.20·absolute_placement + 0.30·stability
```

| Axis | weight | metric → goodness (0–1) |
|---|---|---|
| **pose_fidelity** | 0.50 | `mpjpe_aligned_mm`, band 20–200 mm → `(200−x)/180` clamped. True pose accuracy, **registration-invariant** — the main thing. |
| **absolute_placement** | 0.20 | `mpjpe_mm`, band 50–1000 mm. World-position accuracy; down-weighted because it's dominated by the fusion-registration offset (a calibration artifact). |
| **stability** | 0.30 | `0.5·rank(id_drops) + 0.5·rank(jitter_variance)`. **Rank-normalized** = outlier-robust, so one catastrophic-jitter layout can't flatten the scale. |

Dropped from scoring (no signal in this sweep, report-only): `detection_coverage` (≈1.0),
`pck`, `joint_visibility_*`, `unique_contribution_cam_b`.

### Robust to the weights
- **Validity flag (not a gate):** every layout is ranked, but marked invalid if
  `registration_offset_mm > 300`, `jitter_variance > 50000`, or `coverage < 0.8` (`ranking.validity`).
- **Pareto frontier** over the 3 axes (weight-free) — because pose and absolute are separate
  axes, a great-pose/poor-placement layout still surfaces (and is flagged).
- **Preset sensitivity** (`balanced` / `pose` / `absolute` / `stability`) shows whether the
  top picks move when the weights change.
- **Mounting candidates + insight line:** the top valid+Pareto layouts are printed as
  "mount both cameras at H m height, R m from the subject, separated by AZ°", plus the
  height/tilt/radius correlation.

Tune anything in the `ranking:` block — no code change. Current full-sweep finding: accuracy
is driven by **camera height/tilt** (lower is better), not radius.

## Notes

- Default scene is the **warehouse** with an **in-place moving human** (`scene_type:
  warehouse`, `character_motion: inplace`). Set `scene_type: simple_room` /
  `character_motion: none` to fall back to the static procedural-room setup.
- The **warehouse loads over the network** (Isaac assets root / S3), like the character
  mesh — the machine needs internet. If `get_assets_root_path()` is unset, put a full path
  or the S3 URL in `warehouse_usd` (`config/experiment.yaml`).
- Use `--model accurate --conf 20` (defaults); `fast` doesn't detect the render.
- The ground-truth CSV is written at episode end — don't close the viewport mid-run.
- `results/` is append-only. `reference_scene` (test.usd) is never written back to.
- **Fusion poses are applied verbatim** via `fusion.subscribe(..., override_gravity=True)`
  in `zed/zed_fusion.py` (a ZED-SDK pose flag — *not* Isaac physics gravity). This replaced
  an empirical doubled-pitch hack that only held at one tilt; poses now generalize across
  tilt. Confirm with `registration_offset_mm` staying small/uniform after a re-sweep.
