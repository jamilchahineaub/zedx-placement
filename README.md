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
reference_scene: /path/to/test.usd       # character source, read-only
headless:        true                    # false = show viewport
```

## Changing things

- **Room / occluders** — `config/experiment.yaml` (`room_size_m`, `occluders`);
  geometry in `isaac/scene_builder.py` `_build_room`.
- **Character** — `reference_scene` in the machine config + `character_prim` in
  experiment.yaml. Subject positions: `subject_positions` in experiment.yaml.
- **Animation** — character is static; in-place motion needs `omni.anim.people` wired into
  `scene_builder.py` (planned follow-up). `jitter_variance`/`id_drops` are NaN until then.
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
| `mpjpe_mm` | mean per-joint error, mm (main accuracy number; lower better) |
| `pck30`, `pck50` | fraction of joints within 30 / 50 mm |
| `detection_coverage` | fraction of frames a body was detected |
| `joint_visibility_cam_a/b/either` | fraction of joints in each camera's FOV+range |
| `joint_visibility_both` | fraction visible to both with a good triangulation angle |
| `unique_contribution_cam_b` | fraction cam B adds that cam A misses |
| `jitter_variance`, `id_drops` | NaN until animation is added |

Accuracy columns (`mpjpe`, `pck`, `coverage`) compare ZED prediction vs ground truth;
the rest are geometry from camera + joint positions. Computed in `analysis/metrics.py` and
`analysis/geo_prescreener.py`.

## Notes

- Use `--model accurate --conf 20` (defaults); `fast` doesn't detect the render.
- The ground-truth CSV is written at episode end — don't close the viewport mid-run.
- `results/` is append-only. `reference_scene` (test.usd) is never written back to.
- Current calibrated fusion: ~112 mm MPJPE.
