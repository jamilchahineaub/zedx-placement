# ZED-X Dual-Camera Placement Experiment

**Isaac Sim 5.1 + ZED SDK 5.3.1**

This repo finds the *best physical placement* (height, radius, relative angle) for two
ZED-X cameras around a person, by simulating the scene in NVIDIA Isaac Sim, streaming
the synthetic camera feeds into the real ZED SDK body-tracking + fusion runtime, and
scoring how accurately the fused skeleton matches the simulator's ground-truth skeleton
(MPJPE, PCK, coverage, occlusion robustness).

The simulator gives us **perfect ground truth** (we know exactly where every joint is),
so we can sweep hundreds of camera layouts automatically and rank them.

---

## Table of contents

1. [How it works (the pipeline)](#1-how-it-works-the-pipeline)
2. [Repo layout](#2-repo-layout)
3. [The two-runtime rule (read this first)](#3-the-two-runtime-rule-read-this-first)
4. [Commands — what to run and what each does](#4-commands--what-to-run-and-what-each-does)
5. [Moving to a different machine / folder / addresses](#5-moving-to-a-different-machine--folder--addresses)
6. [Ports: how to check them and kill them](#6-ports-how-to-check-them-and-kill-them)
7. [Troubleshooting / hurdles and their fixes](#7-troubleshooting--hurdles-and-their-fixes)
8. [How to change things](#8-how-to-change-things)
   - [Change the environment / room](#change-the-environment--room)
   - [Change the people / character](#change-the-people--character)
   - [Change what the people are doing (animation)](#change-what-the-people-are-doing-animation)
   - [Change / add metrics](#change--add-metrics)
   - [Change the sweep algorithm](#change-the-sweep-algorithm)
   - [Add a third camera](#add-a-third-camera)
9. [Metrics reference — what each one means and where it's computed](#9-metrics-reference--what-each-one-means-and-where-its-computed)
10. [Coordinate systems (the part that bites you)](#10-coordinate-systems-the-part-that-bites-you)
11. [Outputs / where results land](#11-outputs--where-results-land)

---

## 1. How it works (the pipeline)

One layout = one `(height h, radius r, relative azimuth rel_az, subject position)`.
For each layout the pipeline does:

```
preflight        kill stale processes, free the streaming ports, wipe shared memory
   │
make_fusion_config   write the per-layout ZED fusion JSON (camera poses, converted to ZED frame)
   │
Isaac Sim        build room + character + two ZED camera prims, START STREAMING on ports 30000/30002
   │             and write the ground-truth skeleton CSV every frame (gt_logger)
   │
ZED SDK          open the two streams, run HUMAN_BODY tracking, fuse the two views
   │             → predicted skeleton CSV
   │
metrics          map ZED joints ↔ Isaac joints, time-average, compute MPJPE/PCK/coverage…
   │
results.csv      append one row for this layout
```

The cameras are **virtual** — Isaac renders them and streams them over `127.0.0.1` using
the ZED extension, exactly as if they were two real ZED-X cameras plugged in. The ZED SDK
on the other side cannot tell the difference.

---

## 2. Repo layout

```
zedx-placement/
├── config/
│   ├── machine.laptop.yaml     # per-machine paths (CHANGE THIS on a new machine)
│   └── experiment.yaml         # sweep grid, room, cameras, occluders, metrics params
├── isaac/                      # runs under Isaac's python  (imports pxr, omni, isaacsim)
│   ├── scene_builder.py        # builds room, loads character, places + streams cameras
│   ├── run_episode.py          # one Isaac episode end-to-end (called by the orchestrator)
│   ├── gt_logger.py            # writes ground-truth joint CSV every physics step
│   └── camera_rig.py           # pure-math camera placement + evaluate_layout() (sweep boundary)
├── zed/                        # runs under system python3  (imports pyzed.sl only)
│   ├── zed_single.py           # single-camera body tracking
│   ├── zed_fusion.py           # dual-camera fusion (the real predictor)
│   ├── make_fusion_config.py   # generates the per-layout fusion JSON (Isaac→ZED pose conversion)
│   ├── zed360_template.json    # fusion config template (do NOT run ZED360)
│   └── generate_fusion_template.py
├── analysis/                   # plain python3, unit-tested
│   ├── joint_map.py            # ZED BODY_18 joint names ↔ Isaac rig joint names
│   ├── metrics.py              # MPJPE, PCK, coverage, jitter… + results.csv columns
│   ├── geo_prescreener.py      # cheap geometric pre-filter (VRST) before running Isaac
│   └── tests/                  # pytest suite (run after every analysis/ edit)
├── scripts/                    # plain python3 orchestration
│   ├── run_pipeline.py         # ONE COMMAND: run a single layout end-to-end
│   └── preflight.py            # cleanup gate (kill stale procs, free ports, wipe SHM)
├── sweep.py                    # loops the whole grid, calls evaluate_layout per layout
├── results/                    # OUTPUT — append-only (CSVs, per-layout artifacts, logs)
├── CLAUDE.md                   # engineering ground-truth notes (read before editing)
└── README.md                   # this file
```

---

## 3. The two-runtime rule (read this first)

**There are two completely separate Python environments. Never mix them.**

| Code under… | Runs with… | Imports |
|---|---|---|
| `isaac/` | `/home/jimmy/isaacsim/python.sh` | `pxr`, `omni`, `isaacsim` |
| `zed/`, `analysis/`, `scripts/`, `sweep.py` | system `python3` | `pyzed.sl` (zed only) |

`scripts/run_pipeline.py` and `sweep.py` run under **system python3** and they launch the
Isaac side as a subprocess with `isaacsim/python.sh` for you. You normally never call
`isaacsim/python.sh` by hand except for debugging.

If you ever see `ModuleNotFoundError: No module named 'omni'` you ran an `isaac/` script
with the wrong Python. If you see `No module named 'pyzed'` you ran a `zed/` script with
Isaac's Python.

---

## 4. Commands — what to run and what each does

All commands are run **from the repo root** (`cd /home/jimmy/zedx-placement`).

### Run a single layout end-to-end (the main command)

```bash
python3 scripts/run_pipeline.py \
    --h 1.5 --r 2.5 --rel-az 90 \
    --subject-name center \
    --layout-id my_run \
    --machine laptop \
    --mode both
```

What it does, in order: preflight cleanup → generate the fusion config → launch Isaac
(builds the scene, starts streaming, logs ground truth) → wait for both streams to be
live → run ZED tracking/fusion → shut Isaac down cleanly → leave the artifacts in
`results/`. Prints the sentinels `PREFLIGHT_OK → STREAMING_STARTED → ZED_SINGLE_READY /
FUSION_READY → PIPELINE_OK`.

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--h` | `1.5` | camera height (m), same for both cameras |
| `--r` | `2.5` | camera radius from subject (m), same for both |
| `--rel-az` | `90` | azimuth of cam B relative to cam A (degrees); cam A is always at 0° |
| `--subject-name` | `center` | which subject position to use (names from `experiment.yaml → subject_positions`) |
| `--layout-id` | *(required)* | label for output files (`zed_pred_<id>.csv`, logs, fusion config) |
| `--machine` | `laptop` | which `config/machine.<name>.yaml` to read |
| `--episode-duration` | `150` | how long Isaac runs (s). The GT CSV is only written at episode end — don't cut it short. |
| `--capture-duration` | `20` | how long the ZED side captures (s) |
| `--mode` | `single` | `single` = one camera, `fusion` = fused only, `both` = run single **and** fusion |
| `--model` | `accurate` | `accurate` (HUMAN_BODY_ACCURATE) or `fast`. **Use accurate** — fast does not detect the synthetic render. |
| `--conf` | `20` | body-detection confidence threshold. Keep low (~20) for the synthetic render. |
| `--cams` | `both` | which Isaac cameras to stream: `both`, `a`, or `b` |
| `--transport` | *(cfg)* | override stream transport: `BOTH` / `NETWORK` / `IPC` |
| `--skip-preflight` | off | skip the cleanup gate (only if you know ports are already clean) |

### Run the full sweep (all layouts)

```bash
python3 sweep.py --machine laptop                 # full grid
python3 sweep.py --machine laptop --limit 3       # mini-sweep gate (first 3 survivors)
```

`sweep.py` iterates every `(h, r, rel_az)` in `experiment.yaml`, drops layouts that fail
the **tilt gate** (`tilt > max_tilt_deg`) or the **VRST geometric prescreen** (cheap, no
Isaac), then calls `camera_rig.evaluate_layout(...)` on the survivors and appends one row
per layout to `results/results.csv`. Skipped layouts print `GEO_SKIPPED <id> <reason>`.

| Flag | Default | Meaning |
|---|---|---|
| `--machine` | `laptop` | machine config to use |
| `--subject-name` | `center` | subject position for the whole sweep |
| `--limit` | none | stop after N evaluated layouts (use `3` for the smoke-test gate) |
| `--episode-duration` | `120` | per-layout Isaac duration (s) |
| `--capture-duration` | `20` | per-layout ZED capture (s) |
| `--mode` | `fusion` | `fusion` or `single` |

### Preflight cleanup only (kill stale runs / free ports)

```bash
python3 scripts/preflight.py --machine laptop
```

Kills stale `zed_single.py` / `zed_fusion.py` / `run_episode.py` / Isaac `kit` processes
(without killing your own shell), waits until ports `30000-30005` are free, and removes
`/dev/shm/sl_local_*` shared-memory topics. Prints `PREFLIGHT_OK` or
`PREFLIGHT_FAILED <reason>`. Run this whenever a previous run crashed and the next run
won't start.

### Run the unit tests (after any `analysis/` edit)

```bash
python3 -m pytest analysis/tests/ -v
```

### Generate a fusion config by hand (debugging the camera poses)

```bash
python3 zed/make_fusion_config.py \
    --out results/layouts/fusion_config_test.json \
    --h 1.5 --r 2.5 --rel-az 90
```

### Debug the Isaac side alone (advanced)

```bash
/home/jimmy/isaacsim/python.sh isaac/run_episode.py \
    --h 1.5 --r 2.5 --rel-az 90 --subject-name center \
    --layout-id debug --machine laptop --cams both --transport BOTH
```

Note the **Isaac Python** here, not `python3`. Useful to watch the viewport and confirm
the room, the character, and the two camera prims look right. Don't close the viewport
mid-episode — the GT CSV is only flushed at episode end.

---

## 5. Moving to a different machine / folder / addresses

**Almost everything machine-specific lives in `config/machine.<name>.yaml`. Nothing is
hardcoded in the code — paths are always read from this file.**

`config/machine.laptop.yaml` today:

```yaml
isaac_python:    /home/jimmy/isaacsim/python.sh          # Isaac Sim's python.sh
zed_python:      python3                                  # system python with pyzed
zed_ext_path:    /home/jimmy/zed-isaac-sim/exts           # ZED Isaac extension source tree
reference_scene: /home/jimmy/Desktop/jimmyNYUL/test.usd   # the character source scene (read-only)
headless:        false                                    # laptop shows the viewport; batch = true
```

**To run on a new machine (e.g. the 4090 box):**

1. Copy the repo to the new machine.
2. Create `config/machine.4090.yaml` with that machine's real paths:
   ```yaml
   isaac_python:    /path/to/isaacsim/python.sh
   zed_python:      python3
   zed_ext_path:    /path/to/zed-isaac-sim/exts
   reference_scene: /path/to/test.usd
   headless:        true      # batch machine: no viewport
   ```
3. Run everything with `--machine 4090`:
   ```bash
   python3 sweep.py --machine 4090
   ```

**What each field is and where to get it:**

| Field | What it is | How to find it |
|---|---|---|
| `isaac_python` | Isaac Sim's bundled Python launcher | the `python.sh` in your Isaac Sim install dir |
| `zed_python` | system Python that has `pyzed` installed | usually `python3`; verify with `python3 -c "import pyzed.sl"` |
| `zed_ext_path` | the `exts/` folder of the ZED Isaac-Sim extension | the editable source tree (has `.ogn` + python), not `_build/...` |
| `reference_scene` | the `.usd` file containing the character rig | provided asset; **read-only — never saved back** |
| `headless` | `true` = no GUI (batch), `false` = show viewport | `true` on servers, `false` on a desktop you watch |

**Addresses / ports** are NOT in the machine config — they're in `experiment.yaml`
(`cam_a.port: 30000`, `cam_b.port: 30002`) and the streams are always on `127.0.0.1`
(localhost). The whole pipeline runs on one machine, so the IP never changes. If you ever
split Isaac and the ZED SDK across two machines you'd change `127.0.0.1` to the sender's
IP in `zed/zed_single.py` (`set_from_stream(...)`) and `zed/zed_fusion.py`, and use
`transport: NETWORK` — but that is not the supported path; same-machine SHM is.

**Folder locations:** the repo can live anywhere; code computes its own root. The only
absolute paths are the four in the machine config above. The `results/` folder is created
under the repo root automatically.

---

## 6. Ports: how to check them and kill them

The cameras stream on **TCP/UDP ports 30000 (cam A) and 30002 (cam B)** on localhost
(fusion uses a few more in the 30000-30005 range). A crashed run leaves these ports bound
and the next run fails to start (`stream_dead`, or the receiver falls back to UDP and
finds nothing). This is the #1 recurring hurdle.

### Check which ports are in use

```bash
# Show anything listening on the 3000x streaming ports:
ss -tulnp | grep -E ':3000[0-9]'

# Same, UDP only:
ss -lunp | grep -E ':3000[0-9]'

# Find ZED/Isaac processes that might be holding them:
pgrep -af 'zed_single|zed_fusion|run_episode'
pgrep -af 'isaacsim|/kit'

# Check for leftover shared-memory streaming topics:
ls -l /dev/shm/sl_local_*
```

### Kill the ports / stale runs

**Easiest — let preflight do it (recommended):**

```bash
python3 scripts/preflight.py --machine laptop
```

**Manually, if you need to:**

```bash
# Kill the ZED/Isaac processes by name:
pkill -f zed_single
pkill -f zed_fusion
pkill -f run_episode
pkill -f 'isaacsim.*kit'         # Isaac kit process

# Kill whatever is bound to a specific port (e.g. 30000):
fuser -k 30000/udp 30000/tcp     # or:  kill $(lsof -ti udp:30000)

# Remove leftover shared-memory streaming topics + semaphores:
rm -f /dev/shm/sl_local_*
rm -f /dev/shm/sem.sl_local_*
```

After killing, re-check with `ss -tulnp | grep ':3000'` — it should print nothing before
you start a new run.

---

## 7. Troubleshooting / hurdles and their fixes

| Symptom | Cause | Fix |
|---|---|---|
| **Run won't start / `PREFLIGHT_FAILED ports_busy`** | Stale process from a crashed run holding 30000/30002, or leftover `/dev/shm/sl_local_*`. | `python3 scripts/preflight.py --machine laptop`. If it still fails, kill manually (see §6) and `rm -f /dev/shm/sl_local_*`. |
| **0 body-tracking frames** even though Isaac streams cleanly | (a) stale ports/SHM; (b) receiver started before the stream; (c) `zed.open()` hangs on a dead stream. | All three are handled now: preflight clears (a); the orchestrator waits for `STREAMING_STARTED` before opening (b); `zed_single.py` arms a `SIGALRM` hard-cap on `zed.open()` (c). If you still see it, run preflight and retry. |
| **Detection finds 0 bodies** (frames grabbed but no skeleton) | `fast` model + high confidence can't see the synthetic render. | Use `--model accurate --conf 20` (these are the defaults). Add `--save-frame foo.png` to `zed_single.py` to eyeball what the camera sees. |
| **Fused skeleton is metres off / two bodies never merge** | Wrong camera-pose conversion in the fusion config. | Fixed in `make_fusion_config.py`: world-only permutation `P@R`, store in ZED **image** frame `D(PR)D`, and write **doubled pitch** (`file pitch = -2*tilt`) because the SDK adds back `+tilt`. Final MPJPE ≈ 112 mm. Don't revert this. |
| **Process segfaults at the end (rc = -11)** | The ZED SDK segfaults tearing down cameras while Fusion is still subscribed. | `zed_fusion.py` flushes its CSVs then calls `os._exit(rc)` to skip the broken teardown. Expected; artifacts are already written. |
| **The warehouse / a second scene "merged" into ours** | `reference_scene` (`test.usd`) contains extra prims (a warehouse, extra ZED rigs). | `scene_builder._load_character` now **dynamically strips** everything except the prim subtree that actually contains a `UsdSkel` skeleton. No config change needed; it logs `kept [...], deactivated N`. |
| **`wait_for_streaming` fails with `episode_finished_before_streaming_confirmed`** | The SDK's "initialized successfully" line is filtered out of the log by carb. | Readiness is now defined as the streamer-init lines appearing **and no error for a grace period** — already handled in the orchestrator. |
| **Preflight killed my shell (exit 144)** | Earlier preflight killed its own ancestors. | Fixed: `preflight.py` excludes itself and its parent processes (`_ancestors()`). Update if you copied an old version. |
| **`FileNotFoundError: zed/make_fusion_config.py` in a background run** | Background shells don't inherit the working directory. | Prefix the command with `cd /home/jimmy/zedx-placement && …`. |
| **`No module named 'omni'` / `No module named 'pyzed'`** | Wrong Python for that runtime. | `isaac/` → `isaacsim/python.sh`; everything else → `python3`. See §3. |
| **Character is frozen (joints don't move between frames)** | The biped is a static rigged mesh; motion needs `omni.anim.people` at runtime (not yet wired). | Known/accepted. Static-pose metrics (MPJPE/PCK/coverage) are valid; `jitter_variance`/`id_drops` are `NaN` until animation lands. See [animation](#change-what-the-people-are-doing-animation). |
| **GT CSV is empty / missing** | The viewport was closed mid-episode, or the episode was cut short. | The GT CSV is written at **episode end** — let it finish, don't close the Isaac window mid-run. |
| **Single-cam knees are ~900 mm off but fusion fixes them** | The table occludes camera A's view of the legs. | This is the signal the experiment measures, not a bug — fusion recovers the occluded joints (knees → 75-98 mm). |

---

## 8. How to change things

### Change the environment / room

The room is **procedurally built** (not from the USD file) so it's easy to edit.

- **Room size, occluders:** `config/experiment.yaml`
  ```yaml
  room_size_m: [6.0, 6.0, 6.0]
  occluders:
    table:  true    # 1 x 2 x 0.8 m box at (1.2, 0, 0.4)
    pillar: true    # 0.4 m dia, 2 m tall cylinder at (-1.2, 1.0, 1.0)
  ```
  Flip `table`/`pillar` to `false` to study the scene without occlusion.
- **Geometry / textures / lights / occluder positions:** `isaac/scene_builder.py`,
  function `_build_room(stage, cfg)` (floor, 4 walls, dome lights, table, pillar — all
  checker-textured because ZED needs texture to track). Add or move boxes/cylinders here.
  To add a new toggleable occluder, add a key under `occluders:` in the config and read
  it in `_build_room`.

### Change the people / character

- **Swap the character:** point `reference_scene` in `config/machine.<name>.yaml` at a
  different `.usd`, and set `character_prim` in `experiment.yaml` to the prim path inside
  it. `scene_builder._load_character` auto-keeps only the subtree containing a
  `UsdSkel.Root`/`Skeleton`, so extra props in the file are stripped automatically.
- **Where the subject stands:** `experiment.yaml → subject_positions` (named positions);
  pick one with `--subject-name`. Add new positions here:
  ```yaml
  subject_positions:
    - name: center
      pos:  [0.0, 0.0, 0.0]
    - name: my_spot
      pos:  [1.0, -0.5, 0.0]
  ```
- **If you change the rig** (different skeleton joint names), you must also update the
  joint mapping — see [metrics](#change--add-metrics). The mapping in
  `analysis/joint_map.py` is specific to the current rig (`male_adult_police_04`).

> `reference_scene` / `test.usd` is **read-only**. The pipeline never saves back to it —
> all edits are in-memory and discarded when Isaac closes.

### Change what the people are doing (animation)

Currently the character holds a **static pose** (a rigged mesh with no baked animation).
`gt_logger.py` already logs the skeleton **per frame**, so adding motion needs no pipeline
changes — only the Isaac scene needs to drive the skeleton.

To add in-place motion (gestures, turn-in-place, one step forward/back — *not*
walk-around navigation), wire **`omni.anim.people`** in `isaac/scene_builder.py`:
enable the `omni.anim.people` extension, attach its animation graph/behavior to the biped,
and inject in-place commands; then step the simulation each frame in `run_episode.py`
(already a per-frame loop). Once joints move between frames, `jitter_variance` and
`id_drops` in `metrics.py` become meaningful (they're `NaN` on a static pose). This is the
planned follow-up; the metrics pipeline already supports it.

### Change / add metrics

Three files, in this order:

1. **`analysis/joint_map.py`** — the dict `ZED18_TO_ISAAC` maps ZED `BODY_18` joint names
   (spelled-out, e.g. `RIGHT_SHOULDER`) to the Isaac rig joint names (e.g. `R_Upperarm`).
   If you change the body format (e.g. `BODY_34`/`BODY_38`) or the rig, edit this dict.
   `mapped_pairs()` / `isaac_names()` expose the usable pairs.
2. **`analysis/metrics.py`** — `compute_metrics(...)` computes every value. To **add a
   metric**: compute it inside `compute_metrics`, then add its column name to
   `RESULTS_COLUMNS` (the list that defines the CSV header order). `append_results_row`
   writes the row append-only.
3. **`analysis/tests/`** — add/adjust a pytest case, then run
   `python3 -m pytest analysis/tests/ -v` (required after every `analysis/` edit).

The full required column set is listed in `CLAUDE.md` (`h_m, r_m, rel_az_deg, tilt_deg,
convergence_angle_deg, subject_pos_name, mpjpe_mm, pck30, pck50, detection_coverage,
joint_visibility_cam_a/b/either/both, unique_contribution_cam_b, jitter_variance,
id_drops`). If you add a column, also bump anything downstream that reads `results.csv`.

> Confidence filtering and the time-averaging of both skeletons happen in
> `load_gt_average` / `load_pred_average`. Coordinate conversions
> (`fused_to_isaac`, `single_cam_to_isaac`) also live in `metrics.py`.

### Change the sweep algorithm

The search strategy lives **only** in `sweep.py`. The boundary is
`camera_rig.evaluate_layout(h, r, rel_az_deg, subject_pos, cfg, machine, layout_id, ...)`
— `sweep.py` calls only that, and `evaluate_layout` runs the whole pipeline for one layout
and returns the metrics dict.

- **Change the grid:** `config/experiment.yaml → heights_m`, `radii_m`,
  `relative_azimuths`. The generator is `sweep.layouts(cfg)` (an `itertools.product`).
- **Change the search** (e.g. from brute-force grid to coarse-to-fine, Bayesian opt,
  random search): replace the `layouts()` generator and the main loop in `sweep.py`. As
  long as you keep calling `evaluate_layout(...)` and appending to `results.csv`, the rest
  of the pipeline is untouched.
- **Change the pre-filters:** the two gates in `sweep.py` are the **tilt gate**
  (`tilt > max_tilt_deg` → `GEO_SKIPPED`) and the **VRST geometric prescreen**
  (`analysis/geo_prescreener.py`, free, no Isaac). Tune the thresholds in `experiment.yaml`
  (`max_tilt_deg`, `triangulable_min_deg`, `triangulable_max_deg`, the `zed_x` FOV/range),
  or change the prescreen logic in `geo_prescreener.py`.

### Add a third camera

Cameras are wired by the **port ↔ serial** invariant (`cam A = 30000/1001`,
`cam B = 30002/1002`). A third camera = `cam C = 30004/1003`. Touch these places:

1. **`config/experiment.yaml`** — add a `cam_c` block:
   ```yaml
   cam_c:
     serial: 1003
     port:   30004
     # azimuth handled like cam_b: relative to cam A
   ```
2. **`isaac/scene_builder.py`** — `build_scene(...)` currently builds annotator A and B;
   add a third `ZEDAnnotator` (port 30004, serial "1003") and a third `ZED_X` camera prim,
   placed with `camera_rig.camera_position` / `rotation_matrix_from_look_at`. Extend the
   `cams` parameter to allow a third selection.
3. **`zed/make_fusion_config.py`** + **`zed/zed360_template.json`** — add a third pose
   entry (serial 1003 → `127.0.0.1:30004`) so the fusion config places camera C. The
   Isaac→ZED pose conversion is the same as the existing two.
4. **`zed/zed_fusion.py`** — extend `_serial_to_port` and the open/subscribe loop to open
   the third stream and subscribe it to Fusion.
5. **`scripts/preflight.py`** — the port range already covers `30000-30005`, so 30004 is
   included; verify the streaming-ports list it reads from the config includes `cam_c`.
6. **`analysis/metrics.py`** — if you want a per-camera-C visibility metric
   (`joint_visibility_cam_c`, `unique_contribution_cam_c`), add the columns (see
   [metrics](#change--add-metrics)).

The geometry math (`camera_rig.py`) generalizes to any number of cameras on the ring; the
work is plumbing the third port/serial through the four runtime files above.

---

## 9. Metrics reference — what each one means and where it's computed

Every layout produces **one row** of `results/results.csv` with the 17 columns below.
The row is assembled by `analysis/metrics.py → compute_metrics(...)`
([metrics.py:158](analysis/metrics.py#L158)); the column order is fixed by
`RESULTS_COLUMNS` ([metrics.py:229](analysis/metrics.py#L229)) and written append-only by
`append_results_row` ([metrics.py:238](analysis/metrics.py#L238)).

### Two different kinds of metric (important)

- **Accuracy metrics** (`mpjpe_mm`, `pck30`, `pck50`, `detection_coverage`) compare the
  **ZED prediction** against the **Isaac ground truth**. These are the ones that tell you
  how good a layout actually was.
- **Geometry metrics** (`tilt_deg`, `convergence_angle_deg`, all four
  `joint_visibility_*`, `unique_contribution_cam_b`) are computed **purely from the camera
  positions and the ground-truth joint positions** — no ZED prediction involved. They come
  from `analysis/geo_prescreener.py`, the same code used to pre-filter layouts before
  running Isaac. They explain *why* a layout did well or badly (was the joint even visible
  to both cameras? was the triangulation angle good?).

So a layout can have perfect `joint_visibility_both` (geometry says both cameras see every
joint) but poor `mpjpe_mm` (the detector still struggled) — comparing the two columns is
the point of the experiment.

### The accuracy pipeline (how MPJPE/PCK are obtained)

1. **Load + time-average both skeletons.** The character is static, so instead of matching
   timestamps across two processes, each joint is averaged over all frames:
   `load_gt_average` ([metrics.py:49](analysis/metrics.py#L49)) for the Isaac GT CSV,
   `load_pred_average` ([metrics.py:69](analysis/metrics.py#L69)) for the ZED prediction
   (drops keypoints below `--conf` and any `NaN`).
2. **Put the prediction into Isaac's coordinate frame.** Fusion output is converted with
   `fused_to_isaac` ([metrics.py:99](analysis/metrics.py#L99)); single-cam output with
   `single_cam_to_isaac` ([metrics.py:104](analysis/metrics.py#L104)). Mode is chosen at
   [metrics.py:184](analysis/metrics.py#L184).
3. **Map ZED joints to Isaac joints.** `analysis/joint_map.py → mapped_pairs()` pairs the
   ZED `BODY_18` joint names with the rig's joint names (only the ~15 that correspond).
4. **Score.** `mpjpe_pck` ([metrics.py:138](analysis/metrics.py#L138)) computes the
   per-joint Euclidean distance and reduces it to MPJPE + PCK.

### Column-by-column

| Column | What it represents | How it's computed | Where |
|---|---|---|---|
| `h_m` | camera height (m) — experiment input | passed straight through | [metrics.py:204](analysis/metrics.py#L204) |
| `r_m` | camera radius from subject (m) — input | passed through | [metrics.py:205](analysis/metrics.py#L205) |
| `rel_az_deg` | cam B azimuth relative to cam A (°) — input | passed through | [metrics.py:206](analysis/metrics.py#L206) |
| `tilt_deg` | how far each camera tilts down from horizontal | `atan((h − aim_height)/r)` via `camera_rig.tilt_angle` | [metrics.py:207](analysis/metrics.py#L207) |
| `convergence_angle_deg` | whole-body triangulation angle: angle between the cam A→subject and cam B→subject rays. ~90° is ideal for stereo; near 0° (cameras side-by-side) or 180° (opposed) is poor | `camera_rig.convergence_angle` | [geo_prescreener.py:168](analysis/geo_prescreener.py#L168) → [metrics.py:208](analysis/metrics.py#L208) |
| `subject_pos_name` | which subject position was used | label passed through | [metrics.py:209](analysis/metrics.py#L209) |
| `mpjpe_mm` | **Mean Per-Joint Position Error (mm)** — the headline accuracy number. Average Euclidean distance between each predicted joint and its GT joint. Lower is better (current calibrated fusion ≈ 112 mm) | mean over mapped joints of `1000·‖pred − gt‖` | [metrics.py:147-152](analysis/metrics.py#L147-L152) |
| `pck30` | **Percentage of Correct Keypoints @ 30 mm** — fraction of joints predicted within 30 mm of truth. Higher is better (0–1) | `count(err ≤ 30) / n` | [metrics.py:153](analysis/metrics.py#L153) |
| `pck50` | same, threshold 50 mm | `count(err ≤ 50) / n` | [metrics.py:154](analysis/metrics.py#L154) |
| `detection_coverage` | fraction of grabbed frames in which the ZED detected a body at all (detector reliability) | `frames_with_bodies / frames_grabbed` from the receiver's `_meta.json` | [metrics.py:200-201](analysis/metrics.py#L200-L201) |
| `joint_visibility_cam_a` | fraction of GT joints geometrically inside camera **A**'s FOV cone **and** range | `_joint_in_view` per joint, counted | [geo_prescreener.py:169](analysis/geo_prescreener.py#L169) (logic [:63](analysis/geo_prescreener.py#L63)) |
| `joint_visibility_cam_b` | same for camera **B** | as above for B | [geo_prescreener.py:170](analysis/geo_prescreener.py#L170) |
| `joint_visibility_either` | fraction visible to **A or B** (at least one camera) | `in_a or in_b` | [geo_prescreener.py:171](analysis/geo_prescreener.py#L171) |
| `joint_visibility_both` | **VRST triangulable fraction** — joints in **both** cameras' FOV **and** whose per-joint convergence is inside `[triangulable_min_deg, triangulable_max_deg]`. This is the real "can we 3-D this body well" score | `in_a and in_b and (tri_min ≤ conv ≤ tri_max)` | [geo_prescreener.py:159-163, 172](analysis/geo_prescreener.py#L159-L163) |
| `unique_contribution_cam_b` | fraction of joints camera **B** sees that camera **A** does **not** — how much the second camera adds (occlusion recovery). ~0 means the cameras are redundant | `count(in_b and not in_a) / n` | [geo_prescreener.py:157-158, 173](analysis/geo_prescreener.py#L157-L158) |
| `jitter_variance` | temporal stability of joints across frames | **`NaN`** — needs motion; static pose has none | [metrics.py:219](analysis/metrics.py#L219) |
| `id_drops` | how often the tracker loses/swaps the body ID | **`NaN`** — needs motion | [metrics.py:220](analysis/metrics.py#L220) |

> The geometry metrics are produced in one call to `geo_prescreener.prescreen(...)` at
> [metrics.py:197](analysis/metrics.py#L197), but here it is fed the **real averaged GT
> joints** ([metrics.py:196](analysis/metrics.py#L196)) instead of the canonical placeholder
> skeleton it uses during pre-filtering — so the visibility numbers in `results.csv` reflect
> the actual body, not a stand-in.

`compute_metrics` also returns a few **debug-only** values (not written to `results.csv`):
`_n_gt_frames`, `_n_pred_rows`, `_per_joint_mm` (the per-joint error dict — useful to see
*which* joints were bad, e.g. occluded knees), and `_gravity_axis_deg` (a sanity check on
whether the ZED frame was gravity-aligned). See [metrics.py:222-225](analysis/metrics.py#L222-L225).

### When motion is added later

`jitter_variance` and `id_drops` are the only two columns that need animation. `gt_logger`
already logs per frame, so once `omni.anim.people` motion lands you add a per-frame
association pass in `metrics.py` and fill these two in — no other column changes. See
[Change what the people are doing](#change-what-the-people-are-doing-animation).

---

## 10. Coordinate systems (the part that bites you)

There are **two coordinate frames** and mixing them is the classic source of "fused
skeleton is metres off" bugs.

- **Isaac side** (`scene_builder`, `gt_logger`, `camera_rig`): **RIGHT_HANDED_Z_UP**,
  metres.
- **ZED SDK side** (all `pyzed` init, fusion poses): **RIGHT_HANDED_Y_UP**, metres
  (matches the official `fused_cameras.py` sample).

`make_fusion_config.py` converts Isaac Z-up camera poses into the ZED frame for the fusion
file. The empirically-calibrated rule (don't change it without re-verifying MPJPE):

- Runtime pose fusion applies `(P @ R, P @ t)` with `P(x,y,z) = (-y,-z,x)` — **world side
  only** is permuted.
- The **file** stores poses in the ZED *image* frame, so it must contain
  `file_R = D (P R) D` and `file_t = (-y, z, -x)` with `D = diag(1,-1,-1)`.
- The file **pitch is written doubled** (`file pitch = -2 * tilt`) because the SDK adds
  back `+tilt` when it reconciles the virtual IMU's "level" gravity.

This is documented in full in `CLAUDE.md`. The current calibration gives fused
MPJPE ≈ 112 mm.

---

## 11. Outputs / where results land

Everything is under `results/` (append-only — the pipeline never overwrites):

```
results/
├── results.csv                       # one row per evaluated layout (the experiment output)
├── layouts/
│   ├── fusion_config_<id>.json        # per-layout ZED fusion config
│   ├── zed_single_<id>.csv            # single-cam predicted skeleton
│   ├── zed_single_<id>_meta.json      # frames grabbed / with bodies / timing
│   ├── zed_pred_<id>.csv              # fused predicted skeleton
│   └── zed_pred_<id>_meta.json
├── ground_truth_<id>.csv             # Isaac ground-truth skeleton (per frame)
└── logs/
    └── isaac_<id>_<time>.log         # full Isaac stdout for that run (read this when a run fails)
```

When a run fails, the Isaac log in `results/logs/` is the first thing to read — it
contains `STREAMING_STARTED`, the per-port streamer-init lines, the scene-strip summary
(`kept […], deactivated N`), and any USD/skeleton warnings.

---

### Key sentinel lines (handy when grepping logs)

```
PREFLIGHT_OK / PREFLIGHT_FAILED <reason>
STREAMING_STARTED
ZED_SINGLE_READY / FUSION_READY
EPISODE_DONE
PIPELINE_OK / PIPELINE_FAILED
GEO_SKIPPED <layout_id> <reason>
RUN_FAILED <reason>
```
