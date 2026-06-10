# ZED-X Placement Experiment — Handoff
**Last updated:** 2026-06-11 (end of Phase 5)

This file tells you exactly where the project stands and how to pick back up.
The running status/decision log is in `~/.claude/plans/before-u-hit-the-nifty-newt.md`.

---

## TL;DR — where we are

Phases 1–5 of the playbook are **done and the pipeline runs end-to-end in Isaac**.
A 10-second episode booted Isaac, streamed both ZED cameras, and wrote a real
ground-truth joint CSV. 14 unit tests pass. One known issue (frozen character
animation) is intentionally deferred to Phase 6.

**Next session = Phase 6:** the `zed/` body-tracking + fusion scripts, the metrics,
and then `sweep.py`. Plus fix the animation.

---

## What was built, phase by phase

### Phase 1 — ZED helper node type (DONE)
- The playbook said to `grep` `test.usd` for the node type. That FAILED: `test.usd`
  is a **USDC binary crate**, not text. Found the truth in the extension source
  instead.
- Node type is **`sl.sensor.camera.ZED_Camera`** (typeVersion 2, uiName "ZED Camera
  Helper"). The playbook's guessed attribute names were wrong; corrected them.
- Key discovery: that helper node has **no serial field** — cameras are told apart
  by `streamingPort` (30000 / 30002) + `cameraPrim`. Serials 1001/1002 live only in
  the fusion config, bound to ports.
- Wrote into `config/experiment.yaml`: `zed_helper_node_type`, `zed_helper_attrs`,
  and `zed_stream` (HD1080 @ 30 fps — your correction). Updated `CLAUDE.md`.

### Phase 2 — `isaac/camera_rig.py` (DONE, 9 tests)
- Pure math, no Isaac imports. `camera_position`, `rotation_matrix_from_look_at`
  (copied verbatim from `make_fusion_config.py` so they never diverge),
  `convergence_angle`, `tilt_angle`, `is_valid_layout`, and the `evaluate_layout`
  STUB (the sweep boundary — raises NotImplementedError until Phase 6).
- Caught a kickoff error: convergence at 90° azimuth is only 90° when cameras are at
  subject height; with cameras above the subject it's 78.46°. Tests reflect the real
  physics.

### Phase 3 — `analysis/geo_prescreener.py` (DONE, 5 tests)
- VRST two-camera triangulable pre-filter. `prescreen(...)` returns visibility +
  triangulability fractions; passes if ≥70% of joints are triangulable by both cams.
- `CANONICAL_SKELETON` constant at the top (14 joints, 1.75 m) — **placeholder, to be
  swapped for real GT joints in Phase 6** (per your instruction; comment is in the
  file).

### Phase 4 — `isaac/scene_builder.py` + fusion-pose coordinate fix (DONE)
- `build_scene(h, r, rel_az, subject_name, cfg, machine_cfg)` → boots SimulationApp,
  enables the ZED ext, builds a textured room (floor/4 walls/2 dome lights/table/
  pillar), references the character, places two `ZED_X.usdc` cameras, and starts
  streaming via **two `ZEDAnnotator`s** (NOT a hand-built ActionGraph).
- **Major fix to `zed/make_fusion_config.py`:** the ZED fusion file is read in
  **RIGHT_HANDED_Y_UP** (confirmed against the official ZED multi-cam sample), but the
  code was writing Isaac Z-up poses with NO conversion — wrong camera poses, garbage
  wide-baseline fusion. Added `convert_isaac_to_zed_pose` (translation `(-y,-z,x)`,
  rotation `P·R·Pᵀ`) and `proper_rotation_world_from_cam` (rebuilds a det=+1 rotation,
  because `camera_rig`'s look-at returns an improper det=−1 matrix). Verified: new
  poses are valid rigid transforms and differ from the old ones.
- Coordinate rule is now two-frame: **Isaac side Z-up, ZED side Y-up.** See CLAUDE.md.

### Phase 5 — `isaac/run_episode.py` + `isaac/gt_logger.py` (DONE — ran live)
- `run_episode.py`: CLI → `is_valid_layout` gate (GEO_SKIPPED if bad) → build_scene →
  timeline play → per-frame joint logging → save CSV → EPISODE_DONE → cleanup.
- `gt_logger.py`: finds the `UsdSkel.Root`/Skeleton, logs every joint's WORLD position
  each frame. Confirmed the exact Isaac 5.1 API from compiled symbols:
  `SkeletonQuery.ComputeJointWorldTransforms(UsdGeom.XformCache(timecode))` — the
  kickoff's `ComputeJointWorldTransforms(time)` was wrong (needs an XformCache, and
  the time is a USD timecode = seconds × TimeCodesPerSecond, not raw seconds).
- Two bugs found & fixed during the live run: USD token `strongerThanDescendants`
  (plural), and the seconds→timecode conversion.

---

## The live run result (proof it works)

```
[Port: 30000] Constructed annotator for stereo camera.
[Port: 30002] Constructed annotator for stereo camera.
STREAMING_STARTED
Initializing streamer ... on port 30000 / 30002   (both streaming, IPC backend)
gt_logger: skeleton /World/biped_demo_meters/.../Root with 81 joints
gt_logger: wrote 10206 rows (81 joints) -> results/layouts/ground_truth_test_001.csv
EPISODE_DONE
```
CSV verified: 81 joints, 126 frames, pelvis z = 0.957 m (standing ✓).

---

## ⚠️ KNOWN ISSUE — character is frozen (deferred to Phase 6, your call)

All 81 joints hold the SAME position across all 126 frames — the biped is static, not
animating. Cause (from the console):
```
Warning: Root.skel:animationSource -- Invalid target
  </World/biped_demo_meters/biped_demo_meters/Root/Biped_Demo>
```
Referencing the WHOLE `test.usd` double-nested the prim
(`/World/biped_demo_meters/biped_demo_meters/Root`) and broke the skeleton's relative
`animationSource`. Phase 5's gate doesn't require motion, so we deferred it.

**Fix options for Phase 6** (in `scene_builder._load_character`):
- (a) Open `test.usd` read-only, use `Usd.PrimCompositionQuery` on
  `/World/biped_demo_meters` to get the real character asset path, and reference THAT
  (single nesting → animationSource target stays valid). ← recommended
- (b) Open `test.usd` AS the working stage and layer the room + cameras on top.

This matters in Phase 6 because jitter/MPJPE metrics are meaningless on a static pose.

---

## HOW TO PICK BACK UP TOMORROW

1. **Sanity-check the unit tests still pass** (fast, no Isaac needed):
   ```bash
   cd /home/jimmy/zedx-placement
   python3 -m pytest analysis/tests/ -v        # expect 14 passed
   ```

2. **Re-run the proven episode** to confirm the Isaac pipeline still works:
   ```bash
   /home/jimmy/isaacsim/python.sh isaac/run_episode.py \
     --h 1.5 --r 2.5 --rel-az 90 --subject-name center \
     --duration 10 --layout-id test_001 --machine laptop
   ```
   Watch for STREAMING_STARTED → EPISODE_DONE and the CSV.

3. **Tell Claude: "start Phase 6"** — and that the animation fix happens there.

---

## NEXT STEPS — Phase 6 (and beyond)

Phase 6 is the body-tracking + metrics half. Order:

1. **Fix character animation** in `scene_builder._load_character` (option a above) so
   joints actually move. Re-run the episode; confirm joints differ across frames.

2. **`zed/zed_single.py`** — single-camera ZED body tracking. MUST:
   - read pyzed member names from `/usr/local/zed/samples/` first (they drift),
   - use `init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP` (Y-up!).

3. **`zed/zed_fusion.py`** — subscribe both cameras using the fusion config from
   `make_fusion_config.py`, write fused skeleton to `zed_pred_{id}.csv`. Print
   FUSION_READY. Mirror the official sample:
   `/usr/local/zed/samples/body tracking/multi-camera/python/fused_cameras.py`.

4. **`analysis/joint_map.py`** — map ZED joint names ↔ the 81 Isaac joint names from
   the GT CSV (e.g. ZED "PELVIS" ↔ Isaac "Pelvis"). Build after you can see both CSVs.

5. **`analysis/metrics.py`** — MPJPE, PCK30/50, coverage, jitter, id_drops, etc.
   (the full results.csv column list is in CLAUDE.md). Gate: MPJPE 20–200 mm,
   unique_contribution_cam_b > 0 at 90°.

6. **Fill in `camera_rig.evaluate_layout`** — wire build→stream→fuse→metrics into the
   one function `sweep.py` calls. Replace the prescreener's `CANONICAL_SKELETON` with
   real GT joints now that a CSV exists.

7. **`sweep.py`** — loop the 240 valid layouts, call `evaluate_layout`, append rows to
   `results/results.csv`. Gate: a 3-layout mini-sweep produces 3 rows.

Then: commit, push, and deploy to the 4090 (only `config/machine.4090.yaml` changes,
with `headless: true`).

---

## File map (current)

```
isaac/camera_rig.py      Phase 2  pure math + evaluate_layout STUB
isaac/scene_builder.py   Phase 4  room + cameras + ZEDAnnotators
isaac/gt_logger.py       Phase 5  skeleton -> world joints -> CSV
isaac/run_episode.py     Phase 5  episode orchestrator (CLI)
isaac/probe_node_types.py         ABANDONED (playbook says don't fix; deletable)
analysis/geo_prescreener.py  Phase 3  VRST prefilter + CANONICAL_SKELETON
analysis/tests/          14 tests, all green
zed/make_fusion_config.py    Phase 4 fix: Y-up pose conversion
zed/zed360_template.json     done
zed/generate_fusion_template.py  done
config/experiment.yaml   sweep grid + zed_helper_* + occluders + character_prim
config/machine.laptop.yaml   paths + reference_scene + headless:false
CLAUDE.md                project rules (coordinate section corrected)
results/layouts/ground_truth_test_001.csv   proof-of-life GT data
```

## Two runtimes — never mix
- `isaac/`  → `/home/jimmy/isaacsim/python.sh`
- `zed/`, `analysis/`, `sweep.py` → `python3`
