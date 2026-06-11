# ZED-X Placement Experiment — Handoff
**Last updated:** 2026-06-11 (Phase 6, session 3 — STREAMING + DETECTION + FUSION POSES FIXED)

---

## ⚡ SESSION 3 (2026-06-11 afternoon) — the 2-hour sprint

**One command now runs the whole pipeline:**
```bash
python3 scripts/run_pipeline.py --h 1.5 --r 2.5 --rel-az 90 --subject-name center \
  --layout-id my_run --machine laptop --mode both
# -> PREFLIGHT_OK -> STREAMING_STARTED -> ZED_SINGLE_READY -> FUSION_READY -> PIPELINE_OK
```

### Issue C — ROOT-CAUSED AND FIXED (0 frames on port 30000)
Three stacked problems, all fixed:
1. **Stale machine state**: zombie receivers/crashed runs squat UDP 30000-30003 and
   leave stale `/dev/shm/sl_local_*` topics → receiver misses the SHM topic, falls
   into the UDP cascade (which can NEVER work locally — the sender holds those
   ports). Fix: `scripts/preflight.py` (kill stale procs, wait ports free, clean
   sl_local SHM) runs before every pipeline run.
2. **Timing**: the receiver must start while Isaac is actively streaming.
   `scripts/run_pipeline.py` waits for both "Initializing streamer" lines + a 5s
   no-error grace, then starts the receiver. Result: first frame in **0.1s**.
3. **zed.open() hangs >2min on a dead stream and is NOT interruptible** — fixed
   with a SIGALRM hard cap (`--open-timeout`, exit 142) + a first-frame watchdog
   (`RUN_FAILED stream_dead`, exit 2) so the orchestrator can retry fast.

### Detection — bodies=0 with FAST/conf40, FIXED with ACCURATE/conf20
On our procedural room rendering, HUMAN_BODY_FAST detects nothing;
HUMAN_BODY_ACCURATE + conf 20 detects the cop on EVERY frame (verified: 78/78,
then 147/147). Defaults flipped accordingly (zed_single, zed_fusion,
run_pipeline). `--save-frame` on zed_single captures what the camera sees
(verified aim is correct — user's pitch fix works).

### test.usd "warehouse merge" — FIXED with dynamic strip
test.usd now contains `warehouse_with_forklifts` + a `ZED_X` rig + the cop.
`scene_builder._load_character` now keeps ONLY subtree(s) containing a
UsdSkel.Root/Skeleton and session-layer-deactivates everything else — robust to
whatever test.usd is re-saved with. Log line: `scene_builder: kept [...]`.

### Fusion pose conversion — the old rule was WRONG (both ours and the
### zed-isaac-sim convert script). Fixed empirically, in three layers.
Symptom: fused bodies=2 (views never merged), both metres from GT. Found by
printing `conf.pose` after `read_fusion_configuration_file` and solving against
real measured camera-frame keypoints (incl. a Kabsch fit of the transform the
SDK ACTUALLY applied):
1. Runtime pose fusion applies = **(P @ R, P @ t)** — world-side permutation
   ONLY (P R Pᵀ also permutes the camera axes; wrong).
2. The fusion FILE stores poses in the ZED IMAGE frame (y down, z forward);
   the reader conjugates with D = diag(1,-1,-1) on load. File must contain
   **D (P R) D** and **t = (-y, z, -x)**.
3. **Pitch must be written DOUBLED: file pitch = -2*tilt.** Four controlled
   runs show fusion applies `pitch = file_pitch + tilt` (linear, slope 1) —
   the SDK adds +tilt while reconciling the pose with the virtual IMU's
   "camera is level" gravity. Neither `override_gravity` (true/false) nor
   `enable_imu_fusion=False` nor sender `initial_world_transform` changes
   this (all tested — initial_world_transform is ignored for stream input).
   make_fusion_config now aims each camera's file rotation at the MIRROR of
   the aim point through the camera's horizontal plane (pitch = -2*tilt).
   **Result: fused MPJPE 112.3 mm** (hips/knees 66-98 mm — fusion recovers
   the legs that the table occludes from cam A; wrists/elbows 138-205 mm).
   Inside the 20-200 mm Phase-6 gate. NOTE: the +tilt addition was measured
   at tilt=11.3deg; if a future SDK/Isaac update changes IMU handling,
   re-run the 3-point calibration (file pitch -tilt / 0 / -2tilt).
Also: zed_fusion exits via os._exit right after artifacts are flushed and
never closes the cameras — the SDK segfaults (rc=-11) closing cameras while
Fusion is subscribed; process death reclaims everything and preflight cleans
any residue.
See CLAUDE.md coordinate section + make_fusion_config.convert_isaac_to_zed_pose.

### What landed (files)
- NEW `scripts/preflight.py`, `scripts/run_pipeline.py` — cleanup gate + one-command
  orchestrator (importable building blocks; evaluate_layout uses them).
- NEW `zed/zed_fusion.py` — dual-cam fusion (mirrors fused_cameras.py; opens both
  streams in one process, SHM publish/subscribe, subscribes with RUNTIME serials,
  poses matched by port). FUSION_READY confirmed live.
- NEW `analysis/joint_map.py` (BODY_18 LEFT_*/RIGHT_* names verified from real CSV
  → male_adult_police_04 rig) + `analysis/metrics.py` (MPJPE/PCK30/50/coverage/
  visibility via prescreen on real GT joints; static ⇒ time-averaged skeletons;
  jitter/id_drops NaN) + 12 new tests. **26/26 pytest green.**
- `camera_rig.evaluate_layout` IMPLEMENTED (preflight → fusion config → episode →
  receiver → metrics; python3 only).
- `zed/zed_single.py` hardened (HD1080+NEURAL init like the proven sample,
  open-retry, alarm, watchdog, meta JSON sidecar, frame_idx column, --save-frame).
- Single-cam real-data metrics (h1.5 r2.5 az90): upper body 57-106mm; knees
  ~900mm because the TABLE OCCLUDER hides the legs from cam A (visible in the
  frame capture) — genuine occlusion signal, the thing this experiment measures.

### Still open
- Mid-episode Isaac death rc=-15 seen once (likely the viewport window being
  closed by hand) — GT CSV is only written at episode end; don't close the window.
- Character animation still frozen (omni.anim.people follow-up, unchanged).
- sweep.py not started (next: thin loop over layouts calling evaluate_layout).

This file tells you exactly where the project stands and how to pick back up.
The running status/decision log is in `~/.claude/plans/before-u-hit-the-nifty-newt.md`.

---

## TL;DR — where we are

Phases 1–5 of the playbook are **done and the pipeline runs end-to-end in Isaac**.
A 10-second episode booted Isaac, streamed both ZED cameras, and wrote a real
ground-truth joint CSV. 14 unit tests pass.

Phase 6 is in progress. `isaac/scene_builder.py` has uncommitted WIP changes
(not yet re-tested in Isaac):
- 4-camera bug fix: deactivates `test.usd`'s own `ZED_X`/`ZED_X_01`/`GroundPlane`/
  `DomeLight`/`ActionGraph` prims on the session layer after referencing it.
- Gravity disabled via `/World/PhysicsScene` (`CreateGravityMagnitudeAttr(0.0)`).
- **Camera orientation fix (2026-06-11, this session):** `_rotmat_to_wxyz` had a
  sign bug — `pitch = asin(fz)` instead of `pitch = asin(-fz)`. With the
  `Rz(yaw) * Ry(pitch)` quaternion applied to Isaac's default forward axis (+X),
  `forward.z = -sin(pitch)`, so `pitch` must be `asin(-fz)` (positive pitch =
  tilt down) to match the look-at forward vector `fz`. Fixed; yaw formula
  (`atan2(fy, fx)`) was already correct. **Not yet verified live in Isaac** —
  user is running a quick experiment to check body tracking / camera aim next.

`zed/zed_single.py` (single-camera body tracking, BODY_18, HUMAN_BODY_ACCURATE,
RIGHT_HANDED_Y_UP) is written but untracked/untested.

Known issue (frozen character animation) may already be addressed by the
session-layer deactivation above (Option B variant) — needs re-verification.

### Issue B — root cause found and fixed (2026-06-11, this session)

It was NOT `/dev/shm` SHM slots (those carb-RStringInternals files are normal
Carbonite leftovers, harmless). The real cause: Isaac's
"Error: failed to create RTP Session (err:-74)" / "Error during zed streamer
initialization 0" looping on port 30000 was because **two zombie
`zed_single.py` processes from earlier test runs were still holding UDP sockets
30000 and 30002** (`ss -tulnp` showed both bound by stale `python3
zed/zed_single.py` PIDs).

Mechanism: `zed.grab(runtime)` blocks indefinitely if the Isaac SHM stream never
arrives. The `--duration` check only runs at the top of the while loop, so a
blocked `grab()` means the loop never times out, `finally` is never reached, and
the process never calls `zed.close()` — permanently squatting the port.

Fixes applied:
- Killed the two stale processes (PIDs were on ports 30000/30002) — ports now
  free.
- `zed/zed_single.py`: added a daemon watchdog thread (`threading`, stdlib only)
  that calls `zed.close()` after `--duration` to force an in-progress `grab()`
  to return, so the loop always exits and `finally` always runs. `finally` now
  wraps `disable_body_tracking()`/`close()` in try/except (camera may already be
  closed by the watchdog) so cleanup never masks the real error and the CSV is
  always written.

Still TODO for issue B: a general cleanup script for stale SHM/ports between
Isaac runs would still be useful if Isaac itself crashes mid-run, but the
immediate blocker (zombie zed_single.py processes) is fixed at the source.

**Next:** re-run the proven episode command, confirm camera orientation looks
correct (cameras pointing at hip, not ceiling), confirm joints animate, then
re-test `zed_single.py` on port 30000 now that the port is free and the watchdog
fix is in place. Then continue to `zed/zed_fusion.py`, `analysis/joint_map.py`,
`analysis/metrics.py`, `evaluate_layout`, `sweep.py`.

### Re-run result (2026-06-11, this session) — `results/layouts/ground_truth_test_002.csv`

Ran `--h 1.5 --r 2.5 --rel-az 90 --subject-name center --duration 10
--layout-id test_002 --machine laptop` with the port-squatting fix + pitch-sign
fix in place.

- **Streaming: clean.** Both `[Port: 30000]` and `[Port: 30002]` initialized
  with no RTP/streamer errors (`Initializing streamer with ID 0 on port 30000`,
  `ID 1 on port 30002` — no "failed to create RTP Session" loop). Confirms the
  zombie-process fix for issue B works.
- **gt_logger: 101 joints**, 161 frames over ~2.7s sim time, written to
  `ground_truth_test_002.csv`. Skeleton path is now
  `/World/biped_demo_meters/male_adult_police_04/ManRoot/male_adult_police_04/
  male_adult_police_04/male_adult_police_04` — the reference test.usd character
  is `male_adult_police_04` (101 joints), NOT the old `Biped_Demo` (81 joints).
  Only `_PRIMS_TO_DEACTIVATE` entry `/World/biped_demo_meters/ZED_X` matched
  (`ZED_X_01`/`GroundPlane`/`DomeLight`/`ActionGraph` were not found at those
  paths for this character — IsValid() was false, so they were silently
  skipped). May need to re-grep the live stage hierarchy for this character's
  actual duplicate-prim paths if a 2nd camera/light shows up unexpectedly.
- **Animation: STILL FROZEN.** `Pelvis` x/y/z is bit-for-bit identical
  (z=0.9277017512507277 m, standing) across all 161 frames. The session-layer
  deactivation added this session fixed the duplicate-camera bug but did NOT
  fix the `animationSource` double-nesting problem — Phase 6 step 1 (fix
  character animation, options (a)/(b) in the "Known issue" section above) is
  still outstanding.
- Camera orientation (pitch-sign fix) was NOT visually confirmed — Isaac ran
  with `headless: false` (DISPLAY=:1) but the run finished/closed before
  visual inspection. User wants to do a hands-on experiment next: run a
  longer episode and use `zed_single.py` against port 30000 (now free) to
  check both camera aim and body-tracking detection live.

### Issue C (NEW, 2026-06-11) — zed_single.py gets 0 rows on port 30000

Ran `--duration 60 --layout-id test_003` (Isaac), then `zed_single.py --port
30000 --layout-id test_003 --duration 30`. Isaac's two streamers initialized
cleanly (`ID 0 on port 30002`, `ID 1 on port 30000`, no RTP errors — issue B
fix holds). But `zed_single.py` printed:

```
[Streaming] Warning : receiving port 30000 is not available (already used)... switching to port 30002. Retrying...
[Streaming] Warning : receiving port 30002 is not available (already used)... switching to port 30004. Retrying...
[Streaming] Backward compatibility required.
```

zed.open() succeeded (S/N 49123828, simulated ZED X, NEURAL depth), body
tracking enabled, but **0 frames ever grabbed successfully** in 30s (no
`[diag]` lines, no `ZED_SINGLE_READY`), 0 rows written. Isaac's clock is ~3h
behind the host clock (its log timestamps read ~10:01-10:02Z while the host
was at ~13:01-13:03 local), so there WAS an ~8s overlap where Isaac was still
live — yet still 0 frames. This points to the SDK falling into "Backward
compatibility" mode on port 30004 (plain network receive), which nothing is
sending to — Isaac's SHM-Boost sender is on 30000/30002, not 30004.

Also found: `/usr/local/zed/samples/body tracking/body tracking/python/
body_tracking.py` line 84 has been hand-edited (outside the repo, pre-existing
from an earlier session) to hardcode `init_params.set_from_stream("127.0.0.1",
30000)  # point to Isaac Sim stream` — this is probably what was used for the
manual ZED_Depth_Viewer-style test the user ran, where port 30002 reportedly
showed an image but 30000 didn't. Not yet reconciled with the "receiving port
not available" cascade above (same cascade should, by the +2 fallback logic,
affect both 30000 and 30002 the same way — needs a controlled side-by-side
test).

**Next:** run a longer Isaac episode and start `zed_single.py` on BOTH ports
30000 and 30002 immediately once streaming is confirmed live, to compare
behavior side by side and isolate whether port 30000 is uniquely broken or if
this is a timing/race condition with the SHM-Boost backend.

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
