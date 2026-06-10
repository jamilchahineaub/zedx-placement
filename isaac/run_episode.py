#!/usr/bin/env python3
# isaac/run_episode.py
#
# Orchestrates ONE episode. RUNS UNDER ISAAC PYTHON ONLY:
#   /home/jimmy/isaacsim/python.sh isaac/run_episode.py --h ... --r ... ...
#
# Flow:
#   1. Load machine + experiment config
#   2. is_valid_layout gate  -> GEO_SKIPPED + exit 0 if invalid
#   3. build_scene(...)      -> (app, stage, annotator_a, annotator_b)
#   4. Start gt_logger + timeline
#   5. Tick loop for --duration s: app.update(), log joints each frame
#   6. Stop, save ground_truth CSV, EPISODE_DONE, destroy annotators, close app
#
# Sentinels (sweep.py keys off these): STREAMING_STARTED (from scene_builder),
# EPISODE_DONE, GEO_SKIPPED <layout_id> <reason>, RUN_FAILED <reason>.

import argparse
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)   # camera_rig, scene_builder, gt_logger


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, required=True, help="camera height (m)")
    ap.add_argument("--r", type=float, required=True, help="camera radius (m)")
    ap.add_argument("--rel-az", type=float, required=True, help="cam B rel azimuth (deg)")
    ap.add_argument("--subject-name", default="center", help="name from experiment.yaml")
    ap.add_argument("--duration", type=float, default=30.0, help="episode seconds")
    ap.add_argument("--layout-id", required=True, help="string id for output filenames")
    ap.add_argument("--machine", default="laptop", help="laptop | 4090")
    args = ap.parse_args()

    # 1) Config
    cfg = load_yaml(os.path.join(REPO, "config", "experiment.yaml"))
    machine_cfg = load_yaml(os.path.join(REPO, "config", f"machine.{args.machine}.yaml"))

    # 2) Validity gate (pure math, no Isaac needed) — do it BEFORE booting Isaac.
    import camera_rig
    if not camera_rig.is_valid_layout(args.h, args.r, cfg):
        tilt = camera_rig.tilt_angle(args.h, args.r, cfg["aim_height_m"])
        print(f"GEO_SKIPPED {args.layout_id} tilt={tilt:.1f}deg>=max_tilt_deg={cfg['max_tilt_deg']}")
        sys.exit(0)

    # 3) Build the scene (this boots SimulationApp and prints STREAMING_STARTED).
    import scene_builder
    try:
        app, stage, annotator_a, annotator_b = scene_builder.build_scene(
            args.h, args.r, args.rel_az, args.subject_name, cfg, machine_cfg
        )
    except Exception as e:
        print(f"RUN_FAILED build_scene raised: {e}")
        raise

    # 4) gt_logger + timeline
    import omni.timeline
    from gt_logger import GTLogger

    logger = GTLogger(stage)

    # Stage timecodes/sec — needed to convert seconds -> USD timecode for joint sampling.
    tcps = stage.GetTimeCodesPerSecond() or 60.0

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_looping(True)
    timeline.play()
    app.update()

    # 5) Tick loop
    start_wall = time.time()
    csv_path = os.path.join(REPO, "results", "layouts", f"ground_truth_{args.layout_id}.csv")
    try:
        while True:
            elapsed = time.time() - start_wall
            if elapsed >= args.duration:
                break
            app.update()
            sim_seconds = timeline.get_current_time()
            timecode = sim_seconds * tcps
            # log_frame samples joints at the given USD timecode; sim_seconds stored in CSV.
            logger.log_frame_at(sim_seconds, time.time(), timecode)
    except Exception as e:
        print(f"RUN_FAILED tick loop raised: {e}")
        timeline.stop()
        try:
            annotator_a.destroy(); annotator_b.destroy()
        finally:
            app.close()
        raise

    # 7) Stop sim
    timeline.stop()

    # 8) Save GT CSV
    logger.save(csv_path)

    # 9) Sentinel
    print("EPISODE_DONE", flush=True)

    # 10) Cleanup annotators, 11) close app
    try:
        annotator_a.destroy()
        annotator_b.destroy()
    finally:
        app.close()


if __name__ == "__main__":
    main()
