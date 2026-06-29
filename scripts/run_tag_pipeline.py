#!/usr/bin/env python3
# scripts/run_tag_pipeline.py
#
# One-layout ArUco TAG-DETECTION run. RUNS UNDER SYSTEM python3.
#   preflight -> Isaac episode (3 ring cameras + chest/back tags + spin, streaming)
#   -> zed_tag_detect (cv2.aruco on each stream) -> tag_metrics (visibility ratio).
# Reuses scripts/run_pipeline.py building blocks; does not touch its main().
#
# Usage:
#   python3 scripts/run_tag_pipeline.py --h 3.0 --r 3.5 --rel-az 180 --cam-c-az 90 \
#       --layout-id tagtest --machine 4090 --spin-deg-s 30

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "analysis"), os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import run_pipeline as rp           # noqa: E402
from scripts.preflight import load_cfgs, preflight  # noqa: E402
from scripts.view_tag import gen_marker, MARKER_FRONT, MARKER_BACK  # noqa: E402
import tag_metrics  # noqa: E402


def ensure_markers(cfg):
    aru = cfg.get("aruco") or {}
    if not os.path.exists(MARKER_FRONT):
        gen_marker(MARKER_FRONT, marker_id=int(aru.get("marker_id_front", 23)))
    if not os.path.exists(MARKER_BACK):
        gen_marker(MARKER_BACK, marker_id=int(aru.get("marker_id_back", 42)))


def run_one(h, r, rel_az, cam_c_az, layout_id, machine, spin_deg_s=30.0,
            subject_name="center", episode_duration=120.0, capture_duration=20.0,
            skip_preflight=False, attempts=2):
    """Returns the tag-metrics dict (or None on failure). Appends to results/tag_detect.csv.
    Retries the whole episode `attempts` times to ride out the ZED stream-startup race (a
    3rd-camera open can hang -> SIGALRM kills the receiver; same reason sweep.py retries)."""
    cfg, machine_cfg = load_cfgs(machine)
    ensure_markers(cfg)
    aru = cfg.get("aruco") or {}
    expected = {cfg["cam_a"]["port"], cfg["cam_b"]["port"], cfg["cam_c"]["port"]}

    res = None
    for attempt in range(1, attempts + 1):
        if not skip_preflight and not preflight(cfg, machine_cfg):
            print("TAG_PIPELINE_FAILED preflight")
            return None
        print(f"run_tag_pipeline: {layout_id} (attempt {attempt}/{attempts}) ...", flush=True)
        proc, log_path = rp.launch_isaac(
            h, r, rel_az, subject_name, layout_id, machine, machine_cfg, episode_duration,
            ring_c_az=cam_c_az, chest_tags=True, marker_front=MARKER_FRONT, marker_back=MARKER_BACK,
            spin_deg_s=spin_deg_s)
        res = None
        try:
            rp.wait_for_streaming(log_path, proc, expected)
            rp.shm_snapshot()
            res = rp.run_zed_tag_detect(layout_id, machine_cfg, duration=capture_duration)
            rp.wait_episode_done(log_path, proc, timeout=episode_duration + 60)
        except RuntimeError as e:
            print(f"run_tag_pipeline: streaming/episode error: {e}")
        finally:
            rp.shutdown_isaac(proc, log_path, grace=90)
        if res is not None and res["rc"] == 0 and res["rows"] > 0:
            break
        print(f"run_tag_pipeline: attempt {attempt}/{attempts} failed "
              f"(rc={res and res['rc']} rows={res and res['rows']})", flush=True)
        if attempt < attempts:
            time.sleep(10)   # let Isaac/ZED fully exit + ports free before retry

    if res is None or res["rc"] != 0 or res["rows"] == 0:
        print(f"TAG_PIPELINE_FAILED detect after {attempts} attempts")
        return None

    m = tag_metrics.compute(res["csv"], int(aru.get("marker_id_front", 23)),
                            int(aru.get("marker_id_back", 42)))
    if m is None:
        return None
    m.update({"h_m": h, "r_m": r, "rel_az_deg": rel_az, "cam_c_az_deg": cam_c_az,
              "spin_deg_s": spin_deg_s, "subject_pos_name": subject_name})
    tag_metrics.append_row(os.path.join(REPO, "results", "tag_detect.csv"), m)
    print(f"TAG_PIPELINE_OK {layout_id} visibility_ratio={m['tag_visibility_ratio']:.3f} "
          f"(A={m['detect_rate_cam_a']:.2f} B={m['detect_rate_cam_b']:.2f} C={m['detect_rate_cam_c']:.2f})")
    return m


def main():
    ap = argparse.ArgumentParser(description="One-layout ArUco tag-detection run")
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--rel-az", type=float, required=True)
    ap.add_argument("--cam-c-az", type=float, required=True)
    ap.add_argument("--layout-id", required=True)
    ap.add_argument("--machine", default="4090")
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--spin-deg-s", type=float, default=30.0)
    ap.add_argument("--episode-duration", type=float, default=120.0)
    ap.add_argument("--capture-duration", type=float, default=20.0)
    ap.add_argument("--attempts", type=int, default=2, help="retry the episode on a stream race")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()
    m = run_one(args.h, args.r, args.rel_az, args.cam_c_az, args.layout_id, args.machine,
                spin_deg_s=args.spin_deg_s, subject_name=args.subject_name,
                episode_duration=args.episode_duration, capture_duration=args.capture_duration,
                skip_preflight=args.skip_preflight, attempts=args.attempts)
    sys.exit(0 if m else 1)


if __name__ == "__main__":
    main()
