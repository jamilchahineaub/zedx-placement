#!/usr/bin/env python3
# sweep.py
#
# Top-level layout sweep driver. RUNS UNDER SYSTEM python3 ONLY.
#
# Loops the (h, r, rel_az) grid from config/experiment.yaml, geometric
# pre-filters each layout (tilt gate + VRST prescreen), runs the survivors
# through camera_rig.evaluate_layout (one full Isaac episode + ZED fusion +
# metrics each), and APPENDS one row per layout to results/results.csv.
#
# Usage:
#   python3 sweep.py --machine laptop --limit 3            # mini-sweep gate
#   python3 sweep.py --machine 4090                        # full grid
#
# Search strategy lives HERE only; evaluate_layout is the boundary.

import argparse
import itertools
import os
import sys
import traceback

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, os.path.join(REPO, "isaac"), os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import camera_rig          # noqa: E402
import geo_prescreener     # noqa: E402
import metrics as metrics_mod  # noqa: E402

RESULTS_CSV = os.path.join(REPO, "results", "results.csv")


def layouts(cfg):
    for h, r, az in itertools.product(cfg["heights_m"], cfg["radii_m"],
                                      cfg["relative_azimuths"]):
        yield h, r, az


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="laptop")
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N evaluated layouts (mini-sweep gate: 3)")
    ap.add_argument("--episode-duration", type=float, default=120.0)
    ap.add_argument("--capture-duration", type=float, default=20.0)
    ap.add_argument("--mode", choices=["fusion", "single"], default="fusion")
    args = ap.parse_args()

    with open(os.path.join(REPO, "config", "experiment.yaml")) as f:
        cfg = yaml.safe_load(f)
    subject = next(s["pos"] for s in cfg["subject_positions"]
                   if s["name"] == args.subject_name)

    done = 0
    for h, r, az in layouts(cfg):
        layout_id = f"h{h}_r{r}_az{int(az)}_{args.subject_name}"

        # Gate 1: tilt (pure math, free).
        if not camera_rig.is_valid_layout(h, r, cfg):
            print(f"GEO_SKIPPED {layout_id} tilt")
            continue

        # Gate 2: VRST geometric prescreen (canonical skeleton, free).
        cam_a_az = cfg["cam_a"]["azimuth_deg"]
        pos_a = camera_rig.camera_position(cam_a_az, r, h, subject)
        pos_b = camera_rig.camera_position(cam_a_az + az, r, h, subject)
        pre = geo_prescreener.prescreen(pos_a, pos_b, None, cfg, subject=subject)
        if not pre["passed"]:
            print(f"GEO_SKIPPED {layout_id} prescreen "
                  f"(triangulable={pre['joints_visible_both_triangulable']:.2f})")
            continue

        # Full evaluation: Isaac episode + ZED capture + metrics (~4-6 min).
        print(f"sweep: evaluating {layout_id} ...", flush=True)
        try:
            row = camera_rig.evaluate_layout(
                h, r, az, subject, cfg, machine=args.machine,
                layout_id=layout_id, subject_pos_name=args.subject_name,
                episode_duration=args.episode_duration,
                capture_duration=args.capture_duration, mode=args.mode)
        except Exception as e:
            print(f"RUN_FAILED {layout_id}: {e}")
            traceback.print_exc()
            continue

        metrics_mod.append_results_row(RESULTS_CSV, row)
        done += 1
        print(f"sweep: {layout_id} -> mpjpe={row['mpjpe_mm']:.1f}mm "
              f"({done} rows appended)", flush=True)
        if args.limit and done >= args.limit:
            break

    print(f"sweep: finished, {done} layouts appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
