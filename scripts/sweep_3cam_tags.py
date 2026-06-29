#!/usr/bin/env python3
# scripts/sweep_3cam_tags.py
#
# UNIFIED 3-camera sweep: body pose + ArUco tag visibility + camera convergence in one row.
# RUNS UNDER SYSTEM python3 ONLY.
#
# For each top-N ring layout (reports/ranking_walk.csv), the 3rd camera sits on the SAME ring
# (same height + radius) and we sweep its AZIMUTH only. Candidate azimuths are the geometric
# best for TWO-SIDED tag coverage, EXCLUDING angles within --min-sep of cam A or B (which make
# cam C redundant with a ring camera). We run --azimuths real episodes per layout so you see
# how pose + tag-ratio move with cam-C angle.
#
# Each episode: ring cam C + chest/back tags + spin; the fusion receiver does body tracking AND
# cv2.aruco, so each row carries mpjpe + convergence_ab/ac/bc + tag_visibility_ratio.
# Appends to results/results_walk_3cam_tags.csv (~top-15 x 4 azimuths ≈ 60 runs, ~3-4 h).
#
# Usage:
#   python3 scripts/sweep_3cam_tags.py --machine 4090                 # top-15 x 4 azimuths
#   python3 scripts/sweep_3cam_tags.py --machine 4090 --limit 1       # one layout (4 azimuths)
#   python3 scripts/sweep_3cam_tags.py --machine 4090 --azimuths 1    # 1 best azimuth/layout
#   python3 scripts/sweep_3cam_tags.py --cam-c-az 90                  # force one angle for all

import argparse
import os
import sys
import time
import traceback

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "isaac"), os.path.join(REPO, "analysis"),
           os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import camera_rig                       # noqa: E402
import metrics as metrics_mod           # noqa: E402
import tag_visibility as tv             # noqa: E402
from scripts.sweep_tag import top_layouts          # noqa: E402
from scripts.view_tag import gen_marker, MARKER_FRONT, MARKER_BACK  # noqa: E402

RESULTS_CSV = os.path.join(REPO, "results", "results_walk_3cam_tags.csv")
DEFAULT_RANKING = os.path.join(REPO, "reports", "ranking_walk.csv")


def ensure_markers(cfg):
    aru = cfg.get("aruco") or {}
    if not os.path.exists(MARKER_FRONT):
        gen_marker(MARKER_FRONT, marker_id=int(aru.get("marker_id_front", 23)))
    if not os.path.exists(MARKER_BACK):
        gen_marker(MARKER_BACK, marker_id=int(aru.get("marker_id_back", 42)))


def main():
    ap = argparse.ArgumentParser(description="Unified 3-cam sweep (pose + tags + convergence)")
    ap.add_argument("--machine", default="4090")
    ap.add_argument("--ranking", default=DEFAULT_RANKING)
    ap.add_argument("--out", default=RESULTS_CSV)
    ap.add_argument("--top", type=int, default=15, help="top-N 2-cam layouts to use")
    ap.add_argument("--limit", type=int, default=None, help="only the first N layouts")
    ap.add_argument("--azimuths", type=int, default=4, help="cam-C azimuths to try per layout")
    ap.add_argument("--min-sep", type=float, default=45.0,
                    help="min azimuth separation (deg) of cam C from cam A and cam B")
    ap.add_argument("--cam-c-az", type=float, default=None,
                    help="force this ring cam C azimuth for all layouts (else geometric two-sided best)")
    ap.add_argument("--spin-deg-s", type=float, default=30.0)
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--episode-duration", type=float, default=120.0)
    ap.add_argument("--capture-duration", type=float, default=20.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(REPO, "config", "experiment.yaml")))
    ensure_markers(cfg)
    subject = next(s["pos"] for s in cfg["subject_positions"] if s["name"] == args.subject_name)

    if not os.path.exists(args.ranking):
        print(f"sweep_3cam_tags: ranking not found: {args.ranking}")
        sys.exit(1)
    layouts = top_layouts(args.ranking, args.top)
    if args.limit:
        layouts = layouts[:args.limit]

    total = len(layouts) * (1 if args.cam_c_az is not None else args.azimuths)
    print(f"sweep_3cam_tags: {len(layouts)} layouts x "
          f"{1 if args.cam_c_az is not None else args.azimuths} cam-C azimuths "
          f"(same ring, two-sided geometric pick, >= {args.min_sep:.0f}° from A/B) "
          f"-> up to {total} runs, spin {args.spin_deg_s}°/s")
    done = 0
    for (h, r, az) in layouts:
        if args.cam_c_az is not None:
            az_list = [(args.cam_c_az, None, None)]
        else:
            az_list = tv.best_cam_c_azimuths(h, r, az, cfg, n=args.azimuths,
                                             min_sep_deg=args.min_sep)
            print(f"  h{h}/r{r}/az{int(az)}: geometric cam-C candidates "
                  f"{[(int(a), round(w, 2)) for a, w, m in az_list]} (az, two-sided worst-cov)")
        for (cam_c_az, gw, gm) in az_list:
            layout_id = f"u3_h{h}_r{r}_az{int(az)}_oc{int(cam_c_az)}_{args.subject_name}"
            row = None
            for attempt in (1, 2):
                print(f"sweep_3cam_tags: {layout_id} (attempt {attempt}/2) ...", flush=True)
                try:
                    row = camera_rig.evaluate_layout(
                        h, r, az, subject, cfg, machine=args.machine, layout_id=layout_id,
                        subject_pos_name=args.subject_name, episode_duration=args.episode_duration,
                        capture_duration=args.capture_duration, mode="fusion",
                        ring_c_az=cam_c_az, detect_tags=True, spin_deg_s=args.spin_deg_s,
                        marker_front=MARKER_FRONT, marker_back=MARKER_BACK)
                    break
                except Exception as e:
                    print(f"RUN_FAILED {layout_id} (attempt {attempt}/2): {e}")
                    traceback.print_exc()
                    if attempt < 2:
                        time.sleep(10)
            if row is None:
                print(f"RUN_DROPPED {layout_id}", flush=True)
                continue
            if gw is not None:
                row["_geo_worst_cov"] = gw       # diagnostic only; not a results column
            metrics_mod.append_results_row(args.out, row)
            done += 1
            print(f"sweep_3cam_tags: {layout_id} -> mpjpe={row['mpjpe_mm']:.1f} "
                  f"aligned={row['mpjpe_aligned_mm']:.1f} "
                  f"tag_ratio={row.get('tag_visibility_ratio', float('nan')):.3f} "
                  f"(geo two-sided pred {gw if gw is None else round(gw, 2)}) ({done} appended)",
                  flush=True)

    print(f"sweep_3cam_tags: finished, {done} rows -> {args.out}")
    print(f"sweep_3cam_tags: rank with  python3 analysis/rank.py --results {args.out} "
          f"--out reports/ranking_walk_3cam_tags.csv")


if __name__ == "__main__":
    main()
