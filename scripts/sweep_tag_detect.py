#!/usr/bin/env python3
# scripts/sweep_tag_detect.py
#
# Real ArUco tag-detection sweep. RUNS UNDER SYSTEM python3.
#
# For each of the top-N ring layouts (reports/ranking_walk.csv), run a real detection episode
# (3 ring cameras + chest/back tags + spin -> cv2.aruco) at that layout's GEOMETRIC-BEST cam_c
# azimuth (from results/tag_coverage.csv, produced by scripts/sweep_tag.py). Real episodes cost
# ~3 min each, so this is 15 runs, not a dense azimuth sweep. Appends a tag_visibility_ratio row
# per layout to results/tag_detect.csv.
#
# Usage:
#   python3 scripts/sweep_tag_detect.py --machine 4090            # top-15 at best cam_c az
#   python3 scripts/sweep_tag_detect.py --machine 4090 --limit 3  # smoke test
#   python3 scripts/sweep_tag_detect.py --cam-c-az 90             # force one cam_c angle for all

import argparse
import csv
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import run_tag_pipeline as rtp     # noqa: E402
from scripts.sweep_tag import top_layouts       # noqa: E402  (reuse the ranking reader)

DEFAULT_RANKING = os.path.join(REPO, "reports", "ranking_walk.csv")
COVERAGE_CSV = os.path.join(REPO, "results", "tag_coverage.csv")


def best_cam_c(coverage_csv):
    """{(h,r,az): cam_c_az} — the azimuth with the highest geometric worst_coverage per layout."""
    best = {}
    if not os.path.exists(coverage_csv):
        return best
    for row in csv.DictReader(open(coverage_csv)):
        try:
            k = (round(float(row["h_m"]), 3), round(float(row["r_m"]), 3),
                 round(float(row["rel_az_deg"]), 3))
            w = float(row["worst_coverage"])
            az = float(row["cam_c_az_deg"])
        except (KeyError, ValueError):
            continue
        if k not in best or w > best[k][0]:
            best[k] = (w, az)
    return {k: v[1] for k, v in best.items()}


def main():
    ap = argparse.ArgumentParser(description="Real ArUco tag-detection sweep (top-N layouts)")
    ap.add_argument("--machine", default="4090")
    ap.add_argument("--ranking", default=DEFAULT_RANKING)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cam-c-az", type=float, default=None,
                    help="force this cam C azimuth for all layouts (else geometric-best)")
    ap.add_argument("--spin-deg-s", type=float, default=30.0)
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--episode-duration", type=float, default=120.0)
    ap.add_argument("--capture-duration", type=float, default=20.0)
    args = ap.parse_args()

    if not os.path.exists(args.ranking):
        print(f"sweep_tag_detect: ranking not found: {args.ranking}")
        sys.exit(1)
    layouts = top_layouts(args.ranking, args.top)
    if args.limit:
        layouts = layouts[:args.limit]
    best = best_cam_c(COVERAGE_CSV)
    if args.cam_c_az is None and not best:
        print(f"sweep_tag_detect: no {COVERAGE_CSV}; run scripts/sweep_tag.py first or pass --cam-c-az")
        sys.exit(1)

    print(f"sweep_tag_detect: {len(layouts)} layouts, spin {args.spin_deg_s}°/s, "
          f"cam_c = {'fixed ' + str(args.cam_c_az) + '°' if args.cam_c_az is not None else 'geometric-best'}")
    done = 0
    for (h, r, az) in layouts:
        cam_c_az = args.cam_c_az if args.cam_c_az is not None else \
            best.get((round(h, 3), round(r, 3), round(az, 3)))
        if cam_c_az is None:
            print(f"sweep_tag_detect: no best cam_c for h{h}/r{r}/az{int(az)}, skipping")
            continue
        layout_id = f"tag_h{h}_r{r}_az{int(az)}_oc{int(cam_c_az)}_{args.subject_name}"
        try:
            m = rtp.run_one(h, r, az, cam_c_az, layout_id, args.machine,
                            spin_deg_s=args.spin_deg_s, subject_name=args.subject_name,
                            episode_duration=args.episode_duration,
                            capture_duration=args.capture_duration)
        except Exception as e:
            print(f"RUN_FAILED {layout_id}: {e}")
            traceback.print_exc()
            m = None
        if m is None:
            print(f"RUN_DROPPED {layout_id}", flush=True)
            continue
        done += 1
        print(f"sweep_tag_detect: {layout_id} -> ratio={m['tag_visibility_ratio']:.3f} "
              f"({done} done)", flush=True)
        if args.limit and done >= args.limit:
            break

    print(f"sweep_tag_detect: finished, {done} layouts -> results/tag_detect.csv")


if __name__ == "__main__":
    main()
