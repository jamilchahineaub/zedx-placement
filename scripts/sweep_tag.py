#!/usr/bin/env python3
# scripts/sweep_tag.py
#
# Chest-ArUco third-camera placement sweep. RUNS UNDER SYSTEM python3 — NO Isaac, NO ZED.
#
# Base = the good top-N ring layouts from the 2-cam walk test (reports/ranking_walk.csv).
# For each, cam A and cam B are FIXED at that layout's (h, r, az). We add a third camera on
# the SAME ring (same h, r) and sweep ONLY its azimuth, asking: which angle best keeps a
# chest-mounted ArUco tag decodable as the robot moves AND turns to any facing?
#
# "Captured" = ANY of the 3 cameras decodes the tag (in FOV+range, within the decode cone of
# the tag normal, close enough). Coverage is measured over every floor cell x every facing
# (pure geometry, analysis/tag_visibility.py). Primary metric: worst-case angular coverage
# (min over cells of the fraction of facings >=1 camera can decode) -> 1.0 means the tag is
# always decodable wherever the robot is and whichever way it faces.
#
# Output: results/tag_coverage.csv (recomputed each run; deterministic + instant).
#
# Usage:
#   python3 scripts/sweep_tag.py                          # all top-N layouts
#   python3 scripts/sweep_tag.py --limit 2                # quick check
#   python3 scripts/sweep_tag.py --ranking reports/ranking_walk.csv --out results/tag_coverage.csv

import argparse
import csv
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "isaac"), os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tag_visibility as tv   # noqa: E402

DEFAULT_RANKING = os.path.join(REPO, "reports", "ranking_walk.csv")
DEFAULT_OUT = os.path.join(REPO, "results", "tag_coverage.csv")

COLUMNS = ["h_m", "r_m", "rel_az_deg", "cam_c_az_deg",
           "worst_coverage", "mean_coverage", "blind_wedge_deg",
           "baseline_worst", "baseline_mean", "delta_worst"]


def top_layouts(ranking_csv, top_n):
    """Top-N (h, r, rel_az) ring layouts from a rank.py CSV, in rank order."""
    rows = []
    with open(ranking_csv, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(float(r["rank"])),
                             float(r["h_m"]), float(r["r_m"]), float(r["rel_az_deg"])))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda t: t[0])
    return [(h, r, az) for (_rank, h, r, az) in rows[:top_n]]


def _circ_dist(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def main():
    ap = argparse.ArgumentParser(description="Chest-ArUco 3rd-camera azimuth sweep (geometric)")
    ap.add_argument("--ranking", default=DEFAULT_RANKING)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--config", default=os.path.join(REPO, "config", "experiment.yaml"))
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None, help="only the first N layouts")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    params = tv.aruco_params(cfg)
    aim_h = float(cfg.get("aim_height_m", 1.0))         # ring cams aim at hip (their real aim)
    chest = params["chest_height_m"]
    ws = cfg.get("workspace") or {}
    cx, cy = (list(ws.get("center", [0.0, 0.0])) + [0.0, 0.0])[:2]
    aim_ab = [cx, cy, aim_h]
    aim_c = [cx, cy, chest]
    a0 = float(cfg["cam_a"]["azimuth_deg"])
    step = params["cam_c_azimuth_step_deg"]
    az_grid = [i * step for i in range(int(round(360.0 / step)))]

    if not os.path.exists(args.ranking):
        print(f"sweep_tag: ranking file not found: {args.ranking}")
        sys.exit(1)
    layouts = top_layouts(args.ranking, args.top)
    if args.limit:
        layouts = layouts[:args.limit]
    print(f"sweep_tag: {len(layouts)} ring layouts x {len(az_grid)} cam-C azimuths "
          f"(facing step {params['facing_step_deg']}°, decode<= {params['decode_max_deg']}°, "
          f"range<= {params['max_decode_range_m']}m)")

    rows = []
    best_overall = None
    for (h, r, az) in layouts:
        cam_a = tv.make_cam(a0, r, h, aim_ab)
        cam_b = tv.make_cam(a0 + az, r, h, aim_ab)
        base = tv.coverage_map([cam_a, cam_b], cfg, params)
        b_worst, b_mean = base["worst"], base["mean"]

        best = None
        for az_c in az_grid:
            # skip angles essentially on top of cam A or cam B (can't co-mount)
            if _circ_dist(az_c, a0) < step / 2 or _circ_dist(az_c, a0 + az) < step / 2:
                continue
            cam_c = tv.make_cam(az_c, r, h, aim_c)
            cm = tv.coverage_map([cam_a, cam_b, cam_c], cfg, params)
            row = {"h_m": h, "r_m": r, "rel_az_deg": az, "cam_c_az_deg": az_c,
                   "worst_coverage": cm["worst"], "mean_coverage": cm["mean"],
                   "blind_wedge_deg": cm["blind_wedge_deg"],
                   "baseline_worst": b_worst, "baseline_mean": b_mean,
                   "delta_worst": cm["worst"] - b_worst}
            rows.append(row)
            key = (cm["worst"], cm["mean"])
            if best is None or key > (best["worst_coverage"], best["mean_coverage"]):
                best = row

        print(f"  h{h} r{r} az{int(az)}: 2-cam worst={b_worst:.2f} -> "
              f"+cam_c@{int(best['cam_c_az_deg'])}° worst={best['worst_coverage']:.2f} "
              f"(Δ{best['delta_worst']:+.2f}, mean {best['mean_coverage']:.2f}, "
              f"blind wedge {best['blind_wedge_deg']:.0f}°)")
        if best_overall is None or (best["worst_coverage"], best["mean_coverage"]) > \
                (best_overall["worst_coverage"], best_overall["mean_coverage"]):
            best_overall = best

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r[c] for c in COLUMNS])
    print(f"\nsweep_tag: wrote {len(rows)} rows -> {args.out}")
    if best_overall:
        b = best_overall
        print(f"sweep_tag: BEST overall -> h{b['h_m']} r{b['r_m']} az{int(b['rel_az_deg'])} "
              f"+ cam C @ {int(b['cam_c_az_deg'])}°: worst-case coverage "
              f"{b['worst_coverage']:.2f} (2-cam {b['baseline_worst']:.2f}), "
              f"largest blind wedge {b['blind_wedge_deg']:.0f}°")


if __name__ == "__main__":
    main()
