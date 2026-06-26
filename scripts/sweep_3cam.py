#!/usr/bin/env python3
# scripts/sweep_3cam.py
#
# STAGE-2 overhead (3-camera) sweep. RUNS UNDER SYSTEM python3 ONLY.
#
# Does NOT re-run the whole grid and does NOT touch the 2-cam pipeline. It takes
# the already-ranked top-N 2-cam ring layouts (from a ranking CSV produced by
# analysis/rank.py, e.g. reports/ranking_walk.csv) and, for each one, adds a
# centered overhead (nadir) camera whose HEIGHT is swept over experiment.yaml's
# `overhead:` grid. Each (ring layout x overhead height) is run end-to-end as a
# full 3-camera fusion episode and APPENDED to its own results file
# (default results/results_walk_3cam.csv) — separate from the 2-cam results.
#
# Then rank it the normal way and compare to the 2-cam ranking:
#   python3 analysis/rank.py --results results/results_walk_3cam.csv \
#           --out reports/ranking_walk_3cam.csv
#
# Usage:
#   python3 scripts/sweep_3cam.py --machine laptop                 # full top-N x heights
#   python3 scripts/sweep_3cam.py --machine laptop --limit 2       # smoke test
#   python3 scripts/sweep_3cam.py --ranking reports/ranking_walk.csv --top 15
#
# evaluate_layout(overhead_h=...) is the boundary; this only chooses the work.

import argparse
import csv
import os
import sys
import time
import traceback

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "isaac"), os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import camera_rig              # noqa: E402
import metrics as metrics_mod  # noqa: E402

DEFAULT_RANKING = os.path.join(REPO, "reports", "ranking_walk.csv")
DEFAULT_RESULTS = os.path.join(REPO, "results", "results_walk_3cam.csv")


def overhead_heights(cfg):
    """Inclusive height grid from experiment.yaml `overhead:` (h_range + h_step)."""
    oh = cfg.get("overhead", {}) or {}
    lo, hi = oh.get("h_range", [3.5, 7.0])
    step = oh.get("h_step", 0.5)
    heights = []
    h = float(lo)
    # +1e-6 so the inclusive endpoint is not lost to float drift.
    while h <= float(hi) + 1e-6:
        heights.append(round(h, 6))
        h += float(step)
    return heights


def top_layouts(ranking_csv, top_n):
    """Read the top-N (h_m, r_m, rel_az_deg) ring layouts from a rank.py CSV,
    in rank order (the file is already sorted by rank, but we sort to be safe)."""
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


def main():
    ap = argparse.ArgumentParser(description="Stage-2 overhead (3-cam) sweep")
    ap.add_argument("--machine", default="laptop")
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--ranking", default=DEFAULT_RANKING,
                    help="2-cam ranking CSV to read the top layouts from")
    ap.add_argument("--out", default=DEFAULT_RESULTS,
                    help="append-only results CSV for the 3-cam runs")
    ap.add_argument("--top", type=int, default=None,
                    help="number of top ring layouts (default: experiment.yaml overhead.top_n)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N evaluated (layout x height) runs")
    ap.add_argument("--episode-duration", type=float, default=120.0)
    ap.add_argument("--capture-duration", type=float, default=20.0)
    args = ap.parse_args()

    with open(os.path.join(REPO, "config", "experiment.yaml")) as f:
        cfg = yaml.safe_load(f)
    subject = next(s["pos"] for s in cfg["subject_positions"]
                   if s["name"] == args.subject_name)

    if not os.path.exists(args.ranking):
        print(f"sweep_3cam: ranking file not found: {args.ranking}")
        sys.exit(1)

    top_n = args.top if args.top is not None else (cfg.get("overhead", {}) or {}).get("top_n", 15)
    layouts = top_layouts(args.ranking, top_n)
    heights = overhead_heights(cfg)
    print(f"sweep_3cam: {len(layouts)} ring layouts x {len(heights)} overhead heights "
          f"{heights} -> up to {len(layouts) * len(heights)} runs", flush=True)
    print(f"sweep_3cam: reading top layouts from {args.ranking}", flush=True)
    print(f"sweep_3cam: appending results to {args.out}", flush=True)

    done = 0
    for (h, r, az) in layouts:
        for oh_h in heights:
            layout_id = f"3cam_h{h}_r{r}_az{int(az)}_oh{oh_h}_{args.subject_name}"
            row = None
            for attempt in (1, 2):
                print(f"sweep_3cam: evaluating {layout_id} (attempt {attempt}/2) ...", flush=True)
                try:
                    row = camera_rig.evaluate_layout(
                        h, r, az, subject, cfg, machine=args.machine,
                        layout_id=layout_id, subject_pos_name=args.subject_name,
                        episode_duration=args.episode_duration,
                        capture_duration=args.capture_duration, mode="fusion",
                        overhead_h=oh_h)
                    break
                except Exception as e:
                    print(f"RUN_FAILED {layout_id} (attempt {attempt}/2): {e}")
                    traceback.print_exc()
                    if attempt < 2:
                        time.sleep(10)
            if row is None:
                print(f"RUN_DROPPED {layout_id}: failed after 2 attempts", flush=True)
                continue

            metrics_mod.append_results_row(args.out, row)
            done += 1
            print(f"sweep_3cam: {layout_id} -> mpjpe={row['mpjpe_mm']:.1f}mm "
                  f"aligned={row['mpjpe_aligned_mm']:.1f}mm ({done} rows appended)", flush=True)
            if args.limit and done >= args.limit:
                print(f"sweep_3cam: hit --limit {args.limit}, stopping", flush=True)
                print(f"sweep_3cam: finished, {done} rows appended to {args.out}")
                return

    print(f"sweep_3cam: finished, {done} rows appended to {args.out}")
    print(f"sweep_3cam: now rank with:\n"
          f"  python3 analysis/rank.py --results {args.out} "
          f"--out reports/ranking_walk_3cam.csv")


if __name__ == "__main__":
    main()
