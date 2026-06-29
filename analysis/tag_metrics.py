#!/usr/bin/env python3
# analysis/tag_metrics.py
#
# Turn a zed_tag_detect detection log into a tag-visibility-ratio metric row. Plain python3.
#
# Reads results/layouts/tag_detect_<layout>.csv (frame_idx, wall_clock, cam, ids) and computes:
#   tag_visibility_ratio  = fraction of frames where ANY camera decoded ANY tag (the headline)
#   detect_rate_cam_a/b/c = per-camera fraction of frames that camera decoded a tag
#   detect_rate_front/back= per-tag (front/back marker id) fraction
#   longest_blind_gap_s   = longest continuous stretch (s) with no detection on any camera
# and appends a row (with layout columns) to results/tag_detect.csv (append-only).

import argparse
import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

COLUMNS = ["h_m", "r_m", "rel_az_deg", "cam_c_az_deg", "spin_deg_s",
           "tag_visibility_ratio", "detect_rate_cam_a", "detect_rate_cam_b", "detect_rate_cam_c",
           "detect_rate_front", "detect_rate_back", "longest_blind_gap_s",
           "frames", "subject_pos_name"]


def load_detections(detect_csv):
    """-> {frame_idx: {"wall": float, "cams": {cam: set(ids)}}} in frame order."""
    frames = {}
    with open(detect_csv, newline="") as f:
        for row in csv.DictReader(f):
            fi = int(row["frame_idx"])
            ids = set(int(x) for x in row["ids"].split()) if row["ids"].strip() else set()
            fr = frames.setdefault(fi, {"wall": float(row["wall_clock"]), "cams": {}})
            fr["cams"][row["cam"]] = ids
    return [frames[k] for k in sorted(frames)]


def compute(detect_csv, front_id, back_id):
    frames = load_detections(detect_csv)
    n = len(frames)
    if n == 0:
        return None
    any_seen = []
    cam_hits = {}
    front = back = 0
    for fr in frames:
        all_ids = set().union(*fr["cams"].values()) if fr["cams"] else set()
        seen = len(all_ids) > 0
        any_seen.append(seen)
        front += 1 if front_id in all_ids else 0
        back += 1 if back_id in all_ids else 0
        for cam, ids in fr["cams"].items():
            cam_hits.setdefault(cam, 0)
            cam_hits[cam] += 1 if ids else 0
    # longest blind gap in seconds (consecutive frames with no detection on any camera)
    walls = [fr["wall"] for fr in frames]
    dt = ((walls[-1] - walls[0]) / (n - 1)) if n > 1 else 0.0
    longest = cur = 0
    for s in any_seen:
        cur = 0 if s else cur + 1
        longest = max(longest, cur)
    return {
        "tag_visibility_ratio": sum(any_seen) / n,
        "detect_rate_cam_a": cam_hits.get("A", 0) / n,
        "detect_rate_cam_b": cam_hits.get("B", 0) / n,
        "detect_rate_cam_c": cam_hits.get("C", 0) / n,
        "detect_rate_front": front / n,
        "detect_rate_back": back / n,
        "longest_blind_gap_s": longest * dt,
        "frames": n,
    }


def append_row(results_csv, row):
    new = not os.path.exists(results_csv)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([row.get(c, "") for c in COLUMNS])


def main():
    ap = argparse.ArgumentParser(description="Tag-visibility-ratio metric from a detection log")
    ap.add_argument("--layout-id", required=True)
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--rel-az", type=float, required=True)
    ap.add_argument("--cam-c-az", type=float, required=True)
    ap.add_argument("--spin-deg-s", type=float, default=0.0)
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--results-dir", default=os.path.join(_REPO, "results", "layouts"))
    ap.add_argument("--out", default=os.path.join(_REPO, "results", "tag_detect.csv"))
    ap.add_argument("--front-id", type=int, default=23)
    ap.add_argument("--back-id", type=int, default=42)
    args = ap.parse_args()

    detect_csv = os.path.join(args.results_dir, f"tag_detect_{args.layout_id}.csv")
    if not os.path.exists(detect_csv):
        print(f"tag_metrics: no detection log {detect_csv}")
        return
    m = compute(detect_csv, args.front_id, args.back_id)
    if m is None:
        print(f"tag_metrics: {args.layout_id} has no frames")
        return
    m.update({"h_m": args.h, "r_m": args.r, "rel_az_deg": args.rel_az,
              "cam_c_az_deg": args.cam_c_az, "spin_deg_s": args.spin_deg_s,
              "subject_pos_name": args.subject_name})
    append_row(args.out, m)
    print(f"tag_metrics: {args.layout_id} -> visibility_ratio={m['tag_visibility_ratio']:.3f} "
          f"(A={m['detect_rate_cam_a']:.2f} B={m['detect_rate_cam_b']:.2f} "
          f"C={m['detect_rate_cam_c']:.2f}, front={m['detect_rate_front']:.2f} "
          f"back={m['detect_rate_back']:.2f}, longest gap {m['longest_blind_gap_s']:.1f}s) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
