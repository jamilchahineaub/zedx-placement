#!/usr/bin/env python3
# scripts/view_tag.py
#
# Generate an ArUco marker, then launch the INTERACTIVE Isaac viewport showing one 3-camera
# layout with that tag on the character's chest. RUNS UNDER SYSTEM python3 (needs cv2 for the
# marker; the Isaac scene runs under isaac_python). Visual sanity check only.
#
# Default = the sweep's best config: h3.0 / r3.5 / az180 ring + cam C @ 90°.
#
# Usage:
#   python3 scripts/view_tag.py                       # best config, tag faces cam C
#   python3 scripts/view_tag.py --cam-c-az 270        # try a different 3rd-camera angle
#   python3 scripts/view_tag.py --h 3.5 --r 4.5 --rel-az 180 --cam-c-az 90 --tag-az 45

import argparse
import os
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER_FRONT = os.path.join(REPO, "assets", "aruco_marker_front.png")
MARKER_BACK = os.path.join(REPO, "assets", "aruco_marker_back.png")


def gen_marker(path, marker_id=23, px=600, border=80):
    """Write a DICT_6X6_250 ArUco marker PNG with a white quiet-zone border."""
    import cv2
    import cv2.aruco as aruco
    import numpy as np
    d = aruco.Dictionary_get(aruco.DICT_6X6_250)
    img = aruco.drawMarker(d, marker_id, px)
    canvas = np.full((px + 2 * border, px + 2 * border), 255, np.uint8)
    canvas[border:border + px, border:border + px] = img
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, canvas)
    print(f"view_tag: wrote marker (id {marker_id}, DICT_6X6_250) -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Interactive viewport: chest ArUco tag + 3 cameras")
    ap.add_argument("--h", type=float, default=3.0)
    ap.add_argument("--r", type=float, default=3.5)
    ap.add_argument("--rel-az", type=float, default=180.0)
    ap.add_argument("--cam-c-az", type=float, default=90.0)
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--machine", default="4090")
    ap.add_argument("--tag-size", type=float, default=0.30)
    ap.add_argument("--tag-offset", type=float, default=0.30,
                    help="metres to push each tag off the chest to clear the torso")
    ap.add_argument("--spin-deg-s", type=float, default=30.0,
                    help="character yaw rate (deg/s); 0 = no spin")
    ap.add_argument("--walk-speed", type=float, default=1.0,
                    help="walk speed (m/s) for the viewer (config is 1.5; lower = slower)")
    ap.add_argument("--regen-marker", action="store_true", help="regenerate the marker PNGs")
    args = ap.parse_args()

    if args.regen_marker or not os.path.exists(MARKER_FRONT):
        gen_marker(MARKER_FRONT, marker_id=23)
    if args.regen_marker or not os.path.exists(MARKER_BACK):
        gen_marker(MARKER_BACK, marker_id=42)

    machine_cfg = yaml.safe_load(open(os.path.join(REPO, "config", f"machine.{args.machine}.yaml")))
    isaac_py = machine_cfg["isaac_python"]
    cmd = [isaac_py, os.path.join(REPO, "isaac", "view_tag_scene.py"),
           "--h", str(args.h), "--r", str(args.r), "--rel-az", str(args.rel_az),
           "--cam-c-az", str(args.cam_c_az), "--subject-name", args.subject_name,
           "--machine", args.machine, "--marker-front", MARKER_FRONT, "--marker-back", MARKER_BACK,
           "--tag-size", str(args.tag_size), "--tag-offset", str(args.tag_offset),
           "--spin-deg-s", str(args.spin_deg_s), "--walk-speed", str(args.walk_speed)]
    print("view_tag: launching ->", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=REPO))


if __name__ == "__main__":
    main()
