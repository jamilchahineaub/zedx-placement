#!/usr/bin/env python3
# zed/zed_tag_detect.py
#
# ArUco tag detection across the 3 ring cameras. RUNS UNDER SYSTEM python3 (pyzed + cv2).
#
# Opens the 3 Isaac streams (cam_a/b/c ports), and each frame per camera retrieves the LEFT
# image and runs cv2.aruco.detectMarkers. The two ArUco tags (chest+back) are rendered into
# the scene by isaac/run_episode.py (--chest-tags), so they appear in these streams. No body
# tracking / fusion — just images. Reuses zed_single.open_camera_from_stream and the
# retrieve_image(LEFT) pattern from zed_single.py.
#
# Output: results/layouts/tag_detect_<layout>.csv  (frame_idx, wall_clock, cam, ids)
#         + _meta.json. analysis/tag_metrics.py turns it into a tag_visibility_ratio.
# Sentinels: TAG_READY on first decode; RUN_FAILED stream_dead/no_frames; exit 0 otherwise.

import argparse
import csv
import json
import os
import sys
import threading
import time

import pyzed.sl as sl
import cv2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import zed_single  # noqa: E402  (reuse open_camera_from_stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-id", required=True)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--open-timeout", type=float, default=30.0)
    ap.add_argument("--first-frame-timeout", type=float, default=20.0)
    ap.add_argument("--save-annotated", type=int, default=3,
                    help="save up to N annotated frames where a tag was decoded")
    ap.add_argument("--machine", default="4090")
    args = ap.parse_args()

    with open(os.path.join(REPO, "config", "experiment.yaml")) as f:
        cfg = yaml.safe_load(f)
    aru = cfg.get("aruco") or {}
    aruco_dict = cv2.aruco.Dictionary_get(getattr(cv2.aruco, aru.get("marker_dict", "DICT_6X6_250")))
    params = cv2.aruco.DetectorParameters_create()

    cams = [(nm.upper(), int(cfg[f"cam_{nm}"]["port"])) for nm in ("a", "b", "c")
            if cfg.get(f"cam_{nm}") and "port" in cfg[f"cam_{nm}"]]

    senders = []
    for name, port in cams:
        zed = zed_single.open_camera_from_stream(port, retries=2, open_timeout=args.open_timeout)
        if zed is None:
            print(f"RUN_FAILED tag_detect open cam {name} port {port}")
            for _, z in senders:
                z.close()
            sys.exit(1)
        senders.append((name, zed))
    print(f"zed_tag_detect: opened {len(senders)} cameras {[n for n, _ in senders]}", flush=True)

    out_dir = os.path.join(REPO, "results", "layouts")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    state = {"frames": 0, "frames_any": 0, "stream_dead": False}
    runtime = sl.RuntimeParameters()
    img = sl.Mat()
    saved = 0
    announced = False
    start = time.time()

    stop = threading.Event()

    def _watchdog():
        while not stop.wait(0.5):
            el = time.time() - start
            if state["frames"] == 0 and el >= args.first_frame_timeout:
                state["stream_dead"] = True
                print("[watchdog] no frame in window, closing cameras", flush=True)
                for _, z in senders:
                    z.close()
                return
            if el >= args.duration + 15.0:
                for _, z in senders:
                    z.close()
                return
    threading.Thread(target=_watchdog, daemon=True).start()

    try:
        while True:
            if time.time() - start >= args.duration or state["stream_dead"]:
                break
            fidx = state["frames"] + 1
            wall = time.time()
            any_det = False
            grabbed = False
            for name, zed in senders:
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                grabbed = True
                zed.retrieve_image(img, sl.VIEW.LEFT)
                frame = img.get_data()                       # BGRA numpy
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
                idlist = [] if ids is None else [int(i) for i in ids.flatten()]
                if idlist:
                    any_det = True
                    if saved < args.save_annotated:
                        ann = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                        cv2.imwrite(os.path.join(
                            out_dir, f"tag_detect_{args.layout_id}_{name}_{saved}.png"), ann)
                        saved += 1
                rows.append((fidx, f"{wall:.6f}", name, " ".join(str(i) for i in idlist)))
            if not grabbed:
                continue
            state["frames"] += 1
            if any_det:
                state["frames_any"] += 1
            if not announced and any_det:
                announced = True
                print("TAG_READY", flush=True)
            if state["frames"] % 30 == 1:
                print(f"[diag] frame {state['frames']}, any_tag={any_det}", flush=True)
    finally:
        stop.set()
        out = os.path.join(out_dir, f"tag_detect_{args.layout_id}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "wall_clock", "cam", "ids"])
            w.writerows(rows)
        meta = {
            "layout_id": args.layout_id,
            "cameras": [n for n, _ in senders],
            "frames": state["frames"],
            "frames_with_any_tag": state["frames_any"],
            "rows": len(rows),
            "stream_dead": state["stream_dead"],
            "marker_id_front": aru.get("marker_id_front", 23),
            "marker_id_back": aru.get("marker_id_back", 42),
        }
        with open(os.path.join(out_dir, f"tag_detect_{args.layout_id}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"zed_tag_detect: wrote {len(rows)} rows -> {out}")
        print(f"TAG_SUMMARY frames={state['frames']} frames_with_tag={state['frames_any']} "
              f"rows={len(rows)}", flush=True)
        if state["stream_dead"]:
            print("RUN_FAILED stream_dead", flush=True)
            rc = 2
        elif state["frames"] == 0:
            print("RUN_FAILED no_frames", flush=True)
            rc = 2
        else:
            rc = 0   # frames_with_tag == 0 is a valid 0% measurement, not a failure
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)


if __name__ == "__main__":
    main()
