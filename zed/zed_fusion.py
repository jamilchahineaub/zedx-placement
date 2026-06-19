#!/usr/bin/env python3
# zed/zed_fusion.py
#
# Dual-camera FUSED body tracking. RUNS UNDER SYSTEM python3 (pyzed.sl).
#
# Mirrors the official sample
#   /usr/local/zed/samples/body tracking/multi-camera/python/fused_cameras.py
# with two deliberate deviations for the Isaac virtual-camera case:
#   1. Cameras are opened FROM STREAM (set_from_stream on 127.0.0.1:<port>),
#      not from serial — the fusion JSON's serial (1001/1002) maps to a port
#      via config/experiment.yaml (cam A = 30000 <-> 1001, cam B = 30002 <->
#      1002). The JSON's own communication settings say LOCAL NETWORK; we
#      ignore that branch entirely: both cameras run in THIS process and
#      publish via shared memory (the sample-proven local pattern).
#   2. fusion.subscribe() must use each camera's RUNTIME serial number (the
#      annotator's virtual serial is ignored for camera_model="ZED_X" and the
#      SDK auto-assigns one), while the POSE comes from the JSON entry matched
#      by PORT. If both cameras report the same runtime serial, fusion cannot
#      tell them apart -> RUN_FAILED duplicate_serials (plan B: switch
#      scene_builder annotators to camera_model="VIRTUAL_ZED_X").
#
# Output: results/layouts/zed_pred_<layout>.csv (+ _meta.json sidecar), same
# column layout as zed_single.py. Keypoints are in the FUSION world frame =
# the Y-up conversion of Isaac world written by make_fusion_config.py.
#
# Sentinels / exit codes: FUSION_READY on first fused body; RUN_FAILED
# stream_dead (exit 2) / no_bodies (exit 3); exit 1 on open/subscribe failure.

import argparse
import csv
import json
import os
import sys
import threading
import time

import pyzed.sl as sl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import zed_single  # noqa: E402  (reuse open_camera_from_stream + joint_name)

import yaml  # noqa: E402


def _serial_to_port(cfg):
    return {int(cfg["cam_a"]["serial"]): int(cfg["cam_a"]["port"]),
            int(cfg["cam_b"]["serial"]): int(cfg["cam_b"]["port"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-config", required=True,
                    help="per-layout JSON from zed/make_fusion_config.py")
    ap.add_argument("--layout-id", required=True)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--model", choices=["fast", "accurate"], default="accurate")
    ap.add_argument("--conf", type=int, default=20)
    ap.add_argument("--first-frame-timeout", type=float, default=20.0)
    ap.add_argument("--open-timeout", type=float, default=30.0)
    ap.add_argument("--machine", default="laptop")
    args = ap.parse_args()

    with open(os.path.join(REPO, "config", "experiment.yaml")) as f:
        cfg = yaml.safe_load(f)
    serial_to_port = _serial_to_port(cfg)

    confs = sl.read_fusion_configuration_file(
        args.fusion_config, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
    if len(confs) < 2:
        print(f"RUN_FAILED fusion config has {len(confs)} cameras (need 2): "
              f"{args.fusion_config}")
        sys.exit(1)

    # --- open both cameras from their Isaac streams, publish via SHM ---
    comm = sl.CommunicationParameters()
    comm.set_for_shared_memory()

    pt_params = sl.PositionalTrackingParameters()
    pt_params.set_as_static = True          # sample uses static senders
    # Deterministic sender frame: no IMU input (Isaac's virtual IMU reports the
    # camera as level regardless of its actual tilt). The camera poses are applied
    # by Fusion VERBATIM via subscribe(..., override_gravity=True) below, so no
    # sender-side tilt compensation is needed (this is what the old doubled-pitch /
    # set_initial_world_transform hacks were working around).
    pt_params.enable_imu_fusion = False

    body_params = sl.BodyTrackingParameters()
    body_params.detection_model = zed_single._MODELS[args.model]
    body_params.body_format = sl.BODY_FORMAT.BODY_18
    body_params.enable_body_fitting = False
    body_params.enable_tracking = False     # fusion does the tracking

    senders = {}          # runtime_serial -> sl.Camera
    poses = {}            # runtime_serial -> (conf.pose, override_gravity) matched by PORT
    for conf in confs:
        port = serial_to_port.get(conf.serial_number)
        if port is None:
            print(f"RUN_FAILED fusion config serial {conf.serial_number} not in "
                  f"experiment.yaml cam_a/cam_b")
            sys.exit(1)
        zed = zed_single.open_camera_from_stream(
            port, retries=2, open_timeout=args.open_timeout)
        if zed is None:
            for z in senders.values():
                z.close()
            sys.exit(1)
        runtime_serial = zed.get_camera_information().serial_number
        print(f"zed_fusion: config serial {conf.serial_number} -> port {port} "
              f"-> runtime serial {runtime_serial}", flush=True)
        if runtime_serial in senders:
            print("RUN_FAILED duplicate_serials — both stream cameras report "
                  f"serial {runtime_serial}; switch scene_builder annotators to "
                  "camera_model='VIRTUAL_ZED_X' so serials 1001/1002 are honored")
            for z in senders.values():
                z.close()
            zed.close()
            sys.exit(1)

        if zed.enable_positional_tracking(pt_params) > sl.ERROR_CODE.SUCCESS:
            print(f"RUN_FAILED enable_positional_tracking port {port}")
            sys.exit(1)
        if zed.enable_body_tracking(body_params) > sl.ERROR_CODE.SUCCESS:
            print(f"RUN_FAILED enable_body_tracking port {port}")
            sys.exit(1)
        zed.start_publishing(comm)
        senders[runtime_serial] = zed
        # override_gravity=True => Fusion applies this pose as the ABSOLUTE world
        # pose (no IMU re-leveling). Honors the template's override_gravity field;
        # defaults True since Isaac's virtual IMU is unreliable (reports level).
        poses[runtime_serial] = (conf.pose, bool(getattr(conf, "override_gravity", True)))

    # --- fusion ---
    init_fusion = sl.InitFusionParameters()
    init_fusion.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion.coordinate_units = sl.UNIT.METER
    fusion = sl.Fusion()
    fusion.init(init_fusion)

    # warmup grab (sample does this before subscribing)
    warm = sl.Bodies()
    for zed in senders.values():
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(warm)

    subscribed = 0
    for runtime_serial, (pose, override_gravity) in poses.items():
        uuid = sl.CameraIdentifier()
        uuid.serial_number = runtime_serial
        sub_comm = sl.CommunicationParameters()
        sub_comm.set_for_shared_memory()
        status = fusion.subscribe(uuid, sub_comm, pose, override_gravity)
        if status != sl.FUSION_ERROR_CODE.SUCCESS:
            print(f"RUN_FAILED fusion.subscribe serial {runtime_serial}: {status}")
        else:
            subscribed += 1
    if subscribed < 2:
        for zed in senders.values():
            zed.close()
        sys.exit(1)

    ft_params = sl.BodyTrackingFusionParameters()
    ft_params.enable_tracking = True
    ft_params.enable_body_fitting = False
    fusion.enable_body_tracking(ft_params)

    rt = sl.BodyTrackingFusionRuntimeParameters()
    rt.skeleton_minimum_allowed_keypoints = 7

    # --- capture loop ---
    # Watchdog is a FAILSAFE ONLY here: while frames flow (~7 fps) the loop's
    # own duration check exits cleanly. Closing the cameras from the watchdog
    # while the main thread is inside grab()/fusion.process() SEGFAULTS the
    # SDK (observed rc=-11), killing the process before the CSV is written —
    # so the watchdog only fires if the loop is actually STUCK: no first frame
    # within the window, or duration overrun by 15 s (blocked grab).
    rows = []
    frame_log = []        # (frame_idx, wall_clock, n_bodies) for EVERY grabbed frame
    announced = False
    start = time.time()
    state = {"frames": 0, "frames_with_bodies": 0, "stream_dead": False}
    per_cam = sl.Bodies()
    fused = sl.Bodies()

    watchdog_stop = threading.Event()

    def _watchdog():
        while not watchdog_stop.wait(timeout=0.5):
            elapsed = time.time() - start
            if state["frames"] == 0 and elapsed >= args.first_frame_timeout:
                state["stream_dead"] = True
                print("[watchdog] no fused frame in window, closing cameras",
                      flush=True)
                for z in senders.values():
                    z.close()
                return
            if elapsed >= args.duration + 15.0:
                print("[watchdog] loop stuck past duration+15s, "
                      "force-closing cameras", flush=True)
                for z in senders.values():
                    z.close()
                return

    threading.Thread(target=_watchdog, daemon=True).start()

    try:
        while True:
            if time.time() - start >= args.duration or state["stream_dead"]:
                break
            for zed in senders.values():
                if zed.grab() <= sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_bodies(per_cam)
            if fusion.process() != sl.FUSION_ERROR_CODE.SUCCESS:
                continue
            fusion.retrieve_bodies(fused, rt)
            state["frames"] += 1
            wall = time.time()
            frame_log.append((state["frames"], wall, len(fused.body_list)))

            if state["frames"] % 30 == 1:
                print(f"[diag] fused frame {state['frames']}, "
                      f"bodies={len(fused.body_list)}", flush=True)

            if fused.body_list:
                state["frames_with_bodies"] += 1
            for body in fused.body_list:
                kp = body.keypoint
                kpc = body.keypoint_confidence
                for idx in range(len(kp)):
                    x, y, z = float(kp[idx][0]), float(kp[idx][1]), float(kp[idx][2])
                    conf = float(kpc[idx]) if idx < len(kpc) else float("nan")
                    rows.append((state["frames"], wall, int(body.id),
                                 str(body.tracking_state),
                                 idx, zed_single.joint_name(idx), x, y, z, conf))

            if not announced and len(fused.body_list) > 0:
                announced = True
                print("FUSION_READY", flush=True)
    finally:
        watchdog_stop.set()
        out = os.path.join(REPO, "results", "layouts", f"zed_pred_{args.layout_id}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "wall_clock", "body_id", "tracking_state",
                        "joint_idx", "joint_name", "x", "y", "z", "confidence"])
            w.writerows(rows)
        # Per-frame heartbeat: EVERY grabbed frame (incl. zero-body frames), so the
        # floor-coverage map knows where detection drops to zero.
        frames_out = os.path.join(REPO, "results", "layouts",
                                  f"zed_pred_{args.layout_id}_frames.csv")
        with open(frames_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "wall_clock", "n_bodies"])
            w.writerows(frame_log)
        meta = {
            "layout_id": args.layout_id,
            "mode": "fusion",
            "model": args.model,
            "conf": args.conf,
            "frames_grabbed": state["frames"],
            "frames_with_bodies": state["frames_with_bodies"],
            "rows": len(rows),
            "stream_dead": state["stream_dead"],
            "serials": sorted(senders),
        }
        with open(os.path.join(REPO, "results", "layouts",
                               f"zed_pred_{args.layout_id}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"zed_fusion: wrote {len(rows)} rows -> {out}")
        print(f"ZED_SUMMARY frames={state['frames']} "
              f"frames_with_bodies={state['frames_with_bodies']} rows={len(rows)}",
              flush=True)

        # Do NOT close the cameras or let the interpreter tear down the Fusion
        # object — the SDK segfaults (rc=-11) closing cameras while Fusion is
        # still subscribed (observed even with CSV safely written). All
        # artifacts are flushed; let process death reclaim sockets/SHM handles
        # (preflight cleans any residue before the next run).
        if state["stream_dead"]:
            print("RUN_FAILED stream_dead", flush=True)
            rc = 2
        elif state["frames"] > 0 and len(rows) == 0:
            print(f"RUN_FAILED no_bodies frames={state['frames']}", flush=True)
            rc = 3
        else:
            rc = 0
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)


if __name__ == "__main__":
    main()
