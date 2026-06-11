#!/usr/bin/env python3
# zed/zed_single.py
#
# Single-camera ZED body tracking. RUNS UNDER SYSTEM python3 (pyzed.sl).
# Subscribes to ONE Isaac ZED stream, runs body tracking, and writes per-frame
# skeleton keypoints to CSV (+ a _meta.json sidecar with frame counters that
# analysis/metrics.py needs for detection_coverage).
#
# Coordinate system: RIGHT_HANDED_Y_UP, metres (ZED side is Y-up per CLAUDE.md).
#
# Init parameters MIRROR THE PROVEN RECEIVER — the user-modified sample
# /usr/local/zed/samples/body tracking/body tracking/python/body_tracking.py
# (HD1080, NEURAL depth, Y-up, METER, set_from_stream(127.0.0.1, 30000),
#  HUMAN_BODY_FAST, fitting off, confidence 40, positional tracking default)
# which successfully tracked the Isaac character on this machine.
#
# CLI:
#   python3 zed/zed_single.py --port 30000 --layout-id test_001 --duration 30
#
# Sentinels / exit codes:
#   ZED_SINGLE_READY            first body detected
#   RUN_FAILED stream_dead      opened but no frame within --first-frame-timeout (exit 2)
#   RUN_FAILED no_bodies ...    frames grabbed but no body ever detected (exit 3)
#   exit 1                      zed.open() failed after retries

import argparse
import csv
import json
import os
import select
import signal
import stat
import sys
import threading
import time

import pyzed.sl as sl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# BODY_18 to match the fusion sample (fused_cameras.py uses BODY_18). Keep single and
# fusion on the same format so joint_map / metrics line up.
BODY_FORMAT = sl.BODY_FORMAT.BODY_18
_PARTS_ENUM = sl.BODY_18_PARTS

_MODELS = {
    "fast": sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST,
    "accurate": sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE,
}


def joint_name(idx):
    """Canonical ZED joint name for a keypoint index, e.g. 0 -> 'NOSE'."""
    try:
        return _PARTS_ENUM(idx).name
    except Exception:
        return f"JOINT_{idx}"


def open_camera_from_stream(port, retries=3, retry_sleep=2.0, verbose=False,
                            open_timeout=30.0):
    """Open a ZED camera connected to an Isaac virtual stream on 127.0.0.1:<port>.

    Isaac's ZEDAnnotator (transport BOTH) publishes via SHM-Boost topics
    (/dev/shm/sl_local_*); the receiver must find that topic at open() time —
    the UDP fallback can never work on the same machine because the sender
    already binds the RTP ports. Retry open a few times to ride out the
    sender-side startup race.

    zed.open() against a dead/absent stream blocks >2 min inside the C SDK and
    is NOT interruptible by Python signal handlers or threads. SIGALRM's
    *default* disposition (terminate the process) does fire inside C code, so
    we arm a hard alarm per attempt: if open() hangs past open_timeout, the
    process dies with exit 142 (=128+SIGALRM) and the orchestrator treats it
    like stream_dead.
    """
    last_status = None
    for attempt in range(1, retries + 1):
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        if verbose:
            init.sdk_verbose = 1
        init.set_from_stream("127.0.0.1", port)

        zed = sl.Camera()
        # Reset SIGALRM to the default (terminate) — pytest/parents may have
        # installed handlers that would be swallowed mid-C-call.
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        print(f"zed_single: opening port {port} (attempt {attempt}/{retries}, "
              f"hard timeout {open_timeout:.0f}s)", flush=True)
        signal.alarm(max(1, int(open_timeout)))
        status = zed.open(init)
        signal.alarm(0)
        if status == sl.ERROR_CODE.SUCCESS:
            info = zed.get_camera_information()
            print(f"zed_single: opened port={port} serial={info.serial_number}",
                  flush=True)
            return zed
        last_status = status
        print(f"zed_single: open attempt {attempt}/{retries} on port {port} "
              f"failed: {status}", flush=True)
        try:
            zed.close()
        except Exception:
            pass
        time.sleep(retry_sleep)

    print(f"RUN_FAILED zed.open on port {port}: {last_status}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000, help="Isaac stream port (30000/30002)")
    ap.add_argument("--layout-id", required=True, help="id for the output filename")
    ap.add_argument("--duration", type=float, default=30.0, help="max seconds to capture")
    ap.add_argument("--model", choices=sorted(_MODELS), default="accurate",
                    help="body tracking model (accurate: required for the synthetic room)")
    ap.add_argument("--conf", type=int, default=20,
                    help="detection confidence threshold (20: synthetic rendering)")
    ap.add_argument("--first-frame-timeout", type=float, default=15.0,
                    help="seconds to wait for the first grab before declaring stream_dead")
    ap.add_argument("--open-retries", type=int, default=3)
    ap.add_argument("--open-timeout", type=float, default=30.0,
                    help="hard per-attempt cap on zed.open(); process exits 142 if exceeded")
    ap.add_argument("--static-pt", action="store_true",
                    help="set positional tracking set_as_static (default: off, like the sample)")
    ap.add_argument("--verbose", action="store_true", help="sdk_verbose=1")
    ap.add_argument("--save-frame", default=None, metavar="PNG",
                    help="save the left view of one mid-run frame to this path "
                         "(diagnostic: shows exactly what the camera sees)")
    ap.add_argument("--machine", default="laptop")  # accepted for parity; unused here
    args = ap.parse_args()

    zed = open_camera_from_stream(args.port, retries=args.open_retries,
                                  verbose=args.verbose,
                                  open_timeout=args.open_timeout)
    if zed is None:
        sys.exit(1)

    pt_params = sl.PositionalTrackingParameters()
    if args.static_pt:
        pt_params.set_as_static = True
    zed.enable_positional_tracking(pt_params)

    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.enable_body_fitting = False        # proven sample: off
    body_params.detection_model = _MODELS[args.model]
    body_params.body_format = BODY_FORMAT
    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print(f"RUN_FAILED enable_body_tracking on port {args.port}")
        zed.close()
        sys.exit(1)

    body_runtime = sl.BodyTrackingRuntimeParameters()
    body_runtime.detection_confidence_threshold = args.conf
    bodies = sl.Bodies()
    runtime = sl.RuntimeParameters()

    rows = []
    announced = False
    start = time.time()
    # Shared state for the watchdog (mutated from the grab loop).
    state = {"frames": 0, "frames_with_bodies": 0, "t_first_grab": None,
             "stream_dead": False}

    # zed.grab() blocks indefinitely if the Isaac stream never delivers, so loop
    # checks alone can't enforce timeouts. The watchdog force-closes the camera
    # when either (a) no first frame within --first-frame-timeout (stream dead;
    # fail fast so the orchestrator can retry on the other port), or (b) the
    # full --duration elapsed. close() makes any in-flight grab() return.
    watchdog_stop = threading.Event()

    def _watchdog():
        while not watchdog_stop.wait(timeout=0.5):
            elapsed = time.time() - start
            if state["frames"] == 0 and elapsed >= args.first_frame_timeout:
                state["stream_dead"] = True
                print("[watchdog] no frame within first-frame window, "
                      "closing camera (stream_dead)", flush=True)
                zed.close()
                return
            if elapsed >= args.duration:
                print("[watchdog] duration elapsed, forcing camera close "
                      "to unblock grab()", flush=True)
                zed.close()
                return

    watchdog = threading.Thread(target=_watchdog, daemon=True)
    watchdog.start()

    # Only honor stdin-EOF early-stop when stdin is a real pipe (orchestrator);
    # /dev/null and ttys report immediate EOF and would kill the loop instantly.
    stdin_is_pipe = stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode) if not sys.stdin.closed else False

    last_grab_warn = 0.0
    try:
        while True:
            if time.time() - start >= args.duration:
                break
            if state["stream_dead"]:
                break
            if stdin_is_pipe and _stdin_closed():
                break

            grab_status = zed.grab(runtime)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                now = time.time()
                if now - last_grab_warn >= 5.0:
                    last_grab_warn = now
                    print(f"grab failed: {grab_status}", flush=True)
                if now - start >= args.duration or state["stream_dead"]:
                    break
                continue

            state["frames"] += 1
            if state["t_first_grab"] is None:
                state["t_first_grab"] = time.time() - start
                print(f"zed_single: first frame after {state['t_first_grab']:.1f}s",
                      flush=True)
            zed.retrieve_bodies(bodies, body_runtime)
            wall = time.time()

            if state["frames"] % 30 == 1:
                print(f"[diag] frame {state['frames']}, bodies={len(bodies.body_list)}",
                      flush=True)

            if args.save_frame and state["frames"] == 30:
                try:
                    import cv2  # lazy: only needed for the diagnostic
                    img = sl.Mat()
                    zed.retrieve_image(img, sl.VIEW.LEFT)
                    cv2.imwrite(args.save_frame, img.get_data())
                    print(f"zed_single: saved frame -> {args.save_frame}", flush=True)
                except Exception as e:
                    print(f"zed_single: save-frame failed: {e}", flush=True)

            if bodies.body_list:
                state["frames_with_bodies"] += 1
            for body in bodies.body_list:
                kp = body.keypoint                  # Nx3 (metres, Y-up)
                kpc = body.keypoint_confidence      # per-joint confidence
                for idx in range(len(kp)):
                    x, y, z = float(kp[idx][0]), float(kp[idx][1]), float(kp[idx][2])
                    conf = float(kpc[idx]) if idx < len(kpc) else float("nan")
                    rows.append((state["frames"], wall, int(body.id),
                                 str(body.tracking_state),
                                 idx, joint_name(idx), x, y, z, conf))

            if not announced and len(bodies.body_list) > 0:
                announced = True
                print("ZED_SINGLE_READY", flush=True)

    finally:
        watchdog_stop.set()
        out = os.path.join(REPO, "results", "layouts", f"zed_single_{args.layout_id}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "wall_clock", "body_id", "tracking_state",
                        "joint_idx", "joint_name", "x", "y", "z", "confidence"])
            w.writerows(rows)
        meta = {
            "layout_id": args.layout_id,
            "port": args.port,
            "model": args.model,
            "conf": args.conf,
            "frames_grabbed": state["frames"],
            "frames_with_bodies": state["frames_with_bodies"],
            "t_first_grab_s": state["t_first_grab"],
            "rows": len(rows),
            "stream_dead": state["stream_dead"],
        }
        meta_path = os.path.join(REPO, "results", "layouts",
                                 f"zed_single_{args.layout_id}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"zed_single: wrote {len(rows)} rows (port {args.port}) -> {out}")
        print(f"ZED_SUMMARY frames={state['frames']} "
              f"frames_with_bodies={state['frames_with_bodies']} rows={len(rows)}",
              flush=True)
        # The watchdog may have already force-closed the camera; cleanup on an
        # already-closed camera can raise — never mask the real outcome.
        try:
            zed.disable_body_tracking()
        except Exception:
            pass
        try:
            zed.close()
        except Exception:
            pass

    if state["stream_dead"]:
        print("RUN_FAILED stream_dead")
        sys.exit(2)
    if state["frames"] > 0 and len(rows) == 0:
        print(f"RUN_FAILED no_bodies frames={state['frames']}")
        sys.exit(3)
    sys.exit(0)


def _stdin_closed():
    """True if stdin has hit EOF (parent closed the pipe). Non-blocking."""
    if sys.stdin is None or sys.stdin.closed:
        return True
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            # readable: if it returns empty, that's EOF
            return sys.stdin.read(1) == ""
    except Exception:
        return False
    return False


if __name__ == "__main__":
    main()
