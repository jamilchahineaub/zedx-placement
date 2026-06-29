#!/usr/bin/env python3
# scripts/run_pipeline.py
#
# One-command pipeline orchestrator. RUNS UNDER SYSTEM python3.
#
#   preflight (kill stale procs / free ports / clean SHM)
#   -> launch Isaac run_episode (background, logged)
#   -> wait for STREAMING_STARTED + both ZED streamers initialized
#   -> run zed/zed_single.py against cam A's port (retry once on cam B's port
#      if the stream is dead -- isolates port-asymmetric SHM discovery)
#   -> optionally run zed/zed_fusion.py (--mode fusion|both)
#   -> wait for EPISODE_DONE (ground-truth CSV) and clean shutdown
#
# Sentinels: PIPELINE_OK rows=<n> port=<p> | PIPELINE_FAILED <reason>
# Module functions are import-reusable by sweep.py / camera_rig.evaluate_layout.

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.preflight import load_cfgs, preflight, clean_shm  # noqa: E402

LOGS_DIR = os.path.join(REPO, "results", "logs")

_RE_INIT = re.compile(r"Initializing streamer with ID (\d+) on port (\d+)")
_RE_OK = re.compile(r"ZED Streamer initialized successfully with ID (\d+)")
_ERR_LOOP = "Error during zed streamer initialization"
_ERR_RTP = "failed to create RTP Session"


def _ts():
    return time.strftime("%H%M%S")


def shm_snapshot():
    topics = sorted(p for p in os.listdir("/dev/shm") if "sl_local" in p)
    print(f"pipeline: /dev/shm sl_local topics: {topics or 'NONE'}", flush=True)
    return topics


def launch_isaac(h, r, rel_az, subject_name, layout_id, machine, machine_cfg,
                 duration, cams="both", transport=None, overhead_h=None,
                 ring_c_az=None, chest_tags=False, marker_front=None, marker_back=None,
                 spin_deg_s=None):
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"isaac_{layout_id}_{_ts()}.log")
    cmd = [machine_cfg["isaac_python"], os.path.join(REPO, "isaac", "run_episode.py"),
           "--h", str(h), "--r", str(r), "--rel-az", str(rel_az),
           "--subject-name", subject_name, "--duration", str(duration),
           "--layout-id", layout_id, "--machine", machine, "--cams", cams]
    if transport:
        cmd += ["--transport", transport]
    if overhead_h is not None:
        cmd += ["--overhead-h", str(overhead_h)]
    if ring_c_az is not None:
        cmd += ["--ring-c-az", str(ring_c_az)]
    if chest_tags:
        cmd += ["--chest-tags", "--marker-front", marker_front, "--marker-back", marker_back]
    if spin_deg_s is not None:
        cmd += ["--spin-deg-s", str(spin_deg_s)]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=REPO,
                            start_new_session=True)
    print(f"pipeline: Isaac launched pid={proc.pid}, log={log_path}", flush=True)
    return proc, log_path


def wait_for_streaming(log_path, proc, expected_ports, boot_timeout=240.0,
                       grace=5.0):
    """Block until both streamers are up. Returns {'id_port': {id: port}} or
    raises RuntimeError with a reason string.

    Readiness = STREAMING_STARTED + an "Initializing streamer with ID N on
    port P" line for every expected port + NO streamer error within `grace`
    seconds after the last init line. (The plugin's explicit success line
    "ZED Streamer initialized successfully" is logged at INFO level, which
    carb filters out of the captured log — absence-of-error is the reliable
    observable. A genuinely failed init re-attempts EVERY FRAME with a new ID,
    so failures show up within the grace window.)"""
    deadline = time.time() + boot_timeout
    seen_streaming_started = False
    id_port = {}
    err_count = 0
    pos = 0
    last_init_t = None

    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"isaac_exited_early rc={proc.returncode}")
        try:
            with open(log_path) as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except FileNotFoundError:
            chunk = ""
        for line in chunk.splitlines():
            if "STREAMING_STARTED" in line:
                seen_streaming_started = True
            m = _RE_INIT.search(line)
            if m:
                id_port[int(m.group(1))] = int(m.group(2))
                last_init_t = time.time()
            if _ERR_LOOP in line or _ERR_RTP in line:
                err_count += 1
            if "GEO_SKIPPED" in line:
                raise RuntimeError("geo_skipped")
            if "EPISODE_DONE" in line:
                raise RuntimeError("episode_finished_before_streaming_confirmed")
        if err_count > 5:
            raise RuntimeError(f"streamer_init_error_loop ({err_count} errors)")
        live_ports = set(id_port.values())
        if (seen_streaming_started and expected_ports.issubset(live_ports)
                and err_count == 0 and last_init_t is not None
                and time.time() - last_init_t >= grace):
            print(f"pipeline: streaming confirmed (no errors {grace:.0f}s after "
                  f"init), streamer ID->port map: {id_port}", flush=True)
            return {"id_port": id_port}
        time.sleep(1.0)

    raise RuntimeError(
        f"streaming_timeout started={seen_streaming_started} "
        f"init={id_port} errors={err_count}")


def run_zed_single(port, layout_id, machine_cfg, duration=20.0, model="accurate",
                   conf=20, extra_args=(), overall_timeout=None):
    """Returns dict(rc, rows, meta). overall_timeout is a hard subprocess kill."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"zed_single_{layout_id}_{_ts()}.log")
    cmd = [machine_cfg.get("zed_python", "python3"),
           os.path.join(REPO, "zed", "zed_single.py"),
           "--port", str(port), "--layout-id", layout_id,
           "--duration", str(duration), "--model", model, "--conf", str(conf),
           *extra_args]
    if overall_timeout is None:
        overall_timeout = duration + 90  # open(<=2x30s) + model load + margin
    print(f"pipeline: starting zed_single on port {port} "
          f"(capture {duration}s, hard cap {overall_timeout:.0f}s)", flush=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=REPO, text=True)
        try:
            out, _ = proc.communicate(timeout=overall_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            out += "\npipeline: KILLED by overall_timeout\n"
        log_f.write(out)
    print(out, flush=True)

    rows = 0
    csv_path = os.path.join(REPO, "results", "layouts", f"zed_single_{layout_id}.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = max(0, sum(1 for _ in f) - 1)
    meta = {}
    meta_path = os.path.join(REPO, "results", "layouts",
                             f"zed_single_{layout_id}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return {"rc": proc.returncode, "rows": rows, "meta": meta, "csv": csv_path,
            "log": log_path}


def run_zed_fusion(fusion_config, layout_id, machine_cfg, duration=20.0,
                   model="accurate", conf=20, overall_timeout=None, detect_tags=False):
    """Run zed/zed_fusion.py. Returns dict(rc, rows, csv)."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"zed_fusion_{layout_id}_{_ts()}.log")
    cmd = [machine_cfg.get("zed_python", "python3"),
           os.path.join(REPO, "zed", "zed_fusion.py"),
           "--fusion-config", fusion_config, "--layout-id", layout_id,
           "--duration", str(duration), "--model", model, "--conf", str(conf)]
    if detect_tags:
        cmd += ["--detect-tags"]
    if overall_timeout is None:
        overall_timeout = duration + 150  # two opens + two model loads + margin
    print(f"pipeline: starting zed_fusion (capture {duration}s, "
          f"hard cap {overall_timeout:.0f}s)", flush=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=REPO, text=True)
        try:
            out, _ = proc.communicate(timeout=overall_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            out += "\npipeline: KILLED by overall_timeout\n"
        log_f.write(out)
    print(out, flush=True)

    rows = 0
    csv_path = os.path.join(REPO, "results", "layouts", f"zed_pred_{layout_id}.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = max(0, sum(1 for _ in f) - 1)
    return {"rc": proc.returncode, "rows": rows, "csv": csv_path, "log": log_path}


def run_zed_tag_detect(layout_id, machine_cfg, duration=20.0, overall_timeout=None):
    """Run zed/zed_tag_detect.py (ArUco detection on the 3 streams). Returns dict(rc, csv)."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"zed_tag_{layout_id}_{_ts()}.log")
    cmd = [machine_cfg.get("zed_python", "python3"),
           os.path.join(REPO, "zed", "zed_tag_detect.py"),
           "--layout-id", layout_id, "--duration", str(duration)]
    if overall_timeout is None:
        overall_timeout = duration + 150
    print(f"pipeline: starting zed_tag_detect (capture {duration}s)", flush=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=REPO, text=True)
        try:
            out, _ = proc.communicate(timeout=overall_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            out += "\npipeline: KILLED by overall_timeout\n"
        log_f.write(out)
    print(out, flush=True)
    csv_path = os.path.join(REPO, "results", "layouts", f"tag_detect_{layout_id}.csv")
    rows = 0
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = max(0, sum(1 for _ in f) - 1)
    return {"rc": proc.returncode, "rows": rows, "csv": csv_path, "log": log_path}


def shutdown_isaac(proc, log_path, grace=120.0):
    """Prefer the natural exit (EPISODE_DONE -> app.close()); escalate after grace."""
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"pipeline: Isaac exited rc={proc.returncode}", flush=True)
            return True
        time.sleep(2.0)
    pgid = os.getpgid(proc.pid)
    print("pipeline: Isaac still running past grace, SIGINT process group", flush=True)
    os.killpg(pgid, signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        print("pipeline: escalating to SIGKILL", flush=True)
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()
    clean_shm()
    return False


def wait_episode_done(log_path, proc, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(log_path) as f:
                if "EPISODE_DONE" in f.read():
                    return True
        except FileNotFoundError:
            pass
        if proc.poll() is not None:
            return False
        time.sleep(2.0)
    return False


def main():
    ap = argparse.ArgumentParser(description="ZED-X placement: one-command pipeline")
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--r", type=float, default=2.5)
    ap.add_argument("--rel-az", type=float, default=90.0)
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--layout-id", required=True)
    ap.add_argument("--machine", default="laptop")
    ap.add_argument("--episode-duration", type=float, default=150.0)
    ap.add_argument("--capture-duration", type=float, default=20.0)
    ap.add_argument("--mode", choices=["single", "fusion", "both"], default="single")
    ap.add_argument("--model", choices=["fast", "accurate"], default="accurate")
    ap.add_argument("--conf", type=int, default=20)
    ap.add_argument("--cams", choices=["both", "a", "b"], default="both")
    ap.add_argument("--transport", choices=["BOTH", "NETWORK", "IPC"], default=None)
    ap.add_argument("--overhead-h", type=float, default=None,
                    help="add a centered overhead (nadir) cam C at this height "
                         "(3-cam fusion). Only valid with --mode fusion.")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    cfg, machine_cfg = load_cfgs(args.machine)
    port_a, port_b = cfg["cam_a"]["port"], cfg["cam_b"]["port"]
    expected = {port_a, port_b} if args.cams == "both" else (
        {port_a} if args.cams == "a" else {port_b})
    if args.overhead_h is not None:
        expected.add(cfg["cam_c"]["port"])

    # 1) Preflight
    if not args.skip_preflight:
        if not preflight(cfg, machine_cfg):
            print("PIPELINE_FAILED preflight")
            sys.exit(1)

    # 2) Isaac
    proc, log_path = launch_isaac(args.h, args.r, args.rel_az, args.subject_name,
                                  args.layout_id, args.machine, machine_cfg,
                                  args.episode_duration, cams=args.cams,
                                  transport=args.transport,
                                  overhead_h=args.overhead_h)
    ok = False
    reason = "unknown"
    result = {}
    try:
        # 3) Wait for streamers
        try:
            stream_info = wait_for_streaming(log_path, proc, expected)
        except RuntimeError as e:
            print(f"pipeline: streaming never came up: {e}", flush=True)
            print(f"--- tail of {log_path} ---", flush=True)
            with open(log_path) as f:
                print("".join(f.readlines()[-25:]), flush=True)
            raise

        shm_snapshot()

        # 4) Receiver(s)
        if args.mode in ("single", "both"):
            first_port = port_a if args.cams != "b" else port_b
            result = run_zed_single(first_port, args.layout_id, machine_cfg,
                                    duration=args.capture_duration,
                                    model=args.model, conf=args.conf)
            if result["rc"] in (1, 2, 142) and args.cams == "both":
                print(f"pipeline: port {first_port} dead (rc={result['rc']}); "
                      f"retrying on {port_b}", flush=True)
                shm_snapshot()
                result = run_zed_single(port_b, f"{args.layout_id}_pB", machine_cfg,
                                        duration=args.capture_duration,
                                        model=args.model, conf=args.conf)
                result["port"] = port_b
            else:
                result["port"] = first_port
            ok = result["rc"] == 0 and result["rows"] > 0
            reason = f"zed_single rc={result['rc']} rows={result['rows']}"

        if args.mode in ("fusion", "both") and (ok or args.mode == "fusion"):
            fusion_cfg_path = os.path.join(
                REPO, "results", "layouts", f"fusion_config_{args.layout_id}.json")
            if not os.path.exists(fusion_cfg_path):
                gen_cmd = [machine_cfg.get("zed_python", "python3"),
                           os.path.join(REPO, "zed", "make_fusion_config.py"),
                           "--out", fusion_cfg_path, "--h", str(args.h),
                           "--r", str(args.r), "--rel-az", str(args.rel_az)]
                if args.overhead_h is not None:
                    gen_cmd += ["--overhead-h", str(args.overhead_h)]
                gen = subprocess.run(gen_cmd, cwd=REPO, capture_output=True, text=True)
                print(gen.stdout + gen.stderr, flush=True)
            fres = run_zed_fusion(fusion_cfg_path, args.layout_id, machine_cfg,
                                  duration=args.capture_duration,
                                  model=args.model, conf=args.conf)
            result["fusion"] = fres
            ok = fres["rc"] == 0 and fres["rows"] > 0
            reason = f"zed_fusion rc={fres['rc']} rows={fres['rows']}"

        # 5) Let the episode finish so the ground-truth CSV is written.
        print("pipeline: waiting for EPISODE_DONE (ground truth CSV)...", flush=True)
        wait_episode_done(log_path, proc, timeout=args.episode_duration + 60)
    except RuntimeError as e:
        reason = str(e)
    finally:
        shutdown_isaac(proc, log_path, grace=90)

    if ok:
        port = result.get("port", "?")
        rows = result.get("fusion", result).get("rows", result.get("rows", 0))
        print(f"PIPELINE_OK rows={rows} port={port}")
        sys.exit(0)
    print(f"PIPELINE_FAILED {reason}")
    sys.exit(1)


if __name__ == "__main__":
    main()
