#!/usr/bin/env python3
# scripts/preflight.py
#
# Pre-run cleanup gate for the ZED <-> Isaac streaming pipeline. RUNS UNDER
# SYSTEM python3 (stdlib + yaml only). Importable by scripts/run_pipeline.py and
# later by sweep.py / camera_rig.evaluate_layout.
#
# Why this exists (HANDOFF.md issues B/C): crashed or zombie processes squat the
# stream UDP ports (30000-30003) and leave stale SHM-Boost topics in /dev/shm
# (sl_local_video_* / sl_local_data_* — the SDK itself documents cleanup as
# `rm /dev/shm/sl_local_*`). A receiver started against a dirty machine falls
# into the UDP port cascade (30000 -> 30002 -> 30004 "Backward compatibility")
# and never grabs a frame. This script guarantees a clean slate.
#
# Sentinels: PREFLIGHT_OK / PREFLIGHT_FAILED <reason>

import argparse
import glob
import os
import signal
import subprocess
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Script basenames whose stale instances we are allowed to kill. Matched against
# `pgrep -af` output, never a bare "python" match.
_KILLABLE_SCRIPTS = ("zed_single.py", "zed_fusion.py", "run_episode.py")

# SDK-documented stale stream topics (string present in libsl_zed.so:
# "rm /dev/shm/sl_local_*"). carb-RStringInternals-* files are normal Carbonite
# leftovers and must be left alone.
_SHM_PATTERNS = ("/dev/shm/sl_local_*", "/dev/shm/sem.sl_local_*")


def load_cfgs(machine):
    with open(os.path.join(REPO, "config", "experiment.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(REPO, "config", f"machine.{machine}.yaml")) as f:
        machine_cfg = yaml.safe_load(f)
    return cfg, machine_cfg


def stream_ports(cfg):
    """All ports a run can touch: each camera's RTP port + its control port (+1),
    plus the receiver's +2 fallback cascade landing spots."""
    base = sorted({cfg["cam_a"]["port"], cfg["cam_b"]["port"]})
    ports = set()
    for p in base:
        ports.update({p, p + 1})
    # the receiver cascade can land on max+2 / max+3 during a broken run
    ports.update({max(base) + 2, max(base) + 3})
    return sorted(ports)


def _ancestors():
    """PIDs of this process and all its ancestors (never kill our own
    shell/orchestrator — their cmdlines can mention script names, e.g. a
    `bash -c 'python3 scripts/run_pipeline.py ...'` wrapper)."""
    pids = set()
    pid = os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                # field 4 is ppid; comm (field 2) may contain spaces -> split
                # after the closing paren.
                stat = f.read()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def _pgrep():
    """Return [(pid, cmdline)] for every process except us and our ancestors."""
    out = subprocess.run(["pgrep", "-af", "."], capture_output=True, text=True).stdout
    skip = _ancestors()
    procs = []
    for line in out.splitlines():
        try:
            pid_s, cmd = line.split(" ", 1)
            pid = int(pid_s)
        except ValueError:
            continue
        if pid not in skip:
            procs.append((pid, cmd))
    return procs


def find_stale(machine_cfg):
    """PIDs of stale pipeline processes: our scripts + crashed Isaac kit procs."""
    kit_dir = os.path.join(os.path.dirname(machine_cfg["isaac_python"]), "kit")
    stale = []
    for pid, cmd in _pgrep():
        if any(s in cmd for s in _KILLABLE_SCRIPTS) or kit_dir in cmd:
            stale.append((pid, cmd))
    return stale


def kill_stale(machine_cfg, grace_s=5):
    stale = find_stale(machine_cfg)
    for pid, cmd in stale:
        print(f"preflight: SIGTERM {pid}: {cmd[:100]}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if stale:
        deadline = time.time() + grace_s
        while time.time() < deadline and find_stale(machine_cfg):
            time.sleep(0.5)
        for pid, cmd in find_stale(machine_cfg):
            print(f"preflight: SIGKILL {pid}: {cmd[:100]}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return len(stale)


def busy_ports(ports):
    """Subset of `ports` with a live UDP/TCP socket (ss -tulnH)."""
    out = subprocess.run(["ss", "-tulnH"], capture_output=True, text=True).stdout
    busy = set()
    for line in out.splitlines():
        for p in ports:
            if f":{p} " in line or line.rstrip().endswith(f":{p}"):
                busy.add(p)
    return sorted(busy)


def wait_ports_free(ports, wait_s=30):
    deadline = time.time() + wait_s
    while True:
        busy = busy_ports(ports)
        if not busy:
            return []
        if time.time() >= deadline:
            return busy
        time.sleep(1.0)


def clean_shm():
    removed = []
    for pat in _SHM_PATTERNS:
        for path in glob.glob(pat):
            try:
                os.remove(path)
                removed.append(path)
            except OSError as e:
                print(f"preflight: could not remove {path}: {e}")
    return removed


def preflight(cfg, machine_cfg, wait_s=30):
    """Full gate. Returns True on clean machine, False otherwise (after printing
    the PREFLIGHT_* sentinel)."""
    n_killed = kill_stale(machine_cfg)
    if n_killed:
        print(f"preflight: killed {n_killed} stale process(es)")

    ports = stream_ports(cfg)
    busy = wait_ports_free(ports, wait_s=wait_s)
    if busy:
        print(f"PREFLIGHT_FAILED ports_busy {busy}")
        return False

    removed = clean_shm()
    if removed:
        print(f"preflight: removed stale SHM topics: {removed}")

    print("PREFLIGHT_OK")
    return True


def main():
    ap = argparse.ArgumentParser(description="Clean stale ZED/Isaac stream state")
    ap.add_argument("--machine", default="laptop")
    ap.add_argument("--wait", type=float, default=30.0, help="max s to wait for ports")
    args = ap.parse_args()

    cfg, machine_cfg = load_cfgs(args.machine)
    sys.exit(0 if preflight(cfg, machine_cfg, wait_s=args.wait) else 1)


if __name__ == "__main__":
    main()
