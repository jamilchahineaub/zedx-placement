# isaac/camera_rig.py
#
# Pure-math camera placement for the dual-camera ZED-X ring.
# NO omni / pxr / isaacsim imports — fully testable with plain python3.
#
# The look-at and positioning math is copied VERBATIM from
# zed/make_fusion_config.py so the Isaac side and the fusion side never diverge.
# If you change the math here, change it there too (and vice versa).
#
# Coordinate system: right-handed Z-up, metres. World up = +Z.
#
# This module is the SWEEP BOUNDARY: sweep.py calls only evaluate_layout().

import math


# ---------------------------------------------------------------------------
# Copied verbatim from zed/make_fusion_config.py (keep in sync)
# ---------------------------------------------------------------------------

def rotation_matrix_from_look_at(cam_pos, target_pos):
    """
    3x3 rotation matrix where rows are camera X (right), Y (up), Z (forward).
    Camera looks toward target. World up = Z axis.
    """
    cx, cy, cz = cam_pos
    tx, ty, tz = target_pos

    fx = tx - cx; fy = ty - cy; fz = tz - cz
    fl = math.sqrt(fx*fx + fy*fy + fz*fz)
    if fl < 1e-9:
        raise ValueError(f"Camera {cam_pos} and target {target_pos} are the same point")
    fx /= fl; fy /= fl; fz /= fl

    # World up = Z. If forward is nearly vertical, fall back to Y
    if abs(fz) > 0.999:
        ux, uy, uz = 0.0, 1.0, 0.0
    else:
        ux, uy, uz = 0.0, 0.0, 1.0

    # Right = forward x up
    rx = fy*uz - fz*uy
    ry = fz*ux - fx*uz
    rz = fx*uy - fy*ux
    rl = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx /= rl; ry /= rl; rz /= rl

    # Recompute up = right x forward
    upx = ry*fz - rz*fy
    upy = rz*fx - rx*fz
    upz = rx*fy - ry*fx

    return [
        [rx,  ry,  rz ],
        [upx, upy, upz],
        [fx,  fy,  fz ],
    ]


def camera_position(azimuth_deg, radius, height, subject=(0.0, 0.0, 0.0)):
    """World position of a camera on the ring around subject."""
    az = math.radians(azimuth_deg)
    return [
        subject[0] + radius * math.cos(az),
        subject[1] + radius * math.sin(az),
        height,
    ]


def tilt_angle(h, r, aim_height_m):
    """
    Downward tilt of a camera at height h, radius r, aiming at aim_height_m.
    arctan((h - aim_height_m) / r), in degrees.
    (Same formula as make_fusion_config.compute_tilt_deg.)
    """
    return math.degrees(math.atan((h - aim_height_m) / r))


# ---------------------------------------------------------------------------
# New for camera_rig
# ---------------------------------------------------------------------------

def convergence_angle(pos_a, pos_b, subject=(0.0, 0.0, 0.0)):
    """
    Angle in degrees between the (cam_a -> subject) and (cam_b -> subject)
    ray vectors. This is the true 3D convergence (baseline angle at the
    subject), so it accounts for camera height, not just azimuth separation.
    """
    ax = subject[0] - pos_a[0]; ay = subject[1] - pos_a[1]; az = subject[2] - pos_a[2]
    bx = subject[0] - pos_b[0]; by = subject[1] - pos_b[1]; bz = subject[2] - pos_b[2]

    na = math.sqrt(ax*ax + ay*ay + az*az)
    nb = math.sqrt(bx*bx + by*by + bz*bz)
    if na < 1e-9 or nb < 1e-9:
        raise ValueError("A camera is coincident with the subject")

    dot = (ax*bx + ay*by + az*bz) / (na * nb)
    dot = max(-1.0, min(1.0, dot))   # clamp for numerical safety
    return math.degrees(math.acos(dot))


def overhead_position(overhead_h, center=(0.0, 0.0)):
    """World position of the centered overhead (nadir) camera: workspace centre
    (x,y) at the given height. Looks straight down at the aim point."""
    return [center[0], center[1], float(overhead_h)]


def pairwise_convergences(positions, subject=(0.0, 0.0, 0.0)):
    """Pairwise convergence angles (deg) for a dict of {name: [x,y,z]} cameras.
    Returns {(name_i, name_j): angle} for every unordered pair, where the angle
    is convergence_angle(pos_i, pos_j, subject)."""
    names = list(positions.keys())
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            out[(ni, nj)] = convergence_angle(positions[ni], positions[nj], subject)
    return out


def is_valid_layout(h, r, cfg):
    """
    True if the downward tilt for this (h, r) is below cfg['max_tilt_deg'].
    Aim height comes from cfg['aim_height_m'].
    """
    aim = cfg["aim_height_m"]
    return tilt_angle(h, r, aim) < cfg["max_tilt_deg"]


def evaluate_layout(h, r, rel_az_deg, subject_pos, cfg,
                    machine="laptop", layout_id=None, subject_pos_name="center",
                    episode_duration=240.0, capture_duration=20.0,
                    mode="fusion", overhead_h=None):
    """
    THE SWEEP BOUNDARY. sweep.py calls only this function.

    Runs one full episode end-to-end and returns the metrics dict for one
    results.csv row:
      preflight -> fusion config -> Isaac episode (streaming) -> ZED receiver
      (fusion or single) -> analysis/metrics.compute_metrics.

    RUNS UNDER SYSTEM python3 ONLY (it shells out to Isaac python for the
    episode; never call this from inside Isaac python). All orchestration
    building blocks live in scripts/run_pipeline.py and analysis/metrics.py —
    this function only wires them together.
    """
    import json
    import os
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (repo, os.path.join(repo, "analysis")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from scripts import run_pipeline
    from scripts.preflight import load_cfgs, preflight as run_preflight
    import metrics as metrics_mod

    _, machine_cfg = load_cfgs(machine)
    if layout_id is None:
        layout_id = f"h{h}_r{r}_az{int(rel_az_deg)}_{subject_pos_name}"

    if not is_valid_layout(h, r, cfg):
        raise ValueError(f"invalid layout (tilt >= max_tilt_deg): h={h} r={r}")

    if not run_preflight(cfg, machine_cfg):
        raise RuntimeError("preflight failed")

    # Fusion config for this layout (needed by zed_fusion + documents poses).
    fusion_cfg_path = os.path.join(repo, "results", "layouts",
                                   f"fusion_config_{layout_id}.json")
    mk_cmd = [machine_cfg.get("zed_python", "python3"),
              os.path.join(repo, "zed", "make_fusion_config.py"),
              "--out", fusion_cfg_path, "--h", str(h), "--r", str(r),
              "--rel-az", str(rel_az_deg),
              "--subject", " ".join(str(v) for v in subject_pos)]
    if overhead_h is not None:
        mk_cmd += ["--overhead-h", str(overhead_h)]
    subprocess.run(mk_cmd, cwd=repo, check=True, capture_output=True, text=True)

    proc, log_path = run_pipeline.launch_isaac(
        h, r, rel_az_deg, subject_pos_name, layout_id, machine, machine_cfg,
        episode_duration, overhead_h=overhead_h)
    try:
        expected = {cfg["cam_a"]["port"], cfg["cam_b"]["port"]}
        if overhead_h is not None:
            expected.add(cfg["cam_c"]["port"])
        run_pipeline.wait_for_streaming(log_path, proc, expected)

        if mode == "fusion":
            res = run_pipeline.run_zed_fusion(
                fusion_cfg_path, layout_id, machine_cfg,
                duration=capture_duration)
            pred_csv = res["csv"]
            meta_path = os.path.join(repo, "results", "layouts",
                                     f"zed_pred_{layout_id}_meta.json")
        else:
            res = run_pipeline.run_zed_single(cfg["cam_a"]["port"], layout_id,
                                              machine_cfg,
                                              duration=capture_duration)
            pred_csv = res["csv"]
            meta_path = os.path.join(repo, "results", "layouts",
                                     f"zed_single_{layout_id}_meta.json")
        if res["rc"] != 0 or res["rows"] == 0:
            raise RuntimeError(f"receiver failed rc={res['rc']} rows={res['rows']}")

        run_pipeline.wait_episode_done(log_path, proc,
                                       timeout=episode_duration + 60)
    finally:
        run_pipeline.shutdown_isaac(proc, log_path, grace=90)

    gt_csv = os.path.join(repo, "results", "layouts",
                          f"ground_truth_{layout_id}.csv")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return metrics_mod.compute_metrics(
        gt_csv, pred_csv, meta, h, r, rel_az_deg, cfg,
        subject_pos=tuple(subject_pos), mode=mode,
        subject_pos_name=subject_pos_name, overhead_h=overhead_h)
