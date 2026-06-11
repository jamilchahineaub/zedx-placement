# analysis/metrics.py
#
# Accuracy metrics: ZED body-tracking predictions vs Isaac ground truth.
# Plain python3 (csv/json/math + yaml only — no omni, no pyzed).
#
# Frame/alignment strategy (static character — animation is frozen, confirmed
# bit-identical pelvis across frames): TIME-AVERAGE both skeletons per joint,
# then compare the averaged skeletons. This is robust to detector jitter and
# avoids cross-process wall-clock alignment entirely. When motion lands
# (omni.anim.people follow-up) this module needs a per-frame association pass;
# jitter_variance / id_drops are NaN until then.
#
# Coordinate frames:
#   GT CSV     : Isaac Z-up world.
#   fusion CSV : the fusion world frame = the Y-up conversion of Isaac world
#                that make_fusion_config writes into the camera poses
#                (zed = P @ isaac with P(x,y,z) = (-y,-z,x)), so
#                isaac = (z_zed, -x_zed, -y_zed).
#   single CSV : the camera's own Y-up frame (world anchored at the camera at
#                init). p_isaac = R_wc @ p_cam + cam_pos with R_wc the proper
#                world<-camera rotation (columns [right, up, -forward]) reused
#                from zed/make_fusion_config.proper_rotation_world_from_cam.

import csv
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "isaac"), os.path.join(_REPO, "zed")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import camera_rig                      # noqa: E402  (isaac/ — pure math)
import geo_prescreener                 # noqa: E402  (analysis/)
import joint_map                       # noqa: E402  (analysis/)
from make_fusion_config import proper_rotation_world_from_cam  # noqa: E402


NAN = float("nan")


# ---------------------------------------------------------------------------
# CSV loading (time-averaged skeletons)
# ---------------------------------------------------------------------------

def load_gt_average(gt_csv, joint_filter=None):
    """Average each joint's world position over all frames.
    Returns ({joint_name: (x,y,z)}, n_frames)."""
    sums, counts = {}, {}
    times = set()
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            name = row["joint_name"]
            if joint_filter is not None and name not in joint_filter:
                continue
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            sx, sy, sz = sums.get(name, (0.0, 0.0, 0.0))
            sums[name] = (sx + x, sy + y, sz + z)
            counts[name] = counts.get(name, 0) + 1
            times.add(row["sim_time"])
    avg = {n: (s[0] / counts[n], s[1] / counts[n], s[2] / counts[n])
           for n, s in sums.items()}
    return avg, len(times)


def load_pred_average(pred_csv, conf_min=0.0):
    """Average each ZED joint over all rows with confidence >= conf_min.
    Returns ({zed_joint_name: (x,y,z)}, n_rows_used). NaN keypoints skipped."""
    sums, counts = {}, {}
    used = 0
    with open(pred_csv) as f:
        for row in csv.DictReader(f):
            try:
                conf = float(row["confidence"])
            except ValueError:
                continue
            if math.isnan(conf) or conf < conf_min:
                continue
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            if any(math.isnan(v) for v in (x, y, z)):
                continue
            name = row["joint_name"]
            sx, sy, sz = sums.get(name, (0.0, 0.0, 0.0))
            sums[name] = (sx + x, sy + y, sz + z)
            counts[name] = counts.get(name, 0) + 1
            used += 1
    avg = {n: (s[0] / counts[n], s[1] / counts[n], s[2] / counts[n])
           for n, s in sums.items()}
    return avg, used


# ---------------------------------------------------------------------------
# Coordinate transforms (into Isaac Z-up world)
# ---------------------------------------------------------------------------

def fused_to_isaac(p):
    """Invert make_fusion_config's P: zed=(-y,-z,x)  =>  isaac=(z,-x,-y)."""
    return (p[2], -p[0], -p[1])


def single_cam_to_isaac(p_cam, cam_pos, aim_point):
    """Camera-frame Y-up point -> Isaac Z-up world.
    R_wc columns are [right, up, -forward] (camera looks down its own -Z)."""
    R = proper_rotation_world_from_cam(cam_pos, aim_point)
    x, y, z = p_cam
    return (
        R[0][0] * x + R[0][1] * y + R[0][2] * z + cam_pos[0],
        R[1][0] * x + R[1][1] * y + R[1][2] * z + cam_pos[1],
        R[2][0] * x + R[2][1] * y + R[2][2] * z + cam_pos[2],
    )


def _gravity_alignment_deg(pred_avg):
    """Diagnostic: angle between the predicted hips->neck axis and +Y in RAW
    ZED coords. ~0 deg => SDK gravity-aligned the frame; ~camera-tilt deg =>
    pure camera frame (our default transform assumption)."""
    need = ("NECK", "LEFT_HIP", "RIGHT_HIP")
    if any(n not in pred_avg for n in need):
        return NAN
    neck = pred_avg["NECK"]
    hip = tuple((pred_avg["LEFT_HIP"][i] + pred_avg["RIGHT_HIP"][i]) / 2.0
                for i in range(3))
    v = tuple(neck[i] - hip[i] for i in range(3))
    norm = math.sqrt(sum(c * c for c in v))
    if norm < 1e-9:
        return NAN
    cosang = max(-1.0, min(1.0, v[1] / norm))
    return math.degrees(math.acos(cosang))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mpjpe_pck(gt_avg, pred_avg_isaac):
    """gt_avg keyed by Isaac names; pred_avg_isaac keyed by ZED names but
    already transformed into Isaac world. Returns (mpjpe_mm, pck30, pck50,
    per_joint_mm dict)."""
    errs = {}
    for zed_name, isaac_name in joint_map.mapped_pairs():
        if zed_name not in pred_avg_isaac or isaac_name not in gt_avg:
            continue
        p, g = pred_avg_isaac[zed_name], gt_avg[isaac_name]
        e_mm = 1000.0 * math.sqrt(sum((p[i] - g[i]) ** 2 for i in range(3)))
        errs[zed_name] = e_mm
    if not errs:
        return NAN, NAN, NAN, {}
    vals = list(errs.values())
    mpjpe = sum(vals) / len(vals)
    pck30 = sum(1 for e in vals if e <= 30.0) / len(vals)
    pck50 = sum(1 for e in vals if e <= 50.0) / len(vals)
    return mpjpe, pck30, pck50, errs


def compute_metrics(gt_csv, pred_csv, meta, h, r, rel_az_deg, cfg,
                    subject_pos=(0.0, 0.0, 0.0), mode="fusion",
                    conf_min=0.0, subject_pos_name="center"):
    """
    Full results.csv row (every column from CLAUDE.md).

    gt_csv   : ground_truth_<id>.csv (Isaac Z-up world)
    pred_csv : zed_pred_<id>.csv (fusion frame) or zed_single_<id>.csv (camera frame)
    meta     : dict from the receiver's _meta.json (frames_grabbed,
               frames_with_bodies) — or {} (coverage = NaN)
    mode     : "fusion" | "single"
    """
    aim_h = cfg.get("aim_height_m", 1.0)
    aim_point = [subject_pos[0], subject_pos[1], subject_pos[2] + aim_h]
    cam_a_az = cfg["cam_a"]["azimuth_deg"]
    pos_a = camera_rig.camera_position(cam_a_az, r, h, subject_pos)
    pos_b = camera_rig.camera_position(cam_a_az + rel_az_deg, r, h, subject_pos)

    gt_avg, n_gt_frames = load_gt_average(gt_csv, joint_filter=set(joint_map.isaac_names()))
    pred_avg_raw, n_pred_rows = load_pred_average(pred_csv, conf_min=conf_min)

    grav_deg = _gravity_alignment_deg(pred_avg_raw)
    if not math.isnan(grav_deg):
        print(f"metrics: raw hips->neck axis vs +Y = {grav_deg:.1f} deg "
              f"(≈0 => gravity-aligned frame; ≈camera tilt => camera frame)")

    if mode == "fusion":
        pred_iso = {n: fused_to_isaac(p) for n, p in pred_avg_raw.items()}
    elif mode == "single":
        pred_iso = {n: single_cam_to_isaac(p, pos_a, aim_point)
                    for n, p in pred_avg_raw.items()}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    mpjpe, pck30, pck50, per_joint = mpjpe_pck(gt_avg, pred_iso)

    # Visibility / triangulability on the REAL averaged GT joints (replaces the
    # CANONICAL_SKELETON placeholder per the Phase 6 plan).
    gt_joints = list(gt_avg.values())
    vis = geo_prescreener.prescreen(pos_a, pos_b, gt_joints or None, cfg,
                                    subject=subject_pos)

    frames = meta.get("frames_grabbed") or 0
    coverage = (meta.get("frames_with_bodies", 0) / frames) if frames else NAN

    return {
        "h_m": h,
        "r_m": r,
        "rel_az_deg": rel_az_deg,
        "tilt_deg": camera_rig.tilt_angle(h, r, aim_h),
        "convergence_angle_deg": vis["convergence_angle_deg"],
        "subject_pos_name": subject_pos_name,
        "mpjpe_mm": mpjpe,
        "pck30": pck30,
        "pck50": pck50,
        "detection_coverage": coverage,
        "joint_visibility_cam_a": vis["joints_visible_cam_a"],
        "joint_visibility_cam_b": vis["joints_visible_cam_b"],
        "joint_visibility_either": vis["joints_visible_either"],
        "joint_visibility_both": vis["joints_visible_both_triangulable"],
        "unique_contribution_cam_b": vis["unique_contribution_cam_b"],
        "jitter_variance": NAN,   # N/A: character is static (animation follow-up)
        "id_drops": NAN,          # N/A: character is static (animation follow-up)
        # extras (not in results.csv schema, useful for debugging)
        "_n_gt_frames": n_gt_frames,
        "_n_pred_rows": n_pred_rows,
        "_per_joint_mm": per_joint,
        "_gravity_axis_deg": grav_deg,
    }


RESULTS_COLUMNS = [
    "h_m", "r_m", "rel_az_deg", "tilt_deg", "convergence_angle_deg",
    "subject_pos_name", "mpjpe_mm", "pck30", "pck50",
    "detection_coverage", "joint_visibility_cam_a", "joint_visibility_cam_b",
    "joint_visibility_either", "joint_visibility_both",
    "unique_contribution_cam_b", "jitter_variance", "id_drops",
]


def append_results_row(results_csv, metrics_dict):
    """Append one row to results.csv (header written if file is new).
    Never truncates — results/ is append-only per CLAUDE.md."""
    new = not os.path.exists(results_csv)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(RESULTS_COLUMNS)
        w.writerow([metrics_dict.get(c, NAN) for c in RESULTS_COLUMNS])
