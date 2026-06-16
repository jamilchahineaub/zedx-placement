# analysis/tests/test_motion_metrics.py
#
# Per-frame motion metrics in analysis/metrics.py (the path taken when the
# character is animated): per-frame loaders, frame association, per-frame MPJPE,
# jitter_variance, id_drops, and the static fallback. Plain python3; synthetic
# CSVs in tmp_path. Nothing under results/ is touched.

import csv
import math

import pytest

import joint_map
import metrics


CFG = {
    "aim_height_m": 1.0, "max_tilt_deg": 40.0,
    "cam_a": {"azimuth_deg": 0, "serial": 1001, "port": 30000},
    "cam_b": {"serial": 1002, "port": 30002},
    "zed_x": {"fov_h_deg": 110.0, "fov_v_deg": 70.0,
              "min_range_m": 0.3, "max_range_m": 10.0},
    "triangulable_min_deg": 40, "triangulable_max_deg": 140,
}
META = {"frames_grabbed": 6, "frames_with_bodies": 6}

# 15 mapped Isaac joints (a small standing skeleton, Isaac Z-up).
BASE = {
    "NeckTwist01": (0.00, 0.00, 1.45),
    "R_Upperarm": (0.00, -0.20, 1.40), "L_Upperarm": (0.00, 0.20, 1.40),
    "R_Forearm": (0.00, -0.25, 1.10), "L_Forearm": (0.00, 0.25, 1.10),
    "R_Hand": (0.00, -0.27, 0.85), "L_Hand": (0.00, 0.27, 0.85),
    "R_Thigh": (0.00, -0.10, 0.90), "L_Thigh": (0.00, 0.10, 0.90),
    "R_Calf": (0.00, -0.10, 0.50), "L_Calf": (0.00, 0.10, 0.50),
    "R_Foot": (0.00, -0.10, 0.08), "L_Foot": (0.00, 0.10, 0.08),
    "R_Eye": (0.05, -0.03, 1.62), "L_Eye": (0.05, 0.03, 1.62),
}
INV = {i: z for z, i in joint_map.mapped_pairs()}      # isaac -> zed name
WALL0 = 5000.0
DT = 1.0 / 30.0


def isaac_to_zed(p):
    """Fusion forward permutation (make_fusion_config): zed = (-y, -z, x)."""
    return (-p[1], -p[2], p[0])


def moving_frames(n=6, amp=0.12):
    """n frames; every joint slides in x by amp*sin -> > 2 cm range (motion)."""
    out = []
    for i in range(n):
        dx = amp * math.sin(2 * math.pi * i / n)
        out.append({name: (x + dx, y, z) for name, (x, y, z) in BASE.items()})
    return out


def static_frames(n=4):
    return [dict(BASE) for _ in range(n)]


def write_gt(path, frames):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_time", "wall_clock", "joint_name", "x", "y", "z"])
        for i, fr in enumerate(frames):
            wall = WALL0 + i * DT
            for name, (x, y, z) in fr.items():
                w.writerow([i * DT, wall, name, x, y, z])


def write_pred(path, frames, frame="fusion", conf=95.0, offset_fn=None,
               body_id=0, states=None):
    """Aligned wall clock with GT (WALL0 + i*DT) so frame i pairs with GT i.
    offset_fn(i) -> (dx,dy,dz) in Isaac space added before converting to ZED frame."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "wall_clock", "body_id", "tracking_state",
                    "joint_idx", "joint_name", "x", "y", "z", "confidence"])
        for i, fr in enumerate(frames):
            wall = WALL0 + i * DT
            bid = body_id(i) if callable(body_id) else body_id
            st = states[i] if states else "OBJECT_TRACKING_STATE.OK"
            off = offset_fn(i) if offset_fn else (0.0, 0.0, 0.0)
            for j, (isaac_name, p) in enumerate(fr.items()):
                q = (p[0] + off[0], p[1] + off[1], p[2] + off[2])
                q = isaac_to_zed(q) if frame == "fusion" else q
                w.writerow([i + 1, wall, bid, st, j, INV[isaac_name],
                            q[0], q[1], q[2], conf])


# --------------------------------------------------------------------------- loaders

def test_load_gt_per_frame_groups_by_wall(tmp_path):
    gt = tmp_path / "gt.csv"
    write_gt(gt, moving_frames(n=6))
    frames = metrics.load_gt_per_frame(str(gt), joint_filter=set(joint_map.isaac_names()))
    assert len(frames) == 6
    assert len(frames[0]["joints"]) == 15
    walls = [f["wall"] for f in frames]
    assert walls == sorted(walls)


def test_load_pred_per_frame_conf_filter(tmp_path):
    pred = tmp_path / "pred.csv"
    write_pred(pred, moving_frames(n=5), conf=5.0)
    frames = metrics.load_pred_per_frame(str(pred), conf_min=20.0)
    # all joints filtered out by confidence -> frames have empty bodies / dropped
    assert all(not any(b["joints"] for b in f["bodies"].values()) for f in frames)
    frames_ok = metrics.load_pred_per_frame(str(pred), conf_min=0.0)
    assert len(frames_ok) == 5
    assert 0 in frames_ok[0]["bodies"]
    assert len(frames_ok[0]["bodies"][0]["joints"]) == 15


# --------------------------------------------------------------------------- motion detect + assoc

def test_has_motion_true_and_false(tmp_path):
    names = joint_map.isaac_names()
    g_move = [{"wall": i * DT, "sim_time": i * DT, "joints": fr}
              for i, fr in enumerate(moving_frames(n=6))]
    g_static = [{"wall": i * DT, "sim_time": i * DT, "joints": fr}
                for i, fr in enumerate(static_frames(n=6))]
    assert metrics._has_motion(g_move, names) is True
    assert metrics._has_motion(g_static, names) is False


def test_associate_frames_aligned():
    gt = [{"wall": WALL0 + i * DT, "sim_time": i * DT, "joints": {}} for i in range(5)]
    pred = [{"frame_idx": i + 1, "wall": WALL0 + i * DT, "bodies": {}} for i in range(5)]
    pairs = metrics.associate_frames(gt, pred)
    assert len(pairs) == 5
    for i, (g, p) in enumerate(pairs):
        assert g["wall"] == pytest.approx(p["wall"])  # frame i <-> i


def test_primary_body_picks_most_joints():
    pf = {"frame_idx": 1, "wall": 0.0, "bodies": {
        1: {"tracking_state": "OK", "joints": {"NECK": (0, 0, 0)}},
        2: {"tracking_state": "OK", "joints": {z: (0, 0, 0) for z, _ in joint_map.mapped_pairs()}},
    }}
    assert metrics._primary_body(pf) is pf["bodies"][2]


# --------------------------------------------------------------------------- mpjpe / jitter per frame

def test_mpjpe_per_frame_perfect(tmp_path):
    gt = tmp_path / "gt.csv"; pred = tmp_path / "pred.csv"
    frames = moving_frames(n=6)
    write_gt(gt, frames)
    write_pred(pred, frames)                     # exact match
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    assert m["_motion"] is True
    assert m["mpjpe_mm"] == pytest.approx(0.0, abs=1e-6)
    assert m["pck30"] == 1.0
    assert m["jitter_variance"] == pytest.approx(0.0, abs=1e-6)   # zero, NOT NaN
    assert m["id_drops"] == 0.0


def test_mpjpe_per_frame_constant_offset(tmp_path):
    gt = tmp_path / "gt.csv"; pred = tmp_path / "pred.csv"
    frames = moving_frames(n=6)
    write_gt(gt, frames)
    write_pred(pred, frames, offset_fn=lambda i: (0.04, 0.0, 0.0))  # constant 40 mm
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    assert m["mpjpe_mm"] == pytest.approx(40.0, abs=1e-6)
    assert m["jitter_variance"] == pytest.approx(0.0, abs=1e-6)   # constant error -> no jitter


def test_jitter_positive_when_error_varies(tmp_path):
    gt = tmp_path / "gt.csv"; pred = tmp_path / "pred.csv"
    frames = moving_frames(n=6)
    write_gt(gt, frames)
    # error grows each frame -> error magnitude varies -> positive variance
    write_pred(pred, frames, offset_fn=lambda i: (0.002 * i, 0.0, 0.0))
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    assert m["jitter_variance"] > 0.0


# --------------------------------------------------------------------------- id_drops

def test_id_drops_clean_single_id(tmp_path):
    pred = tmp_path / "pred.csv"
    write_pred(pred, moving_frames(n=6), body_id=7)
    frames = metrics.load_pred_per_frame(str(pred))
    assert metrics.compute_id_drops(frames) == 0.0


def test_id_drops_identity_switch(tmp_path):
    pred = tmp_path / "pred.csv"
    write_pred(pred, moving_frames(n=6), body_id=lambda i: 1 if i < 3 else 2)
    frames = metrics.load_pred_per_frame(str(pred))
    assert metrics.compute_id_drops(frames) == 1.0   # two distinct OK ids -> 1 switch


def test_id_drops_tracking_gap(tmp_path):
    pred = tmp_path / "pred.csv"
    states = ["OBJECT_TRACKING_STATE.OK"] * 6
    states[3] = "OBJECT_TRACKING_STATE.OFF"          # one dropped frame
    write_pred(pred, moving_frames(n=6), body_id=4, states=states)
    frames = metrics.load_pred_per_frame(str(pred))
    assert metrics.compute_id_drops(frames) == 1.0   # one tracked->untracked transition


# --------------------------------------------------------------------------- static fallback

def test_static_character_uses_fallback(tmp_path):
    gt = tmp_path / "gt.csv"; pred = tmp_path / "pred.csv"
    frames = static_frames(n=4)
    write_gt(gt, frames)
    write_pred(pred, frames)
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    assert m["_motion"] is False
    assert math.isnan(m["jitter_variance"])
    assert math.isnan(m["id_drops"])
    assert m["mpjpe_mm"] == pytest.approx(0.0, abs=1e-6)
