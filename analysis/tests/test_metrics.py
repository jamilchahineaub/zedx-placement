# analysis/tests/test_metrics.py
#
# Synthetic-CSV tests for analysis/metrics.py. All fixtures are written to
# tmp_path; nothing under results/ is touched.

import csv
import math

import pytest

import joint_map
import metrics


CFG = {
    "aim_height_m": 1.0,
    "max_tilt_deg": 40.0,
    "cam_a": {"azimuth_deg": 0, "serial": 1001, "port": 30000},
    "cam_b": {"serial": 1002, "port": 30002},
    "zed_x": {"fov_h_deg": 110.0, "fov_v_deg": 70.0,
              "min_range_m": 0.3, "max_range_m": 10.0},
    "triangulable_min_deg": 40,
    "triangulable_max_deg": 140,
}

# A small standing skeleton in Isaac Z-up world (subject at origin).
GT_POS = {
    "NeckTwist01": (0.00, 0.00, 1.45),
    "R_Upperarm": (0.00, -0.20, 1.40),
    "L_Upperarm": (0.00, 0.20, 1.40),
    "R_Forearm": (0.00, -0.25, 1.10),
    "L_Forearm": (0.00, 0.25, 1.10),
    "R_Hand": (0.00, -0.27, 0.85),
    "L_Hand": (0.00, 0.27, 0.85),
    "R_Thigh": (0.00, -0.10, 0.90),
    "L_Thigh": (0.00, 0.10, 0.90),
    "R_Calf": (0.00, -0.10, 0.50),
    "L_Calf": (0.00, 0.10, 0.50),
    "R_Foot": (0.00, -0.10, 0.08),
    "L_Foot": (0.00, 0.10, 0.08),
    "R_Eye": (0.05, -0.03, 1.62),
    "L_Eye": (0.05, 0.03, 1.62),
}


def isaac_to_zed(p):
    """Forward permutation from make_fusion_config: zed = (-y, -z, x)."""
    return (-p[1], -p[2], p[0])


def write_gt(path, n_frames=3):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_time", "wall_clock", "joint_name", "x", "y", "z"])
        for fi in range(n_frames):
            t = fi / 30.0
            for name, (x, y, z) in GT_POS.items():
                w.writerow([t, 1000.0 + t, name, x, y, z])


def write_pred(path, offset=(0.0, 0.0, 0.0), conf=95.0, n_frames=3,
               frame="fusion"):
    """Write a zed_single/zed_pred-format CSV. Points are GT (optionally
    offset in Isaac space) converted into the requested frame."""
    inv = {i: z for z, i in joint_map.mapped_pairs()}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "wall_clock", "body_id", "tracking_state",
                    "joint_idx", "joint_name", "x", "y", "z", "confidence"])
        for fi in range(n_frames):
            for isaac_name, p in GT_POS.items():
                zed_name = inv[isaac_name]
                q = (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])
                if frame == "fusion":
                    q = isaac_to_zed(q)
                w.writerow([fi + 1, 2000.0 + fi, 0, "OK", 0, zed_name,
                            q[0], q[1], q[2], conf])


META = {"frames_grabbed": 20, "frames_with_bodies": 15}


def test_permutation_round_trip():
    p = (0.123, -4.5, 6.789)
    assert metrics.fused_to_isaac(isaac_to_zed(p)) == pytest.approx(p)


def test_perfect_prediction_fusion(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    write_gt(gt)
    write_pred(pred)
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    assert m["mpjpe_mm"] == pytest.approx(0.0, abs=1e-6)
    assert m["pck30"] == 1.0
    assert m["pck50"] == 1.0
    assert m["mpjpe_aligned_mm"] == pytest.approx(0.0, abs=1e-6)
    assert m["registration_offset_mm"] == pytest.approx(0.0, abs=1e-6)


def test_offset_40mm(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    write_gt(gt)
    write_pred(pred, offset=(0.04, 0.0, 0.0))   # 40 mm along Isaac x
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    assert m["mpjpe_mm"] == pytest.approx(40.0, abs=1e-6)
    assert m["pck30"] == 0.0
    assert m["pck50"] == 1.0
    # a CONSTANT 40 mm shift is pure registration offset -> aligned MPJPE ~0.
    assert m["mpjpe_aligned_mm"] == pytest.approx(0.0, abs=1e-6)
    assert m["registration_offset_mm"] == pytest.approx(40.0, abs=1e-6)


def test_single_cam_transform_aim_ray():
    """A point straight ahead of the camera at distance d (camera frame
    (0,0,-d), looking down -Z) must land on the camera->aim ray."""
    h, r = 1.5, 2.5
    subject = (0.0, 0.0, 0.0)
    aim = (0.0, 0.0, 1.0)
    import camera_rig
    cam = camera_rig.camera_position(0, r, h, subject)
    d = 1.0
    p = metrics.single_cam_to_isaac((0.0, 0.0, -d), cam, aim)
    # expected: cam + d * unit(aim - cam)
    v = [aim[i] - cam[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in v))
    expected = [cam[i] + d * v[i] / n for i in range(3)]
    assert p == pytest.approx(expected, abs=1e-9)


def test_low_confidence_excluded(tmp_path):
    pred = tmp_path / "pred.csv"
    write_pred(pred, conf=5.0)
    avg, used = metrics.load_pred_average(str(pred), conf_min=20.0)
    assert used == 0
    assert avg == {}


def test_detection_coverage(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    write_gt(gt)
    write_pred(pred)
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    assert m["detection_coverage"] == pytest.approx(15 / 20)


_CAM_C_COLS = ["cam_c_h_m", "joint_visibility_cam_c", "unique_contribution_cam_c",
               "convergence_ab_deg", "convergence_ac_deg", "convergence_bc_deg"]


def test_all_results_columns_present(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    write_gt(gt)
    write_pred(pred)
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    # 2-cam mode: every column EXCEPT the 3-cam-only (overhead) ones is present.
    for col in metrics.RESULTS_COLUMNS:
        if col in _CAM_C_COLS:
            assert col not in m, f"2-cam row should not carry {col}"
        else:
            assert col in m, f"missing column {col}"
    assert math.isnan(m["jitter_variance"])     # N/A on the static character
    assert math.isnan(m["id_drops"])


def test_overhead_adds_cam_c_columns(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    write_gt(gt)
    write_pred(pred)
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion", overhead_h=4.5)
    for col in metrics.RESULTS_COLUMNS:
        assert col in m, f"missing column {col}"     # 3-cam row carries them all
    assert m["cam_c_h_m"] == 4.5


def test_append_results_row(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    write_gt(gt)
    write_pred(pred)
    m = metrics.compute_metrics(str(gt), str(pred), META, h=1.5, r=2.5,
                                rel_az_deg=90, cfg=CFG, mode="fusion")
    out = tmp_path / "results.csv"
    metrics.append_results_row(str(out), m)
    metrics.append_results_row(str(out), m)     # append, never truncate
    with open(out) as f:
        lines = f.read().strip().splitlines()
    assert lines[0].split(",") == metrics.RESULTS_COLUMNS
    assert len(lines) == 3
