# analysis/tests/test_floor_coverage.py
#
# Tests for analysis/floor_coverage.py. Synthetic walk: the pelvis slides along +x;
# detection is LOST once x > 1.0. The right-side cells must show detection_rate < 1,
# the centre cell ~1.0 with an MPJPE. Plain python3; no matplotlib needed.

import csv
import math

import pytest

import floor_coverage as fc
import joint_map


CFG = {"workspace": {"center": [0.0, 0.0], "size_m": [5.0, 5.0]}, "grid_cell_m": 1.0}

INV = {i: z for z, i in joint_map.mapped_pairs()}   # isaac -> zed
NECK_I, NECK_Z = "NeckTwist01", INV["NeckTwist01"]
RUA_I, RUA_Z = "R_Upperarm", INV["R_Upperarm"]
W0, DT = 1000.0, 1.0 / 30.0
N = 30


def isaac_to_zed(p):
    return (-p[1], -p[2], p[0])


def _x(f):
    return -2.4 + 4.8 * f / (N - 1)        # slides -2.4 -> +2.4


def _detected(f):
    return _x(f) <= 1.0                      # lost once past x=1


def _write(tmp_path):
    lid = "t"
    gt = tmp_path / f"ground_truth_{lid}.csv"
    pred = tmp_path / f"zed_pred_{lid}.csv"
    hb = tmp_path / f"zed_pred_{lid}_frames.csv"
    with open(gt, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sim_time", "wall_clock", "joint_name", "x", "y", "z"])
        for i in range(N):
            wall = W0 + i * DT; x = _x(i)
            w.writerow([i * DT, wall, "Pelvis", x, 0.0, 0.9])
            w.writerow([i * DT, wall, NECK_I, x, 0.0, 1.45])
            w.writerow([i * DT, wall, RUA_I, x, -0.2, 1.40])
    with open(pred, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "wall_clock", "body_id", "tracking_state",
                    "joint_idx", "joint_name", "x", "y", "z", "confidence"])
        for i in range(N):
            if not _detected(i):
                continue
            wall = W0 + i * DT; x = _x(i)
            for j, (iname, p) in enumerate([(NECK_Z, (x, 0.0, 1.45)), (RUA_Z, (x, -0.2, 1.40))]):
                q = isaac_to_zed(p)
                w.writerow([i + 1, wall, 1, "OBJECT_TRACKING_STATE.OK", j, iname,
                            q[0], q[1], q[2], 95.0])
    with open(hb, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_idx", "wall_clock", "n_bodies"])
        for i in range(N):
            w.writerow([i + 1, W0 + i * DT, 1 if _detected(i) else 0])
    return lid


def test_grid_and_cell():
    g = fc.grid_dims(CFG)
    assert (g["nx"], g["ny"], g["cell"]) == (5, 5, 1.0)
    assert fc.cell_of(0.0, 0.0, g) == (2, 2)        # centre
    assert fc.cell_of(-2.4, 0.0, g) == (0, 2)
    assert fc.cell_of(10.0, 0.0, g) is None         # outside box


def test_pelvis_loader(tmp_path):
    lid = _write(tmp_path)
    pelvis, name = fc.load_gt_pelvis_per_frame(str(tmp_path / f"ground_truth_{lid}.csv"))
    assert name == "Pelvis"
    assert len(pelvis) == N
    assert pelvis[0][1] == pytest.approx(_x(0))     # x increases over frames
    assert pelvis[-1][1] > pelvis[0][1]


def test_coverage_lost_on_right(tmp_path):
    lid = _write(tmp_path)
    cells, g, info = fc.compute_floor_coverage(lid, CFG, str(tmp_path), conf_min=20.0)
    assert info["pelvis_joint"] == "Pelvis" and info["heartbeat"] is True

    # centre cell (x~0, ix=2): always detected + has an MPJPE (~0 for perfect pred)
    centre = cells[(2, 2)]
    assert centre["n"] > 0 and centre["det"] == centre["n"]      # detection_rate == 1
    assert centre["mpjpe"] and sum(centre["mpjpe"]) / len(centre["mpjpe"]) < 1.0

    # right cell (x in [1.5,2.5) -> ix=4): person is there but ALWAYS lost
    right = cells[(4, 2)]
    assert right["n"] > 0 and right["det"] == 0                  # detection_rate == 0

    # overall coverage strictly between 0 and 1 (some regions covered, some lost)
    tot_n = sum(d["n"] for d in cells.values())
    tot_det = sum(d["det"] for d in cells.values())
    assert 0.0 < tot_det / tot_n < 1.0


def test_outputs_smoke(tmp_path):
    lid = _write(tmp_path)
    cells, g, _ = fc.compute_floor_coverage(lid, CFG, str(tmp_path), conf_min=20.0)
    txt = fc.ascii_map(cells, g)
    assert "X" in txt and "#" in txt              # lost cells and covered cells both shown
    out = tmp_path / "floor_t.csv"
    fc.write_cell_csv(cells, g, str(out))
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == g["nx"] * g["ny"]
    assert {"detection_rate", "mpjpe_mean_mm", "x_center"} <= set(rows[0])
