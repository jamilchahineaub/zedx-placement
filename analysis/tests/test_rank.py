# analysis/tests/test_rank.py  (v2)
#
# Tests for analysis/rank.py v2: three axes (pose_fidelity / absolute_placement /
# stability), validity FLAG (not gate), robust rank-normalization, Pareto over the
# accuracy axes, and the mounting-candidate output. Plain python3, synthetic data.

import csv
import math

import pytest

import rank


NAN = float("nan")

# Mirror of config/experiment.yaml -> ranking:
CFG = {
    "ranking": {
        "validity": {"max_registration_offset_mm": 300.0,
                     "max_jitter_variance": 50000.0,
                     "coverage_floor": 0.8},
        "pose_band_mm": [20.0, 200.0],
        "absolute_band_mm": [50.0, 1000.0],
        "axes": {
            "pose_fidelity": {"weight": 0.50,
                              "metrics": {"mpjpe_aligned_mm": {"w": 1.0, "norm": "pose_band"}}},
            "absolute_placement": {"weight": 0.20,
                                   "metrics": {"mpjpe_mm": {"w": 1.0, "norm": "absolute_band"}}},
            "stability": {"weight": 0.30,
                          "metrics": {"id_drops": {"w": 0.5, "norm": "rank"},
                                      "jitter_variance": {"w": 0.5, "norm": "rank"}}},
        },
        "presets": {"balanced": [0.50, 0.20, 0.30],
                    "pose": [0.70, 0.10, 0.20],
                    "absolute": [0.20, 0.60, 0.20],
                    "stability": [0.30, 0.10, 0.60]},
    }
}

RESULTS_HEADER = [
    "h_m", "r_m", "rel_az_deg", "tilt_deg", "convergence_angle_deg",
    "subject_pos_name", "mpjpe_mm", "mpjpe_aligned_mm", "registration_offset_mm",
    "pck30", "pck50", "detection_coverage",
    "joint_visibility_cam_a", "joint_visibility_cam_b", "joint_visibility_either",
    "joint_visibility_both", "unique_contribution_cam_b", "jitter_variance", "id_drops",
]


def L(h=1.5, r=1.5, az=90, aligned=70.0, absm=100.0, offset=100.0,
      idd=0.0, jit=2000.0, cov=1.0, conv=60.0, tilt=18.0, n=1):
    """Pre-aggregated layout dict (what aggregate_by_layout produces)."""
    return {
        "h_m": h, "r_m": r, "rel_az_deg": az, "n_subjects": n,
        "mpjpe_mm": absm, "mpjpe_worst": absm, "mpjpe_aligned_mm": aligned,
        "registration_offset_mm": offset, "coverage_min": cov, "detection_coverage": cov,
        "id_drops": idd, "jitter_variance": jit,
        "joint_visibility_both": 1.0, "convergence_angle_deg": conv, "tilt_deg": tilt,
    }


def _write_results(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RESULTS_HEADER)
        for r in rows:
            w.writerow(r)


# row order matches RESULTS_HEADER
ROW = lambda az, absm, aln, off, idd, jit: [
    1.5, 1.5, az, 18.43, 60.0, "center", absm, aln, off, 0, 0, 1.0, 1, 1, 1, 1.0, 0.0, jit, idd]


# --------------------------------------------------------------------------- load + aggregate

def test_load_and_aggregate(tmp_path):
    p = tmp_path / "results.csv"
    _write_results(p, [ROW(60, 100, 70, 90, 0, 2000), ROW(90, 130, 80, 110, 1, 3000)])
    rows = rank.load_results(str(p))
    assert len(rows) == 2 and rows[0]["mpjpe_aligned_mm"] == pytest.approx(70)
    lays = rank.aggregate_by_layout(rows)
    assert len(lays) == 2
    assert lays[0]["registration_offset_mm"] == pytest.approx(90)


def test_aggregate_means_across_subjects():
    rows = [
        {"h_m": 1.5, "r_m": 2.0, "rel_az_deg": 90, "subject_pos_name": "center",
         "mpjpe_mm": 100.0, "mpjpe_aligned_mm": 70.0, "registration_offset_mm": 80.0,
         "detection_coverage": 1.0, "jitter_variance": 2000.0, "id_drops": 0.0},
        {"h_m": 1.5, "r_m": 2.0, "rel_az_deg": 90, "subject_pos_name": "corner",
         "mpjpe_mm": 140.0, "mpjpe_aligned_mm": 90.0, "registration_offset_mm": 120.0,
         "detection_coverage": 0.6, "jitter_variance": 4000.0, "id_drops": 2.0},
    ]
    lays = rank.aggregate_by_layout(rows)
    a = lays[0]
    assert a["n_subjects"] == 2
    assert a["mpjpe_aligned_mm"] == pytest.approx(80.0)   # mean
    assert a["coverage_min"] == pytest.approx(0.6)        # worst-case


# --------------------------------------------------------------------------- normalization

def test_pose_and_absolute_bands():
    g = rank.metric_goodness([20.0, 110.0, 200.0, 10.0, 300.0], "pose_band", CFG)
    assert g[0] == pytest.approx(1.0) and g[1] == pytest.approx(0.5) and g[2] == pytest.approx(0.0)
    assert g[3] == pytest.approx(1.0) and g[4] == pytest.approx(0.0)   # clamped
    a = rank.metric_goodness([50.0, 525.0, 1000.0], "absolute_band", CFG)
    assert a[0] == pytest.approx(1.0) and a[1] == pytest.approx(0.5) and a[2] == pytest.approx(0.0)


def test_rank_goodness_outlier_robust():
    raw = [2000.0, 3000.0, 5000.0, 10_000_000.0]   # lower is better; one huge outlier
    g = rank._rank_goodness(raw, lower_better=True)
    assert g[0] == pytest.approx(1.0) and g[3] == pytest.approx(0.0)
    assert g[0] > g[1] > g[2] > g[3]
    # the outlier's MAGNITUDE must not change the others' goodness (the whole point)
    g2 = rank._rank_goodness([2000.0, 3000.0, 5000.0, 5.0e12], lower_better=True)
    assert g2[:3] == pytest.approx(g[:3])


def test_nan_metric_drops_from_axis():
    lays = [L(jit=NAN, idd=NAN), L(jit=NAN, idd=NAN)]   # stability metrics all NaN
    per, names, notes = rank.axis_scores(lays, CFG)
    assert per[0]["stability"] is None        # axis drops out
    assert "dropped" in notes["jitter_variance"]


# --------------------------------------------------------------------------- validity flag (not gate)

def test_validity_flags_but_keeps():
    lays = [L(offset=100, jit=2000, cov=1.0), L(offset=500, jit=2000, cov=1.0),
            L(offset=100, jit=99999, cov=1.0), L(offset=100, jit=2000, cov=0.5)]
    v = rank.validity(lays, CFG)
    assert v[0] == (True, "ok")
    assert v[1][0] is False and "offset" in v[1][1]
    assert v[2][0] is False and "jitter" in v[2][1]
    assert v[3][0] is False and "coverage" in v[3][1]
    # nothing is excluded: rank() still returns ALL layouts
    ranked, _, _ = rank.rank(lays, CFG, rank.default_axis_weights(CFG))
    assert len(ranked) == 4


# --------------------------------------------------------------------------- axes + pareto

def test_pareto_pose_vs_absolute_both_on_frontier():
    A = L(az=60, aligned=66, absm=900, offset=900)    # best pose, worst absolute
    B = L(az=90, aligned=120, absm=80, offset=80)     # worse pose, best absolute
    C = L(az=120, aligned=130, absm=950, offset=950, idd=5, jit=8000)  # dominated by both
    per, names, _ = rank.axis_scores([A, B, C], CFG)
    flags, used = rank.pareto_front(per, names)
    assert set(used) == {"pose_fidelity", "absolute_placement", "stability"}
    assert flags == [True, True, False]   # the pose<->absolute tradeoff keeps A and B both


def test_rank_orders_by_composite_and_flags():
    good = L(az=60, aligned=66, absm=80, offset=80, idd=0, jit=2000)     # valid, best
    failed = L(az=90, aligned=130, absm=2000, offset=2000, idd=10, jit=80000)  # invalid
    ranked, _, _ = rank.rank([failed, good], CFG, rank.default_axis_weights(CFG))
    assert rank.layout_label(ranked[0]) == "h1.5 r1.5 az60"
    assert ranked[0]["valid"] is True
    assert ranked[1]["valid"] is False and "offset" in ranked[1]["flag_reason"]


# --------------------------------------------------------------------------- mounting + presets

def test_mounting_candidates_are_valid():
    A = L(az=60, aligned=66, absm=80, offset=80)       # valid
    B = L(az=90, aligned=64, absm=900, offset=900)     # invalid (offset>300), best pose
    ranked, _, _ = rank.rank([A, B], CFG, rank.default_axis_weights(CFG))
    cands = rank.mounting_candidates(ranked, 3)
    assert cands and all(c["valid"] for c in cands)     # invalid B never a mounting candidate
    assert any("az60" in rank.layout_label(c) for c in cands)


def test_preset_weights():
    assert rank.preset_weights(CFG, "pose") == {
        "pose_fidelity": 0.70, "absolute_placement": 0.10, "stability": 0.20}
    cw = rank.parse_weights("pose_fidelity=0.6,stability=0.3", CFG)
    assert cw["pose_fidelity"] == 0.6 and cw["absolute_placement"] == 0.20


# --------------------------------------------------------------------------- render smoke

def test_render_smoke(tmp_path):
    p = tmp_path / "results.csv"
    _write_results(p, [ROW(60, 100, 70, 90, 0, 2000),
                       ROW(90, 2000, 130, 2000, 10, 80000),   # a failure
                       ROW(120, 130, 80, 110, 1, 3000)])
    rows = rank.load_results(str(p))
    lays = rank.aggregate_by_layout(rows)
    cw = rank.default_axis_weights(CFG)
    rank._active_weights = cw
    ranked, notes, pareto_axes = rank.rank(lays, CFG, cw)

    class A:
        top = 15
        subject = None
    text = rank.render(ranked, notes, pareto_axes, CFG, A())
    assert "Physical mounting candidates" in text
    assert "Insight:" in text
    assert "mount both cameras" in text
    assert "valid" in text
