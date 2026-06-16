# analysis/tests/test_rank.py
#
# Tests for analysis/rank.py. Plain python3, no fixtures under results/.
# Synthetic CSV written to tmp_path where a file is needed.

import csv
import math

import pytest

import rank


NAN = float("nan")

# Mirror of config/experiment.yaml -> ranking:
CFG = {
    "ranking": {
        "coverage_floor": 0.8,
        "mpjpe_good_mm": 20.0,
        "mpjpe_bad_mm": 200.0,
        "categories": {
            "accuracy": {"weight": 0.50,
                         "metrics": {"mpjpe_aligned_mm": 0.80, "pck50": 0.20}},
            "reliability": {"weight": 0.30,
                            "metrics": {"detection_coverage": 0.50,
                                        "jitter_variance": 0.35,
                                        "id_drops": 0.15}},
            "geometry": {"weight": 0.20,
                         "metrics": {"joint_visibility_both": 0.70,
                                     "unique_contribution_cam_b": 0.30}},
        },
        "presets": {"balanced": [0.50, 0.30, 0.20],
                    "accuracy": [0.70, 0.20, 0.10],
                    "robustness": [0.34, 0.33, 0.33]},
    }
}

RESULTS_HEADER = [
    "h_m", "r_m", "rel_az_deg", "tilt_deg", "convergence_angle_deg",
    "subject_pos_name", "mpjpe_mm", "mpjpe_aligned_mm", "registration_offset_mm",
    "pck30", "pck50", "detection_coverage",
    "joint_visibility_cam_a", "joint_visibility_cam_b", "joint_visibility_either",
    "joint_visibility_both", "unique_contribution_cam_b", "jitter_variance", "id_drops",
]


def L(h=1.5, r=1.5, az=90, mpjpe=110.0, mpjpe_aligned=None, pck50=0.0, cov=1.0,
      visb=1.0, uniq=0.0, jit=NAN, idd=NAN, conv=60.0, tilt=18.0, n=1):
    """A pre-aggregated layout dict (what aggregate_by_layout produces)."""
    return {
        "h_m": h, "r_m": r, "rel_az_deg": az, "n_subjects": n,
        "mpjpe_mm": mpjpe, "mpjpe_worst": mpjpe, "coverage_min": cov,
        "mpjpe_aligned_mm": mpjpe if mpjpe_aligned is None else mpjpe_aligned,
        "registration_offset_mm": NAN,
        "pck50": pck50, "pck30": 0.0, "detection_coverage": cov,
        "joint_visibility_both": visb, "unique_contribution_cam_b": uniq,
        "jitter_variance": jit, "id_drops": idd,
        "joint_visibility_cam_a": 1.0, "joint_visibility_cam_b": 1.0,
        "joint_visibility_either": 1.0,
        "convergence_angle_deg": conv, "tilt_deg": tilt,
    }


def _write_results(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RESULTS_HEADER)
        for r in rows:
            w.writerow(r)


REAL_ROWS = [  # cols: ..., mpjpe_mm, mpjpe_aligned_mm, registration_offset_mm, pck30, pck50, cov, ...
    [1.5, 1.5, 60, 18.43, 41.41, "center", 109.98, 42.9, 93, 0, 0, 1.0, 1, 1, 1, 1.0, 0.0, "nan", "nan"],
    [1.5, 1.5, 90, 18.43, 60.00, "center", 122.33, 47.9, 102, 0, 0, 1.0, 1, 1, 1, 1.0, 0.0, "nan", "nan"],
    [1.5, 1.5, 120, 18.43, 75.52, "center", 126.99, 69.6, 109, 0, 0, 1.0, 1, 1, 1, 0.8667, 0.0, "nan", "nan"],
]


# --------------------------------------------------------------------------- load + aggregate

def test_load_results_parses_nan_and_strings(tmp_path):
    p = tmp_path / "results.csv"
    _write_results(p, REAL_ROWS)
    rows = rank.load_results(str(p))
    assert len(rows) == 3
    assert rows[0]["subject_pos_name"] == "center"
    assert rows[0]["mpjpe_mm"] == pytest.approx(109.98)
    assert math.isnan(rows[0]["jitter_variance"])


def test_aggregate_means_across_subjects():
    rows = [
        {"h_m": 1.5, "r_m": 2.0, "rel_az_deg": 90, "subject_pos_name": "center",
         "mpjpe_mm": 100.0, "detection_coverage": 1.0, "pck50": 0.2,
         "joint_visibility_both": 1.0, "unique_contribution_cam_b": 0.0,
         "jitter_variance": NAN, "id_drops": NAN,
         "convergence_angle_deg": 60.0, "tilt_deg": 12.0},
        {"h_m": 1.5, "r_m": 2.0, "rel_az_deg": 90, "subject_pos_name": "corner",
         "mpjpe_mm": 140.0, "detection_coverage": 0.6, "pck50": 0.0,
         "joint_visibility_both": 0.8, "unique_contribution_cam_b": 0.1,
         "jitter_variance": NAN, "id_drops": NAN,
         "convergence_angle_deg": 60.0, "tilt_deg": 12.0},
    ]
    lays = rank.aggregate_by_layout(rows)
    assert len(lays) == 1
    a = lays[0]
    assert a["n_subjects"] == 2
    assert a["mpjpe_mm"] == pytest.approx(120.0)      # mean
    assert a["mpjpe_worst"] == pytest.approx(140.0)   # max
    assert a["coverage_min"] == pytest.approx(0.6)    # worst-case for the gate
    assert a["detection_coverage"] == pytest.approx(0.8)  # mean for the score


def test_aggregate_subject_filter():
    rows = [
        {"h_m": 1.5, "r_m": 2.0, "rel_az_deg": 90, "subject_pos_name": "center",
         "mpjpe_mm": 100.0, "detection_coverage": 1.0},
        {"h_m": 1.5, "r_m": 2.0, "rel_az_deg": 90, "subject_pos_name": "corner",
         "mpjpe_mm": 140.0, "detection_coverage": 1.0},
    ]
    lays = rank.aggregate_by_layout(rows, subject="center")
    assert len(lays) == 1 and lays[0]["n_subjects"] == 1
    assert lays[0]["mpjpe_mm"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- goodness

def test_mpjpe_band_goodness():
    lays = [L(mpjpe=20.0), L(mpjpe=110.0), L(mpjpe=200.0), L(mpjpe=10.0), L(mpjpe=250.0)]
    goods, _ = rank.goodness_columns(lays, CFG)
    g = goods["mpjpe_aligned_mm"]   # accuracy now ranks on the aligned MPJPE (band-scored)
    assert g[0] == pytest.approx(1.0)     # at the good ref
    assert g[1] == pytest.approx(0.5)     # midpoint of the band
    assert g[2] == pytest.approx(0.0)     # at the bad ref
    assert g[3] == pytest.approx(1.0)     # better than ref -> clamped
    assert g[4] == pytest.approx(0.0)     # worse than ref -> clamped


def test_absolute_metric_used_as_is_not_minmax():
    # coverage 0.9 and 1.0 -> goodness stays 0.9 and 1.0 (NOT 0 and 1).
    lays = [L(cov=0.9), L(cov=1.0)]
    goods, _ = rank.goodness_columns(lays, CFG)
    assert goods["detection_coverage"][0] == pytest.approx(0.9)
    assert goods["detection_coverage"][1] == pytest.approx(1.0)


def test_all_nan_metric_drops_out():
    lays = [L(jit=NAN, idd=NAN), L(jit=NAN, idd=NAN)]
    goods, notes = rank.goodness_columns(lays, CFG)
    assert goods["jitter_variance"] == [None, None]
    assert "dropped" in notes["jitter_variance"]
    # reliability then comes purely from coverage.
    cats, _, _ = rank.category_scores(lays, CFG)
    assert cats[0]["reliability"] == pytest.approx(1.0)  # cov=1.0 only


def test_relative_lower_metric_normalized_within_sweep():
    lays = [L(jit=2.0), L(jit=10.0)]  # lower is better
    goods, _ = rank.goodness_columns(lays, CFG)
    assert goods["jitter_variance"][0] == pytest.approx(1.0)  # smallest -> best
    assert goods["jitter_variance"][1] == pytest.approx(0.0)  # largest -> worst


# --------------------------------------------------------------------------- categories + composite

def test_category_scores_known_value():
    lay = L(mpjpe=110.0, pck50=0.0, cov=1.0, visb=1.0, uniq=0.0)
    cats, _, _ = rank.category_scores([lay], CFG)
    c = cats[0]
    # accuracy = 0.8*0.5 + 0.2*0.0 (renorm over present) = 0.4
    assert c["accuracy"] == pytest.approx(0.40)
    assert c["reliability"] == pytest.approx(1.0)   # jitter/id NaN -> coverage only
    assert c["geometry"] == pytest.approx(0.70)     # 0.7*1.0 + 0.3*0.0


def test_composite_renormalizes_over_present_categories():
    # geometry absent (both geo metrics NaN) -> composite over acc+rel only.
    lay = L(mpjpe=110.0, visb=NAN, uniq=NAN)
    cats, _, _ = rank.category_scores([lay], CFG)
    assert cats[0]["geometry"] is None
    scores = rank.composite(cats, {"accuracy": 0.5, "reliability": 0.3, "geometry": 0.2})
    expected = (0.5 * cats[0]["accuracy"] + 0.3 * cats[0]["reliability"]) / 0.8
    assert scores[0] == pytest.approx(expected)


# --------------------------------------------------------------------------- gate

def test_gate_drops_low_coverage_and_nan_mpjpe():
    lays = [L(az=60, cov=1.0), L(az=90, cov=0.3), L(az=120, mpjpe=NAN)]
    kept, dropped = rank.gate(lays, 0.8)
    assert [k["rel_az_deg"] for k in kept] == [60]
    reasons = {d[0]["rel_az_deg"]: d[1] for d in dropped}
    assert "coverage" in reasons[90]
    assert "NaN" in reasons[120]


# --------------------------------------------------------------------------- pareto + ranking

def test_pareto_sole_winner_on_real_data():
    lays = [L(az=60, mpjpe=109.98, visb=1.0),
            L(az=90, mpjpe=122.33, visb=1.0),
            L(az=120, mpjpe=126.99, visb=0.8667)]
    cats, names, _ = rank.category_scores(lays, CFG)
    flags, used = rank.pareto_front(cats, names)
    assert flags == [True, False, False]              # az60 dominates the others
    assert set(used) == {"accuracy", "reliability", "geometry"}


def test_rank_orders_by_composite():
    lays = [L(az=120, mpjpe=126.99, visb=0.8667),
            L(az=60, mpjpe=109.98, visb=1.0),
            L(az=90, mpjpe=122.33, visb=1.0)]
    ranked, _, _ = rank.rank(lays, CFG, rank.default_category_weights(CFG))
    assert [r["rel_az_deg"] for r in ranked] == [60, 90, 120]
    assert ranked[0]["rank"] == 1 and ranked[0]["pareto"] is True
    assert ranked[0]["score"] > ranked[1]["score"] > ranked[2]["score"]


def test_az60_stable_across_presets():
    lays = [L(az=60, mpjpe=109.98, visb=1.0),
            L(az=90, mpjpe=122.33, visb=1.0),
            L(az=120, mpjpe=126.99, visb=0.8667)]
    for preset in ("balanced", "accuracy", "robustness"):
        ranked, _, _ = rank.rank(lays, CFG, rank.preset_weights(CFG, preset))
        assert ranked[0]["rel_az_deg"] == 60


# --------------------------------------------------------------------------- weights parsing

def test_preset_and_parse_weights():
    assert rank.preset_weights(CFG, "accuracy") == {
        "accuracy": 0.70, "reliability": 0.20, "geometry": 0.10}
    cw = rank.parse_weights("accuracy=0.6,geometry=0.1", CFG)
    assert cw["accuracy"] == 0.6 and cw["geometry"] == 0.1
    assert cw["reliability"] == 0.30   # untouched default


# --------------------------------------------------------------------------- end-to-end render smoke

def test_render_smoke(tmp_path):
    p = tmp_path / "results.csv"
    _write_results(p, REAL_ROWS)
    rows = rank.load_results(str(p))
    lays = rank.aggregate_by_layout(rows)
    kept, dropped = rank.gate(lays, 0.8)
    cw = rank.default_category_weights(CFG)
    rank._active_weights = cw
    ranked, notes, pareto_cats = rank.rank(kept, CFG, cw)

    class A:  # stand-in for argparse Namespace
        top = 10
        subject = None
    text = rank.render(ranked, notes, pareto_cats, kept, dropped, CFG, A(), 0.8)
    assert "h1.5 r1.5 az60" in text
    assert "SOLE Pareto winner" in text
    assert "STABLE" in text
