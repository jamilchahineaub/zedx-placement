# analysis/tests/test_prescreener.py
#
# Unit tests for analysis/geo_prescreener.py (plain python3).
# Run from repo root:  python3 -m pytest analysis/tests/test_prescreener.py -v

import pytest

import camera_rig as cr
import geo_prescreener as gp


# cfg mirroring the relevant fields of config/experiment.yaml
CFG = {
    "zed_x": {
        "fov_h_deg": 110.0,
        "fov_v_deg": 70.0,
        "min_range_m": 0.3,
        "max_range_m": 10.0,
    },
    "triangulable_min_deg": 40,
    "triangulable_max_deg": 140,
    "aim_height_m": 1.0,
}


def _cams(rel_az_deg, r, h, subject=(0, 0, 0)):
    """Both cameras on the ring: A at az 0, B at rel_az."""
    a = cr.camera_position(0, r, h, subject)
    b = cr.camera_position(rel_az_deg, r, h, subject)
    return a, b


# ---------------------------------------------------------------------------
# Co-located cameras (≈0 deg separation) -> B contributes nothing unique
# ---------------------------------------------------------------------------

def test_colocated_no_unique_contribution():
    a, b = _cams(1, r=4, h=2.5)   # ~1 deg apart: effectively co-located
    out = gp.prescreen(a, b, None, CFG)
    assert out["unique_contribution_cam_b"] == pytest.approx(0.0, abs=1e-9)
    # rays nearly parallel -> convergence well below triangulable_min -> none triangulable
    assert out["joints_visible_both_triangulable"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 90 deg separation, r=4, h=2.5 (valid tilt) -> strong triangulable coverage
# ---------------------------------------------------------------------------

def test_90deg_good_coverage():
    a, b = _cams(90, r=4, h=2.5)
    out = gp.prescreen(a, b, None, CFG)
    assert out["joints_visible_both_triangulable"] > 0.7
    assert out["passed"] is True


# ---------------------------------------------------------------------------
# Beyond max range -> nothing visible to either camera
# ---------------------------------------------------------------------------

def test_beyond_max_range_nothing_visible():
    a, b = _cams(90, r=12, h=2.5)   # 12 m > max_range_m (10)
    out = gp.prescreen(a, b, None, CFG)
    assert out["joints_visible_either"] == pytest.approx(0.0, abs=1e-9)
    assert out["passed"] is False


# ---------------------------------------------------------------------------
# Sanity on the canonical skeleton constant
# ---------------------------------------------------------------------------

def test_canonical_skeleton_shape_and_height():
    js = gp.canonical_skeleton()
    assert len(js) == 14
    zs = [j[2] for j in js]
    # head near 1.6 m, lowest joint (knee) well below; total span < 1.75 m
    assert max(zs) == pytest.approx(1.62, abs=1e-6)
    assert min(zs) >= 0.0


def test_canonical_skeleton_translates_with_subject():
    base = gp.canonical_skeleton((0, 0, 0))
    moved = gp.canonical_skeleton((1.5, -2.0, 0))
    assert moved[0][0] == pytest.approx(base[0][0] + 1.5)
    assert moved[0][1] == pytest.approx(base[0][1] - 2.0)
