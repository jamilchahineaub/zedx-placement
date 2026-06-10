# analysis/tests/test_camera_rig.py
#
# Unit tests for isaac/camera_rig.py (pure math, plain python3).
# Run from repo root:  python3 -m pytest analysis/tests/test_camera_rig.py -v

import math
import pytest

import camera_rig as cr


# --- minimal cfg matching config/experiment.yaml fields used here ---
CFG = {"max_tilt_deg": 40.0, "aim_height_m": 1.0}


# ---------------------------------------------------------------------------
# camera_position
# ---------------------------------------------------------------------------

def test_position_az0():
    # azimuth 0, radius 4, height 2 -> (4, 0, 2) exactly
    p = cr.camera_position(0, 4, 2)
    assert p[0] == pytest.approx(4.0, abs=1e-9)
    assert p[1] == pytest.approx(0.0, abs=1e-9)
    assert p[2] == pytest.approx(2.0, abs=1e-9)


def test_position_az90():
    # azimuth 90, radius 4, height 2 -> (0, 4, 2) within 1e-3
    p = cr.camera_position(90, 4, 2)
    assert p[0] == pytest.approx(0.0, abs=1e-3)
    assert p[1] == pytest.approx(4.0, abs=1e-3)
    assert p[2] == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# convergence_angle
#
# NOTE: 90 deg azimuth separation maps to exactly 90 deg convergence ONLY when
# the cam->subject rays are horizontal (cams at subject height). With cameras
# above the subject the convergence is smaller (shared +z component). So this
# test isolates the azimuth->convergence relationship by placing both cameras
# AT subject height (h=0, subject at origin).
# ---------------------------------------------------------------------------

def test_convergence_90deg_at_subject_height():
    a = cr.camera_position(0, 4, 0)    # at subject height
    b = cr.camera_position(90, 4, 0)
    ang = cr.convergence_angle(a, b, subject=(0, 0, 0))
    assert ang == pytest.approx(90.0, abs=1e-3)


def test_convergence_180deg_at_subject_height():
    a = cr.camera_position(0, 4, 0)
    b = cr.camera_position(180, 4, 0)
    ang = cr.convergence_angle(a, b, subject=(0, 0, 0))
    assert ang == pytest.approx(180.0, abs=1e-3)


def test_convergence_above_subject_is_less_than_azimuth():
    # Sanity: cameras above the subject -> convergence < azimuth separation.
    a = cr.camera_position(0, 4, 2)
    b = cr.camera_position(90, 4, 2)
    ang = cr.convergence_angle(a, b, subject=(0, 0, 0))
    assert ang < 90.0
    assert ang == pytest.approx(78.463, abs=1e-2)


# ---------------------------------------------------------------------------
# tilt_angle
# ---------------------------------------------------------------------------

def test_tilt_known_value():
    # h=2.0, r=2.5, aim=1.0 -> arctan(0.4) ~ 21.8 deg
    t = cr.tilt_angle(2.0, 2.5, 1.0)
    assert t == pytest.approx(math.degrees(math.atan(0.4)), abs=1e-6)
    assert t == pytest.approx(21.8, abs=0.1)


# ---------------------------------------------------------------------------
# is_valid_layout
# ---------------------------------------------------------------------------

def test_invalid_layout_too_steep():
    # h=5.0, r=1.5 -> tilt ~69 deg > 40 -> invalid
    assert cr.is_valid_layout(5.0, 1.5, CFG) is False


def test_valid_layout_shallow():
    # h=1.5, r=2.5 -> tilt ~11 deg < 40 -> valid
    assert cr.is_valid_layout(1.5, 2.5, CFG) is True


# ---------------------------------------------------------------------------
# evaluate_layout stub
# ---------------------------------------------------------------------------

def test_evaluate_layout_is_stub():
    with pytest.raises(NotImplementedError):
        cr.evaluate_layout(1.5, 2.5, 90, (0, 0, 0), CFG)
