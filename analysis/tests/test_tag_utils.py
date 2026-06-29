# analysis/tests/test_tag_utils.py
#
# Unit tests for the pure-python parts of isaac/tag_utils.py (quat_from_cols, find_joints).
# The pxr-dependent functions (create_tag_quad/chest_pose/place_tag) run only under Isaac.
# Run from repo root:  python3 -m pytest analysis/tests/test_tag_utils.py -v

import math

import pytest

import tag_utils as tu


def test_quat_identity():
    w, x, y, z = tu.quat_from_cols((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert (w, x, y, z) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_quat_90deg_yaw_about_z():
    # local +X -> world +Y, +Y -> -X, +Z -> +Z  ==  +90° about Z
    w, x, y, z = tu.quat_from_cols((0, 1, 0), (-1, 0, 0), (0, 0, 1))
    assert math.sqrt(w * w + x * x + y * y + z * z) == pytest.approx(1.0)
    assert w == pytest.approx(math.cos(math.pi / 4))
    assert z == pytest.approx(math.sin(math.pi / 4))
    assert (x, y) == pytest.approx((0.0, 0.0))


def test_find_joints_picks_shoulders_and_top_spine():
    names = ["Hips", "Spine01", "Spine02", "Spine03", "L_Upperarm", "R_Upperarm", "Neck", "Head"]
    li, ri, ci = tu.find_joints(names)
    assert names[li] == "L_Upperarm"
    assert names[ri] == "R_Upperarm"
    assert names[ci] == "Spine03"          # highest-numbered spine = upper chest


def test_find_joints_missing_returns_none():
    li, ri, ci = tu.find_joints(["Bone1", "Bone2"])
    assert li is None and ri is None
