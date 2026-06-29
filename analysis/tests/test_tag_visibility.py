# analysis/tests/test_tag_visibility.py
#
# Unit tests for analysis/tag_visibility.py (plain python3).
# Run from repo root:  python3 -m pytest analysis/tests/test_tag_visibility.py -v

import math

import pytest

import camera_rig as cr
import tag_visibility as tv


CFG_ZED = {"fov_h_deg": 110.0, "fov_v_deg": 70.0, "min_range_m": 0.3, "max_range_m": 10.0}
DECODE_MAX = 60.0
MAX_RANGE = 6.0
CHEST = 1.30


def _cam_aimed_at(pos, tag_pos):
    return (pos, cr.rotation_matrix_from_look_at(pos, tag_pos))


# --------------------------------------------------------------------------- tag_visible

def test_front_camera_decodes():
    tag = [0.0, 0.0, CHEST]
    normal = [1.0, 0.0, 0.0]                 # tag faces +X
    cam = _cam_aimed_at([3.0, 0.0, CHEST], tag)   # directly in front, 3 m
    assert tv.tag_visible(*cam, tag, normal, CFG_ZED, DECODE_MAX, MAX_RANGE) is True


def test_behind_camera_cannot_decode():
    tag = [0.0, 0.0, CHEST]
    normal = [1.0, 0.0, 0.0]
    cam = _cam_aimed_at([-3.0, 0.0, CHEST], tag)  # behind the tag (normal points away)
    assert tv.tag_visible(*cam, tag, normal, CFG_ZED, DECODE_MAX, MAX_RANGE) is False


def test_just_past_decode_cone_fails():
    tag = [0.0, 0.0, CHEST]
    normal = [1.0, 0.0, 0.0]
    # camera at 70 deg off the normal (>60) -> too oblique to decode
    a = math.radians(70.0)
    cam = _cam_aimed_at([3.0 * math.cos(a), 3.0 * math.sin(a), CHEST], tag)
    assert tv.tag_visible(*cam, tag, normal, CFG_ZED, DECODE_MAX, MAX_RANGE) is False
    # ...but at 50 deg (<60) it decodes
    a2 = math.radians(50.0)
    cam2 = _cam_aimed_at([3.0 * math.cos(a2), 3.0 * math.sin(a2), CHEST], tag)
    assert tv.tag_visible(*cam2, tag, normal, CFG_ZED, DECODE_MAX, MAX_RANGE) is True


def test_beyond_decode_range_fails():
    tag = [0.0, 0.0, CHEST]
    normal = [1.0, 0.0, 0.0]
    cam = _cam_aimed_at([8.0, 0.0, CHEST], tag)   # 8 m > max_decode_range 6 m
    assert tv.tag_visible(*cam, tag, normal, CFG_ZED, DECODE_MAX, MAX_RANGE) is False


# --------------------------------------------------------------------------- cell_coverage

def _facings(step=15):
    return [i * step for i in range(int(360 / step))]


def test_three_cams_120deg_apart_cover_all_facings():
    aim = [0.0, 0.0, CHEST]
    cams = [tv.make_cam(az, 3.0, CHEST, aim) for az in (0, 120, 240)]
    frac, _ = tv.cell_coverage(cams, (0.0, 0.0), CHEST, _facings(), CFG_ZED, DECODE_MAX, MAX_RANGE)
    assert frac == pytest.approx(1.0)        # every facing within 60 deg of some camera


def test_two_same_side_cams_leave_blind_wedge():
    aim = [0.0, 0.0, CHEST]
    cams = [tv.make_cam(az, 3.0, CHEST, aim) for az in (-30, 30)]   # both on +X side
    frac, bools = tv.cell_coverage(cams, (0.0, 0.0), CHEST, _facings(), CFG_ZED,
                                   DECODE_MAX, MAX_RANGE)
    assert frac < 0.6                         # ~half the facings blind
    assert tv.largest_blind_wedge(bools, 15) >= 150.0   # a big blind arc on the far side


# --------------------------------------------------------------------------- coverage_map

def _cfg_small():
    return {
        "zed_x": CFG_ZED,
        "workspace": {"center": [0.0, 0.0], "size_m": [2.0, 2.0]},
        "grid_cell_m": 1.0,                   # 2x2 grid (4 cells)
        "aruco": {"chest_height_m": CHEST, "decode_max_deg": DECODE_MAX,
                  "max_decode_range_m": MAX_RANGE, "facing_step_deg": 15},
    }


def test_two_sided_covers_more_than_one_sided():
    aim = [0.0, 0.0, CHEST]
    cams = [tv.make_cam(az, 3.0, CHEST, aim) for az in (-30, 30)]   # both on +X side
    one, _ = tv.cell_coverage(cams, (0, 0), CHEST, _facings(), CFG_ZED, DECODE_MAX,
                              MAX_RANGE, two_sided=False)
    two, _ = tv.cell_coverage(cams, (0, 0), CHEST, _facings(), CFG_ZED, DECODE_MAX,
                              MAX_RANGE, two_sided=True)
    assert two > one                      # the back tag covers the opposite-facing hemisphere
    assert two >= 0.95                    # two same-side cams + two faces ~= full coverage


def _cfg_full():
    return {
        "zed_x": CFG_ZED, "aim_height_m": 1.0,
        "workspace": {"center": [0.0, 0.0], "size_m": [4.0, 4.0]}, "grid_cell_m": 1.0,
        "cam_a": {"azimuth_deg": 0},
        "aruco": {"chest_height_m": CHEST, "decode_max_deg": DECODE_MAX,
                  "max_decode_range_m": MAX_RANGE, "facing_step_deg": 30,
                  "cam_c_azimuth_step_deg": 30},
    }


def test_best_cam_c_azimuths_excludes_near_a_and_b():
    cfg = _cfg_full()
    picks = tv.best_cam_c_azimuths(h=3.5, r=4.5, rel_az=180, cfg=cfg, n=4, min_sep_deg=45)
    assert picks                                  # some valid candidates exist
    for az, worst, mean in picks:
        assert tv._circ_dist(az, 0) >= 45         # not clustered with cam A (az 0)
        assert tv._circ_dist(az, 180) >= 45       # not clustered with cam B (az 180)
    # returned best-first by worst-cell coverage
    worsts = [w for _, w, _ in picks]
    assert worsts == sorted(worsts, reverse=True)


def test_coverage_map_three_cams_better_than_two():
    cfg = _cfg_small()
    aim = [0.0, 0.0, CHEST]
    two = [tv.make_cam(az, 3.0, CHEST, aim) for az in (0, 150)]
    three = two + [tv.make_cam(270, 3.0, CHEST, aim)]
    cm2 = tv.coverage_map(two, cfg)
    cm3 = tv.coverage_map(three, cfg)
    # adding a 3rd camera never reduces coverage and should lift the worst cell
    assert cm3["worst"] >= cm2["worst"]
    assert cm3["mean"] >= cm2["mean"]
    assert 0.0 <= cm3["worst"] <= cm3["mean"] <= 1.0
    assert set(cm3["cells"].keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
