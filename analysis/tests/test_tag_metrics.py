# analysis/tests/test_tag_metrics.py
#
# Unit tests for analysis/tag_metrics.py (visibility ratio from a detection log). Plain python3.
# Run from repo root:  python3 -m pytest analysis/tests/test_tag_metrics.py -v

import csv

import pytest

import tag_metrics


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "wall_clock", "cam", "ids"])
        w.writerows(rows)


def test_all_seen_front_on_cam_a(tmp_path):
    p = tmp_path / "d.csv"
    rows = []
    for fi in range(1, 11):
        rows += [(fi, fi * 0.1, "A", "23"), (fi, fi * 0.1, "B", ""), (fi, fi * 0.1, "C", "")]
    _write(p, rows)
    m = tag_metrics.compute(str(p), front_id=23, back_id=42)
    assert m["tag_visibility_ratio"] == pytest.approx(1.0)
    assert m["detect_rate_cam_a"] == pytest.approx(1.0)
    assert m["detect_rate_cam_b"] == pytest.approx(0.0)
    assert m["detect_rate_front"] == pytest.approx(1.0)
    assert m["detect_rate_back"] == pytest.approx(0.0)
    assert m["longest_blind_gap_s"] == pytest.approx(0.0)
    assert m["frames"] == 10


def test_blind_gap_in_middle(tmp_path):
    p = tmp_path / "d.csv"
    # back tag (id 42) seen at the ends, blind frames 3..8 (6 consecutive), dt = 0.1 s
    rows = [(fi, fi * 0.1, "C", "42" if fi in (1, 2, 9, 10) else "") for fi in range(1, 11)]
    _write(p, rows)
    m = tag_metrics.compute(str(p), front_id=23, back_id=42)
    assert m["tag_visibility_ratio"] == pytest.approx(0.4)
    assert m["detect_rate_back"] == pytest.approx(0.4)
    assert m["detect_rate_front"] == pytest.approx(0.0)
    assert m["longest_blind_gap_s"] == pytest.approx(0.6)


def test_either_camera_counts(tmp_path):
    p = tmp_path / "d.csv"
    # frame seen if A OR C decodes; here A every even frame, C every odd -> always covered
    rows = []
    for fi in range(1, 11):
        rows += [(fi, fi * 0.1, "A", "23" if fi % 2 == 0 else ""),
                 (fi, fi * 0.1, "C", "42" if fi % 2 == 1 else "")]
    _write(p, rows)
    m = tag_metrics.compute(str(p), front_id=23, back_id=42)
    assert m["tag_visibility_ratio"] == pytest.approx(1.0)
    assert m["detect_rate_cam_a"] == pytest.approx(0.5)
    assert m["detect_rate_cam_c"] == pytest.approx(0.5)


def test_empty_log_is_none(tmp_path):
    p = tmp_path / "d.csv"
    _write(p, [])
    assert tag_metrics.compute(str(p), 23, 42) is None
