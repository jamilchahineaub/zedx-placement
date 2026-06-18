# analysis/tests/test_fusion_pose.py
#
# Pure-python (no ZED SDK) checks of zed/make_fusion_config.py after the
# override_gravity fix: the file pose must round-trip to the true (P·R, P·t)
# runtime pose, and the camera must aim at the REAL aim point (true tilt), NOT
# the old doubled-pitch aim.

import json
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "zed"))
import make_fusion_config as mfc  # noqa: E402

TEMPLATE = os.path.join(_REPO, "zed", "zed360_template.json")


def _matvec(M, v):
    return [M[i][0] * v[0] + M[i][1] * v[1] + M[i][2] * v[2] for i in range(3)]


def test_D_is_involution():
    I = mfc._matmul3(mfc._D, mfc._D)
    for i in range(3):
        for j in range(3):
            assert abs(I[i][j] - (1.0 if i == j else 0.0)) < 1e-12


def test_file_pose_recovers_runtime_PR_Pt():
    # The SDK converts file->runtime by D·R·D and D·t. That must equal (P·R, P·t)
    # — the absolute world<-cam pose Fusion applies under override_gravity=True.
    pos = [2.5, 0.0, 1.5]
    aim = [0.0, 0.0, 1.0]
    R = mfc.proper_rotation_world_from_cam(pos, aim)
    file_pos, file_R = mfc.convert_isaac_to_zed_pose(pos, R)

    runtime_R = mfc._matmul3(mfc._matmul3(mfc._D, file_R), mfc._D)
    runtime_pos = _matvec(mfc._D, file_pos)
    PR = mfc._matmul3(mfc._P, R)
    Pt = _matvec(mfc._P, pos)

    for i in range(3):
        assert abs(runtime_pos[i] - Pt[i]) < 1e-9
        for j in range(3):
            assert abs(runtime_R[i][j] - PR[i][j]) < 1e-9


def test_generate_uses_true_aim_not_doubled(tmp_path):
    # tilt ~31 deg (h2.5/r2.5): true vs doubled aim differ a lot here.
    h, r, rel_az = 2.5, 2.5, 90.0
    out = tmp_path / "f.json"
    mfc.generate(TEMPLATE, str(out), h, r, rel_az,
                 subject_pos=(0, 0, 0), aim_height_m=1.0, cam_a_az=0)
    pose = json.load(open(out))["1001"]["FusionConfiguration"]["pose"]

    pos_a = mfc.camera_position(0, r, h, (0, 0, 0))
    real_aim = [0.0, 0.0, 1.0]
    zp, zR = mfc.convert_isaac_to_zed_pose(
        pos_a, mfc.proper_rotation_world_from_cam(pos_a, real_aim))
    assert pose == mfc.make_pose_string(zp, zR)             # true aim

    doubled = [0.0, 0.0, pos_a[2] - 2.0 * (pos_a[2] - 1.0)]   # the OLD aim
    zp2, zR2 = mfc.convert_isaac_to_zed_pose(
        pos_a, mfc.proper_rotation_world_from_cam(pos_a, doubled))
    assert pose != mfc.make_pose_string(zp2, zR2)           # NOT doubled


def test_camera_aims_down_by_true_tilt():
    # forward = -(3rd column) of the proper rotation (cols [right, up, -forward]).
    h, r = 2.5, 2.5
    pos = mfc.camera_position(0, r, h, (0, 0, 0))
    R = mfc.proper_rotation_world_from_cam(pos, [0.0, 0.0, 1.0])
    fwd = [-R[0][2], -R[1][2], -R[2][2]]
    tilt_meas = math.degrees(math.atan2(-fwd[2], math.hypot(fwd[0], fwd[1])))
    assert abs(tilt_meas - mfc.compute_tilt_deg(h, r, 1.0)) < 1e-6   # true tilt, not 2x
