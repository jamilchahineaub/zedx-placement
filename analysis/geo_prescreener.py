# analysis/geo_prescreener.py
#
# Geometric pre-filter for camera layouts. Plain python3 — no omni, no pyzed.
# Decides whether a (cam_a, cam_b) layout is worth running a full Isaac episode,
# using the VRST two-camera triangulable criterion (NOT Kim & Jo single-view).
#
# Reuses the camera math from isaac/camera_rig.py (camera_position,
# rotation_matrix_from_look_at, convergence_angle) so geometry never diverges.
#
# Coordinate system: right-handed Z-up, metres.

import math

import camera_rig as cr


# ---------------------------------------------------------------------------
# Canonical standing skeleton — PLACEHOLDER.
#
# This is a synthetic 14-joint standing pose (hip -> head), total height 1.75 m,
# defined relative to a subject origin at the floor (z grows upward). It exists so
# the prescreener can filter layouts NOW, before any Isaac ground-truth exists.
#
# >>> IN PHASE 6 this will be REPLACED by real GT joint positions read from a
# >>> ground_truth_*.csv produced by isaac/gt_logger.py. Once a CSV exists, swap
# >>> this constant for measured joints and the prescreener becomes accurate
# >>> (correct limb spread, real proportions) with no other code change — the
# >>> visibility/triangulability math stays identical.
#
# Format: list of (joint_name, x_offset_m, y_offset_m, z_height_m) relative to the
# subject's floor position. Lateral spread on shoulders/elbows/wrists/hips/knees so
# the set is not a degenerate vertical line (convergence math needs off-axis points).
# ---------------------------------------------------------------------------
CANONICAL_SKELETON = [
    # name        x      y     z (metres above floor)
    ("pelvis",    0.00,  0.00, 0.96),
    ("spine",     0.00,  0.00, 1.08),
    ("chest",     0.00,  0.00, 1.26),
    ("neck",      0.00,  0.00, 1.44),
    ("head",      0.00,  0.00, 1.62),
    ("l_shoulder", -0.18, 0.00, 1.40),
    ("r_shoulder",  0.18, 0.00, 1.40),
    ("l_elbow",   -0.22,  0.00, 1.12),
    ("r_elbow",    0.22,  0.00, 1.12),
    ("l_wrist",   -0.22,  0.00, 0.85),
    ("r_wrist",    0.22,  0.00, 0.85),
    ("l_hip",     -0.10,  0.00, 0.92),
    ("r_hip",      0.10,  0.00, 0.92),
    ("l_knee",    -0.10,  0.00, 0.50),
]


def canonical_skeleton(subject=(0.0, 0.0, 0.0)):
    """The CANONICAL_SKELETON translated to a subject floor position -> list of [x,y,z]."""
    sx, sy, sz = subject
    return [[sx + xo, sy + yo, sz + z] for (_n, xo, yo, z) in CANONICAL_SKELETON]


# ---------------------------------------------------------------------------
# Per-camera visibility
# ---------------------------------------------------------------------------

def _joint_in_view(cam_pos, R, joint, cfg_zed):
    """
    True if `joint` is inside the camera's FOV cone AND within range.
    R rows are right(x)/up(y)/forward(z), from rotation_matrix_from_look_at.
    """
    vx = joint[0] - cam_pos[0]
    vy = joint[1] - cam_pos[1]
    vz = joint[2] - cam_pos[2]

    dist = math.sqrt(vx*vx + vy*vy + vz*vz)
    if dist < cfg_zed["min_range_m"] or dist > cfg_zed["max_range_m"]:
        return False

    right, up, fwd = R[0], R[1], R[2]
    f = vx*fwd[0] + vy*fwd[1] + vz*fwd[2]
    if f <= 0.0:
        return False  # behind the camera

    h_comp = vx*right[0] + vy*right[1] + vz*right[2]
    v_comp = vx*up[0]    + vy*up[1]    + vz*up[2]

    ang_h = math.degrees(math.atan2(abs(h_comp), f))
    ang_v = math.degrees(math.atan2(abs(v_comp), f))

    return ang_h <= cfg_zed["fov_h_deg"] / 2.0 and ang_v <= cfg_zed["fov_v_deg"] / 2.0


def _joint_convergence(cam_a, cam_b, joint):
    """Angle (deg) between (cam_a->joint) and (cam_b->joint) ray vectors."""
    ax = cam_a[0] - joint[0]; ay = cam_a[1] - joint[1]; az = cam_a[2] - joint[2]
    bx = cam_b[0] - joint[0]; by = cam_b[1] - joint[1]; bz = cam_b[2] - joint[2]
    na = math.sqrt(ax*ax + ay*ay + az*az)
    nb = math.sqrt(bx*bx + by*by + bz*bz)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    d = (ax*bx + ay*by + az*bz) / (na * nb)
    d = max(-1.0, min(1.0, d))
    return math.degrees(math.acos(d))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prescreen(cam_a_pos, cam_b_pos, joints, cfg, subject=(0.0, 0.0, 0.0)):
    """
    Geometric VRST pre-screen of a two-camera layout.

    cam_a_pos, cam_b_pos : [x,y,z] world camera positions
    joints               : list of [x,y,z] joint positions; if None, the canonical
                           standing skeleton at `subject` is used.
    cfg                  : experiment config dict (needs cfg['zed_x'],
                           cfg['triangulable_min_deg'], cfg['triangulable_max_deg'],
                           cfg['aim_height_m']).
    subject              : subject floor position (for the aim point + default skel).

    Returns a dict:
      passed                            bool (joints_visible_both_triangulable >= 0.70)
      convergence_angle_deg             whole-body convergence at the subject
      joints_visible_cam_a              fraction in cam A FOV+range
      joints_visible_cam_b              fraction in cam B FOV+range
      joints_visible_either             fraction in A or B
      joints_visible_both_triangulable  VRST fraction (in both + convergence in band)
      unique_contribution_cam_b         fraction B sees that A misses
    """
    cfg_zed = cfg["zed_x"]
    tri_min = cfg["triangulable_min_deg"]
    tri_max = cfg["triangulable_max_deg"]
    aim_h = cfg["aim_height_m"]

    if joints is None:
        joints = canonical_skeleton(subject)

    aim_point = [subject[0], subject[1], subject[2] + aim_h]
    R_a = cr.rotation_matrix_from_look_at(cam_a_pos, aim_point)
    R_b = cr.rotation_matrix_from_look_at(cam_b_pos, aim_point)

    n = len(joints)
    if n == 0:
        raise ValueError("prescreen got an empty joint list")

    vis_a = 0
    vis_b = 0
    vis_either = 0
    both_tri = 0
    unique_b = 0

    for j in joints:
        in_a = _joint_in_view(cam_a_pos, R_a, j, cfg_zed)
        in_b = _joint_in_view(cam_b_pos, R_b, j, cfg_zed)

        vis_a += in_a
        vis_b += in_b
        vis_either += (in_a or in_b)
        if in_b and not in_a:
            unique_b += 1
        if in_a and in_b:
            cang = _joint_convergence(cam_a_pos, cam_b_pos, j)
            if tri_min <= cang <= tri_max:
                both_tri += 1

    frac_both_tri = both_tri / n

    return {
        "passed": frac_both_tri >= 0.70,
        "convergence_angle_deg": cr.convergence_angle(cam_a_pos, cam_b_pos, subject),
        "joints_visible_cam_a": vis_a / n,
        "joints_visible_cam_b": vis_b / n,
        "joints_visible_either": vis_either / n,
        "joints_visible_both_triangulable": frac_both_tri,
        "unique_contribution_cam_b": unique_b / n,
    }
