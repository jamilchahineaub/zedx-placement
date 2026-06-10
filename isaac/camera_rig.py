# isaac/camera_rig.py
#
# Pure-math camera placement for the dual-camera ZED-X ring.
# NO omni / pxr / isaacsim imports — fully testable with plain python3.
#
# The look-at and positioning math is copied VERBATIM from
# zed/make_fusion_config.py so the Isaac side and the fusion side never diverge.
# If you change the math here, change it there too (and vice versa).
#
# Coordinate system: right-handed Z-up, metres. World up = +Z.
#
# This module is the SWEEP BOUNDARY: sweep.py calls only evaluate_layout().

import math


# ---------------------------------------------------------------------------
# Copied verbatim from zed/make_fusion_config.py (keep in sync)
# ---------------------------------------------------------------------------

def rotation_matrix_from_look_at(cam_pos, target_pos):
    """
    3x3 rotation matrix where rows are camera X (right), Y (up), Z (forward).
    Camera looks toward target. World up = Z axis.
    """
    cx, cy, cz = cam_pos
    tx, ty, tz = target_pos

    fx = tx - cx; fy = ty - cy; fz = tz - cz
    fl = math.sqrt(fx*fx + fy*fy + fz*fz)
    if fl < 1e-9:
        raise ValueError(f"Camera {cam_pos} and target {target_pos} are the same point")
    fx /= fl; fy /= fl; fz /= fl

    # World up = Z. If forward is nearly vertical, fall back to Y
    if abs(fz) > 0.999:
        ux, uy, uz = 0.0, 1.0, 0.0
    else:
        ux, uy, uz = 0.0, 0.0, 1.0

    # Right = forward x up
    rx = fy*uz - fz*uy
    ry = fz*ux - fx*uz
    rz = fx*uy - fy*ux
    rl = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx /= rl; ry /= rl; rz /= rl

    # Recompute up = right x forward
    upx = ry*fz - rz*fy
    upy = rz*fx - rx*fz
    upz = rx*fy - ry*fx

    return [
        [rx,  ry,  rz ],
        [upx, upy, upz],
        [fx,  fy,  fz ],
    ]


def camera_position(azimuth_deg, radius, height, subject=(0.0, 0.0, 0.0)):
    """World position of a camera on the ring around subject."""
    az = math.radians(azimuth_deg)
    return [
        subject[0] + radius * math.cos(az),
        subject[1] + radius * math.sin(az),
        height,
    ]


def tilt_angle(h, r, aim_height_m):
    """
    Downward tilt of a camera at height h, radius r, aiming at aim_height_m.
    arctan((h - aim_height_m) / r), in degrees.
    (Same formula as make_fusion_config.compute_tilt_deg.)
    """
    return math.degrees(math.atan((h - aim_height_m) / r))


# ---------------------------------------------------------------------------
# New for camera_rig
# ---------------------------------------------------------------------------

def convergence_angle(pos_a, pos_b, subject=(0.0, 0.0, 0.0)):
    """
    Angle in degrees between the (cam_a -> subject) and (cam_b -> subject)
    ray vectors. This is the true 3D convergence (baseline angle at the
    subject), so it accounts for camera height, not just azimuth separation.
    """
    ax = subject[0] - pos_a[0]; ay = subject[1] - pos_a[1]; az = subject[2] - pos_a[2]
    bx = subject[0] - pos_b[0]; by = subject[1] - pos_b[1]; bz = subject[2] - pos_b[2]

    na = math.sqrt(ax*ax + ay*ay + az*az)
    nb = math.sqrt(bx*bx + by*by + bz*bz)
    if na < 1e-9 or nb < 1e-9:
        raise ValueError("A camera is coincident with the subject")

    dot = (ax*bx + ay*by + az*bz) / (na * nb)
    dot = max(-1.0, min(1.0, dot))   # clamp for numerical safety
    return math.degrees(math.acos(dot))


def is_valid_layout(h, r, cfg):
    """
    True if the downward tilt for this (h, r) is below cfg['max_tilt_deg'].
    Aim height comes from cfg['aim_height_m'].
    """
    aim = cfg["aim_height_m"]
    return tilt_angle(h, r, aim) < cfg["max_tilt_deg"]


def evaluate_layout(h, r, rel_az_deg, subject_pos, cfg):
    """
    THE SWEEP BOUNDARY. sweep.py calls only this function.

    Will run one full episode (Isaac scene + ZED fusion + metrics) and return a
    dict of every metric in results.csv. Stubbed until Phase 6.
    """
    raise NotImplementedError(
        "evaluate_layout not yet implemented — will be filled in Phase 6"
    )
