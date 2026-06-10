# zed/make_fusion_config.py
# Generates a per-layout fusion config from the template.
# Both cameras sit at the same height and radius.
# Cam A is at azimuth 0, cam B is at relative_azimuth from A.
# Cameras aim at hip height (aim_height_m from experiment.yaml).
# ZED SDK 5.3.1 verified.
#
# COORDINATE SYSTEMS (see CLAUDE.md):
#   Isaac side is RIGHT_HANDED_Z_UP. The ZED fusion file is read by the SDK in
#   RIGHT_HANDED_Y_UP (this is what the official multi-camera body-tracking sample
#   does). So the camera poses written here MUST be converted Z-up -> Y-up, matching
#   scripts/convert_isaac_pose_to_zed_fusion.py from the zed-isaac-sim repo:
#       ZED translation = (-isaac_y, -isaac_z, isaac_x)
#       ZED rotation    = P R P^T   with   P (x,y,z) = (-y, -z, x)
#   Without this conversion the fused cameras land in the wrong relative poses and
#   wide-baseline layouts produce garbage. convert_isaac_to_zed_pose() applies it.

import json, argparse, math, os, yaml


def load_config(experiment_yaml="config/experiment.yaml"):
    with open(experiment_yaml) as f:
        return yaml.safe_load(f)


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


# Isaac (x,y,z) -> ZED Y-up (-y,-z,x). Signed permutation, det = +1.
_P = [
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
]


def _matmul3(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose3(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]


def proper_rotation_world_from_cam(cam_pos, target_pos):
    """
    Proper (det = +1) world<-camera rotation, columns [right, up, -forward]
    (OpenGL/USD convention: the camera looks down its own -Z).

    NOTE: camera_rig.rotation_matrix_from_look_at returns rows [right, up, +forward],
    which is a LEFT-handed basis (det = -1) — fine for direction math, but invalid as
    a rigid-body rotation in a fusion pose. We rebuild a proper rotation here so the
    fusion 4x4 is a valid SE(3) transform. World up = +Z.
    """
    cx, cy, cz = cam_pos
    tx, ty, tz = target_pos
    fx, fy, fz = tx - cx, ty - cy, tz - cz
    fl = math.sqrt(fx*fx + fy*fy + fz*fz)
    if fl < 1e-9:
        raise ValueError(f"Camera {cam_pos} and target {target_pos} are the same point")
    fx, fy, fz = fx/fl, fy/fl, fz/fl

    # up = +Z unless forward is near-vertical, then fall back to +Y
    if abs(fz) > 0.999:
        ux, uy, uz = 0.0, 1.0, 0.0
    else:
        ux, uy, uz = 0.0, 0.0, 1.0

    # right = forward x up
    rx = fy*uz - fz*uy
    ry = fz*ux - fx*uz
    rz = fx*uy - fy*ux
    rl = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx, ry, rz = rx/rl, ry/rl, rz/rl

    # true up = right x forward
    upx = ry*fz - rz*fy
    upy = rz*fx - rx*fz
    upz = rx*fy - ry*fx

    # columns [right, up, -forward] -> stored row-major (matrix[row][col])
    return [
        [rx,  upx, -fx],
        [ry,  upy, -fy],
        [rz,  upz, -fz],
    ]


def convert_isaac_to_zed_pose(pos, R_world_from_cam):
    """
    Convert an Isaac Z-up world<-camera pose to the ZED fusion Y-up frame.

    pos               : [x, y, z] camera world position (Isaac Z-up, metres)
    R_world_from_cam  : 3x3 proper rotation (det +1), Isaac Z-up

    Returns (zed_pos, zed_R):
      zed_pos = (-y, -z, x)
      zed_R   = P R P^T
    Both remain a valid rigid transform (det +1) in Y-up.
    """
    zed_pos = [-pos[1], -pos[2], pos[0]]
    zed_R = _matmul3(_matmul3(_P, R_world_from_cam), _transpose3(_P))
    return zed_pos, zed_R


def make_pose_string(pos, R):
    """Row-major 4x4 transform string: r00 r01 r02 tx r10 r11 r12 ty ..."""
    x, y, z = pos
    v = [
        R[0][0], R[0][1], R[0][2], x,
        R[1][0], R[1][1], R[1][2], y,
        R[2][0], R[2][1], R[2][2], z,
        0.0,     0.0,     0.0,     1.0,
    ]
    return " ".join(f"{n:.6f}" for n in v)


def compute_tilt_deg(h, r, aim_h):
    return math.degrees(math.atan((h - aim_h) / r))


def camera_position(azimuth_deg, radius, height, subject=(0.0, 0.0, 0.0)):
    """World position of a camera on the ring around subject."""
    az = math.radians(azimuth_deg)
    return [
        subject[0] + radius * math.cos(az),
        subject[1] + radius * math.sin(az),
        height,
    ]


def generate(template_path, out_path, h, r, rel_az_deg,
             subject_pos=(0.0, 0.0, 0.0), aim_height_m=1.0,
             cam_a_az=0):
    """
    h           : height of both cameras in metres
    r           : radius of both cameras from subject in metres
    rel_az_deg  : azimuth of cam B relative to cam A in degrees
    subject_pos : where the subject stands
    aim_height_m: cameras aim at this height above subject floor pos
    cam_a_az    : azimuth of cam A in degrees (default 0)
    """
    with open(template_path) as f:
        cfg = json.load(f)

    aim_point = [subject_pos[0], subject_pos[1], subject_pos[2] + aim_height_m]

    # Camera A — build a proper (det +1) Isaac Z-up pose, then convert to ZED Y-up.
    pos_a = camera_position(cam_a_az, r, h, subject_pos)
    R_a   = proper_rotation_world_from_cam(pos_a, aim_point)
    zpos_a, zR_a = convert_isaac_to_zed_pose(pos_a, R_a)
    cfg["1001"]["FusionConfiguration"]["pose"] = make_pose_string(zpos_a, zR_a)

    # Camera B
    pos_b = camera_position(cam_a_az + rel_az_deg, r, h, subject_pos)
    R_b   = proper_rotation_world_from_cam(pos_b, aim_point)
    zpos_b, zR_b = convert_isaac_to_zed_pose(pos_b, R_b)
    cfg["1002"]["FusionConfiguration"]["pose"] = make_pose_string(zpos_b, zR_b)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=4)

    tilt = compute_tilt_deg(h, r, aim_height_m)
    return {
        "pos_a": pos_a, "pos_b": pos_b,
        "tilt_deg": round(tilt, 1),
        "convergence_approx_deg": round(rel_az_deg, 1),
    }


def print_layout(info, h, r, rel_az):
    print(f"\nLayout: h={h}m  r={r}m  rel_az={rel_az}°")
    print(f"  Cam A position : [{info['pos_a'][0]:.3f}, {info['pos_a'][1]:.3f}, {info['pos_a'][2]:.3f}]")
    print(f"  Cam B position : [{info['pos_b'][0]:.3f}, {info['pos_b'][1]:.3f}, {info['pos_b'][2]:.3f}]")
    print(f"  Downward tilt  : {info['tilt_deg']}°")
    print(f"  Rel azimuth    : {info['convergence_approx_deg']}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template",   default="zed/zed360_template.json")
    parser.add_argument("--out",        required=True)
    parser.add_argument("--h",          type=float, required=True, help="height in metres")
    parser.add_argument("--r",          type=float, required=True, help="radius in metres")
    parser.add_argument("--rel-az",     type=float, required=True, help="relative azimuth of cam B in degrees")
    parser.add_argument("--subject",    default="0.0 0.0 0.0")
    parser.add_argument("--experiment", default="config/experiment.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.experiment)
    aim_h   = cfg.get("aim_height_m", 1.0)
    max_t   = cfg.get("max_tilt_deg", 40.0)
    cam_a_az= cfg["cam_a"]["azimuth_deg"]
    S       = [float(v) for v in args.subject.split()]

    tilt = compute_tilt_deg(args.h, args.r, aim_h)
    if tilt >= max_t:
        print(f"SKIPPED — tilt {tilt:.1f}° exceeds max {max_t}°")
        exit(1)

    info = generate(
        template_path = args.template,
        out_path      = args.out,
        h             = args.h,
        r             = args.r,
        rel_az_deg    = args.rel_az,
        subject_pos   = S,
        aim_height_m  = aim_h,
        cam_a_az      = cam_a_az,
    )

    print_layout(info, args.h, args.r, args.rel_az)
    print(f"\nConfig written: {args.out}")
