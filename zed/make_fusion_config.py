import json, argparse, math, os

def rotation_matrix_from_look_at(cam_pos, target_pos):
    """
    Returns 3x3 rotation matrix R where:
    - R's rows are the camera's X (right), Y (up), Z (forward) axes
      expressed in world space
    - Camera looks toward target (forward = target - cam, normalised)
    - World up = Z axis
    """
    cx, cy, cz = cam_pos
    tx, ty, tz = target_pos

    # Forward: cam → target
    fx = tx - cx; fy = ty - cy; fz = tz - cz
    fl = math.sqrt(fx*fx + fy*fy + fz*fz)
    if fl < 1e-9:
        raise ValueError("Camera and target are at the same position")
    fx /= fl; fy /= fl; fz /= fl

    # World up = Z. If forward is nearly parallel to Z, fall back to Y
    if abs(fz) > 0.999:
        ux, uy, uz = 0.0, 1.0, 0.0
    else:
        ux, uy, uz = 0.0, 0.0, 1.0

    # Right = forward × up
    rx = fy*uz - fz*uy
    ry = fz*ux - fx*uz
    rz = fx*uy - fy*ux
    rl = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx /= rl; ry /= rl; rz /= rl

    # Recompute up = right × forward (guarantees orthogonal basis)
    upx = ry*fz - rz*fy
    upy = rz*fx - rx*fz
    upz = rx*fy - ry*fx

    # Rotation matrix rows: [right, recomputed_up, forward]
    return [
        [rx,  ry,  rz ],
        [upx, upy, upz],
        [fx,  fy,  fz ],
    ]

def make_pose_string(pos, R):
    """
    16-float row-major 4x4 string the SDK expects:
    r00 r01 r02 tx | r10 r11 r12 ty | r20 r21 r22 tz | 0 0 0 1
    """
    x, y, z = pos
    v = [
        R[0][0], R[0][1], R[0][2], x,
        R[1][0], R[1][1], R[1][2], y,
        R[2][0], R[2][1], R[2][2], z,
        0.0,     0.0,     0.0,     1.0,
    ]
    return " ".join(f"{n:.6f}" for n in v)

def subject_centric_pose(h, theta_deg, r, S=(0.0, 0.0, 0.0)):
    """Camera B on a ring around S at height h, azimuth theta, radius r."""
    theta = math.radians(theta_deg)
    pos = [S[0] + r*math.cos(theta),
           S[1] + r*math.sin(theta),
           h]
    R = rotation_matrix_from_look_at(pos, S)
    return pos, R

def make_fusion_config(template_path, out_path,
                       cam_a_pos, cam_b_pos,
                       cam_b_h, cam_b_theta, cam_b_r,
                       S=(0.0, 0.0, 0.0)):
    with open(template_path) as f:
        cfg = json.load(f)

    R_a = rotation_matrix_from_look_at(cam_a_pos, S)
    cfg["1001"]["FusionConfiguration"]["pose"] = make_pose_string(cam_a_pos, R_a)

    cam_b_pos_computed, R_b = subject_centric_pose(cam_b_h, cam_b_theta, cam_b_r, S)
    cfg["1002"]["FusionConfiguration"]["pose"] = make_pose_string(cam_b_pos_computed, R_b)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=4)

    return cam_b_pos_computed, R_b

def verify_rotation(R, label):
    """Print the three axes so you can visually sanity-check direction."""
    print(f"\n{label}")
    print(f"  right   (X row): [{R[0][0]:+.3f}, {R[0][1]:+.3f}, {R[0][2]:+.3f}]")
    print(f"  up      (Y row): [{R[1][0]:+.3f}, {R[1][1]:+.3f}, {R[1][2]:+.3f}]")
    print(f"  forward (Z row): [{R[2][0]:+.3f}, {R[2][1]:+.3f}, {R[2][2]:+.3f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template",    required=True)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--cam-a-pos",   default="-2.5 0.0 1.5")
    parser.add_argument("--cam-b-h",     type=float, required=True)
    parser.add_argument("--cam-b-theta", type=float, required=True)
    parser.add_argument("--cam-b-r",     type=float, required=True)
    parser.add_argument("--subject",     default="0.0 0.0 0.0")
    args = parser.parse_args()

    S   = [float(v) for v in args.subject.split()]
    a_p = [float(v) for v in args.cam_a_pos.split()]

    b_p, R_b = make_fusion_config(
        template_path = args.template,
        out_path      = args.out,
        cam_a_pos     = a_p,
        cam_b_pos     = None,
        cam_b_h       = args.cam_b_h,
        cam_b_theta   = args.cam_b_theta,
        cam_b_r       = args.cam_b_r,
        S             = S,
    )

    R_a = rotation_matrix_from_look_at(a_p, S)

    verify_rotation(R_a, f"Cam A at {a_p} looking at {S}")
    verify_rotation(R_b, f"Cam B at {b_p} looking at {S}")

    print(f"\nConfig written: {args.out}")
    print(f"Cam B position : [{b_p[0]:.3f}, {b_p[1]:.3f}, {b_p[2]:.3f}]")
