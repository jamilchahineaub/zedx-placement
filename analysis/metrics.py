# analysis/metrics.py
#
# Accuracy metrics: ZED body-tracking predictions vs Isaac ground truth.
# Plain python3 (csv/json/math + yaml only — no omni, no pyzed).
#
# Frame/alignment strategy (static character — animation is frozen, confirmed
# bit-identical pelvis across frames): TIME-AVERAGE both skeletons per joint,
# then compare the averaged skeletons. This is robust to detector jitter and
# avoids cross-process wall-clock alignment entirely. When motion lands
# (omni.anim.people follow-up) this module needs a per-frame association pass;
# jitter_variance / id_drops are NaN until then.
#
# Coordinate frames:
#   GT CSV     : Isaac Z-up world.
#   fusion CSV : the fusion world frame = the Y-up conversion of Isaac world
#                that make_fusion_config writes into the camera poses
#                (zed = P @ isaac with P(x,y,z) = (-y,-z,x)), so
#                isaac = (z_zed, -x_zed, -y_zed).
#   single CSV : the camera's own Y-up frame (world anchored at the camera at
#                init). p_isaac = R_wc @ p_cam + cam_pos with R_wc the proper
#                world<-camera rotation (columns [right, up, -forward]) reused
#                from zed/make_fusion_config.proper_rotation_world_from_cam.

import csv
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "isaac"), os.path.join(_REPO, "zed")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import camera_rig                      # noqa: E402  (isaac/ — pure math)
import geo_prescreener                 # noqa: E402  (analysis/)
import joint_map                       # noqa: E402  (analysis/)
from make_fusion_config import proper_rotation_world_from_cam  # noqa: E402


NAN = float("nan")


# ---------------------------------------------------------------------------
# CSV loading (time-averaged skeletons)
# ---------------------------------------------------------------------------

def load_gt_average(gt_csv, joint_filter=None):
    """Average each joint's world position over all frames.
    Returns ({joint_name: (x,y,z)}, n_frames)."""
    sums, counts = {}, {}
    times = set()
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            name = row["joint_name"]
            if joint_filter is not None and name not in joint_filter:
                continue
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            sx, sy, sz = sums.get(name, (0.0, 0.0, 0.0))
            sums[name] = (sx + x, sy + y, sz + z)
            counts[name] = counts.get(name, 0) + 1
            times.add(row["sim_time"])
    avg = {n: (s[0] / counts[n], s[1] / counts[n], s[2] / counts[n])
           for n, s in sums.items()}
    return avg, len(times)


def load_pred_average(pred_csv, conf_min=0.0):
    """Average each ZED joint over all rows with confidence >= conf_min.
    Returns ({zed_joint_name: (x,y,z)}, n_rows_used). NaN keypoints skipped."""
    sums, counts = {}, {}
    used = 0
    with open(pred_csv) as f:
        for row in csv.DictReader(f):
            try:
                conf = float(row["confidence"])
            except ValueError:
                continue
            if math.isnan(conf) or conf < conf_min:
                continue
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            if any(math.isnan(v) for v in (x, y, z)):
                continue
            name = row["joint_name"]
            sx, sy, sz = sums.get(name, (0.0, 0.0, 0.0))
            sums[name] = (sx + x, sy + y, sz + z)
            counts[name] = counts.get(name, 0) + 1
            used += 1
    avg = {n: (s[0] / counts[n], s[1] / counts[n], s[2] / counts[n])
           for n, s in sums.items()}
    return avg, used


# ---------------------------------------------------------------------------
# Coordinate transforms (into Isaac Z-up world)
# ---------------------------------------------------------------------------

def fused_to_isaac(p):
    """Invert make_fusion_config's P: zed=(-y,-z,x)  =>  isaac=(z,-x,-y)."""
    return (p[2], -p[0], -p[1])


def single_cam_to_isaac(p_cam, cam_pos, aim_point):
    """Camera-frame Y-up point -> Isaac Z-up world.
    R_wc columns are [right, up, -forward] (camera looks down its own -Z)."""
    R = proper_rotation_world_from_cam(cam_pos, aim_point)
    x, y, z = p_cam
    return (
        R[0][0] * x + R[0][1] * y + R[0][2] * z + cam_pos[0],
        R[1][0] * x + R[1][1] * y + R[1][2] * z + cam_pos[1],
        R[2][0] * x + R[2][1] * y + R[2][2] * z + cam_pos[2],
    )


def _gravity_alignment_deg(pred_avg):
    """Diagnostic: angle between the predicted hips->neck axis and +Y in RAW
    ZED coords. ~0 deg => SDK gravity-aligned the frame; ~camera-tilt deg =>
    pure camera frame (our default transform assumption)."""
    need = ("NECK", "LEFT_HIP", "RIGHT_HIP")
    if any(n not in pred_avg for n in need):
        return NAN
    neck = pred_avg["NECK"]
    hip = tuple((pred_avg["LEFT_HIP"][i] + pred_avg["RIGHT_HIP"][i]) / 2.0
                for i in range(3))
    v = tuple(neck[i] - hip[i] for i in range(3))
    norm = math.sqrt(sum(c * c for c in v))
    if norm < 1e-9:
        return NAN
    cosang = max(-1.0, min(1.0, v[1] / norm))
    return math.degrees(math.acos(cosang))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mpjpe_pck(gt_avg, pred_avg_isaac):
    """gt_avg keyed by Isaac names; pred_avg_isaac keyed by ZED names but
    already transformed into Isaac world. Returns (mpjpe_mm, pck30, pck50,
    per_joint_mm dict)."""
    errs = {}
    for zed_name, isaac_name in joint_map.mapped_pairs():
        if zed_name not in pred_avg_isaac or isaac_name not in gt_avg:
            continue
        p, g = pred_avg_isaac[zed_name], gt_avg[isaac_name]
        e_mm = 1000.0 * math.sqrt(sum((p[i] - g[i]) ** 2 for i in range(3)))
        errs[zed_name] = e_mm
    if not errs:
        return NAN, NAN, NAN, {}
    vals = list(errs.values())
    mpjpe = sum(vals) / len(vals)
    pck30 = sum(1 for e in vals if e <= 30.0) / len(vals)
    pck50 = sum(1 for e in vals if e <= 50.0) / len(vals)
    return mpjpe, pck30, pck50, errs


# ---------------------------------------------------------------------------
# Per-frame loading + motion metrics (used when the character is animated)
# ---------------------------------------------------------------------------

def load_gt_per_frame(gt_csv, joint_filter=None):
    """GT grouped into frames by wall_clock (gt_logger writes one wall_clock per
    frame). Returns [ {wall, sim_time, joints:{name:(x,y,z)}} ] sorted by wall."""
    frames = {}
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            name = row["joint_name"]
            if joint_filter is not None and name not in joint_filter:
                continue
            wall = float(row["wall_clock"])
            fr = frames.setdefault(wall, {"wall": wall,
                                          "sim_time": float(row["sim_time"]),
                                          "joints": {}})
            fr["joints"][name] = (float(row["x"]), float(row["y"]), float(row["z"]))
    return [frames[w] for w in sorted(frames)]


def load_pred_per_frame(pred_csv, conf_min=0.0):
    """Prediction grouped into frames by frame_idx, each with per-body joints.
    Returns [ {frame_idx, wall, bodies:{body_id:{tracking_state, joints:{zed_name:(x,y,z)}}}} ]
    sorted by frame_idx. Joints below conf_min (or NaN conf / NaN xyz) are dropped."""
    frames = {}
    with open(pred_csv) as f:
        for row in csv.DictReader(f):
            try:
                fi = int(float(row["frame_idx"]))
            except (KeyError, ValueError):
                continue
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            if any(math.isnan(v) for v in (x, y, z)):
                continue
            try:
                conf = float(row["confidence"])
            except ValueError:
                conf = NAN
            if math.isnan(conf) or conf < conf_min:
                continue
            wall = float(row["wall_clock"])
            bid = int(float(row["body_id"]))
            fr = frames.setdefault(fi, {"frame_idx": fi, "wall": wall, "bodies": {}})
            b = fr["bodies"].setdefault(bid, {"tracking_state": row.get("tracking_state", ""),
                                              "joints": {}})
            b["joints"][row["joint_name"]] = (x, y, z)
    return [frames[k] for k in sorted(frames)]


def associate_frames(gt_frames, pred_frames, offset_s=0.0):
    """Pair each pred frame with the GT frame nearest in wall_clock (+offset).
    Returns [(gt_frame, pred_frame)]."""
    import bisect
    if not gt_frames or not pred_frames:
        return []
    gt_sorted = sorted(gt_frames, key=lambda f: f["wall"])
    gt_walls = [f["wall"] for f in gt_sorted]
    pairs = []
    for pf in pred_frames:
        target = pf["wall"] + offset_s
        i = bisect.bisect_left(gt_walls, target)
        cands = [c for c in (i, i - 1) if 0 <= c < len(gt_walls)]
        j = min(cands, key=lambda k: abs(gt_walls[k] - target))
        pairs.append((gt_sorted[j], pf))
    return pairs


def _has_motion(gt_frames, names, thresh_m=0.02):
    """True if any mapped GT joint moves more than thresh_m across the frames."""
    if len(gt_frames) < 3:
        return False
    for name in names:
        xs, ys, zs = [], [], []
        for fr in gt_frames:
            p = fr["joints"].get(name)
            if p is not None:
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        if len(xs) >= 3:
            rng = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            if rng > thresh_m:
                return True
    return False


def _primary_body(pred_frame):
    """The body in a pred frame with the most mapped ZED joints present."""
    best, best_n = None, 0
    for b in pred_frame["bodies"].values():
        n = sum(1 for z, _ in joint_map.mapped_pairs() if z in b["joints"])
        if n > best_n:
            best, best_n = b, n
    return best


def _gt_matched_body(gt_joints, pred_frame, transform_fn, min_joints=7):
    """Pick the predicted body that best matches the GT person this frame: the
    body with the lowest mean per-joint error to GT (over its mapped joints),
    among bodies exposing at least `min_joints` mapped joints.

    Used ONLY for the 3-camera (overhead) path. A third camera spawns extra
    GHOST bodies (phantom fused skeletons with no real human), and the
    most-complete-skeleton heuristic in _primary_body locks onto a ghost in a
    large fraction of frames, inflating MPJPE/jitter. Matching the scored body
    to the single real person (standard multi-person MPJPE) measures the actual
    tracking quality. Ghost COUNT is still penalized separately via id_drops."""
    best, best_err = None, float("inf")
    for b in pred_frame["bodies"].values():
        e = _frame_errors(gt_joints, b["joints"], transform_fn)
        if len(e) < min_joints:
            continue
        m = sum(e.values()) / len(e)
        if m < best_err:
            best, best_err = b, m
    return best


def _frame_errors(gt_joints, pred_zed_joints, transform_fn):
    """Per-mapped-joint Euclidean error (mm) for one matched frame."""
    errs = {}
    for zed_name, isaac_name in joint_map.mapped_pairs():
        if zed_name not in pred_zed_joints or isaac_name not in gt_joints:
            continue
        p = transform_fn(pred_zed_joints[zed_name])
        g = gt_joints[isaac_name]
        errs[isaac_name] = 1000.0 * math.sqrt(sum((p[i] - g[i]) ** 2 for i in range(3)))
    return errs


def _aligned_and_offset(err_vectors_m):
    """Given per-joint error vectors (pred-gt, metres) for ONE frame/pose, return
    (translation_aligned_mpjpe_mm, registration_offset_mm): the offset is the joint
    centroid of the error (a global translation between the fused and GT frames);
    the aligned MPJPE is the per-joint error after removing that translation (true
    pose fidelity, independent of frame-registration bias)."""
    if not err_vectors_m:
        return NAN, NAN
    n = len(err_vectors_m)
    off = [sum(v[k] for v in err_vectors_m) / n for k in range(3)]
    aln = 1000.0 * sum(math.sqrt(sum((v[k] - off[k]) ** 2 for k in range(3)))
                       for v in err_vectors_m) / n
    off_mm = 1000.0 * math.sqrt(sum(o * o for o in off))
    return aln, off_mm


def mpjpe_pck_per_frame(pairs, transform_fn, select=None):
    """Pool per-(frame,joint) errors over matched frames. Alignment (offset removal)
    is done PER FRAME. Returns (mpjpe, pck30, pck50, per_joint_mean, mpjpe_aligned,
    registration_offset).

    select: optional callable (gt_f, pred_f) -> body, choosing which predicted
    body to score each frame. Defaults to _primary_body (2-camera behavior). The
    3-camera path passes a GT-matched selector to ignore ghost bodies."""
    per_joint = {}
    abs_pool, aln_pool, offs = [], [], []
    for gt_f, pred_f in pairs:
        body = select(gt_f, pred_f) if select else _primary_body(pred_f)
        if body is None:
            continue
        vecs, names = [], []
        for zed_name, isaac_name in joint_map.mapped_pairs():
            if zed_name in body["joints"] and isaac_name in gt_f["joints"]:
                p = transform_fn(body["joints"][zed_name])
                g = gt_f["joints"][isaac_name]
                vecs.append([p[k] - g[k] for k in range(3)])
                names.append(isaac_name)
        if not vecs:
            continue
        n = len(vecs)
        off = [sum(v[k] for v in vecs) / n for k in range(3)]
        offs.append(1000.0 * math.sqrt(sum(o * o for o in off)))
        for v, nm in zip(vecs, names):
            e = 1000.0 * math.sqrt(sum(v[k] ** 2 for k in range(3)))
            ea = 1000.0 * math.sqrt(sum((v[k] - off[k]) ** 2 for k in range(3)))
            abs_pool.append(e)
            aln_pool.append(ea)
            per_joint.setdefault(nm, []).append(e)
    if not abs_pool:
        return NAN, NAN, NAN, {}, NAN, NAN
    mpjpe = sum(abs_pool) / len(abs_pool)
    pck30 = sum(1 for v in abs_pool if v <= 30.0) / len(abs_pool)
    pck50 = sum(1 for v in abs_pool if v <= 50.0) / len(abs_pool)
    per_joint_mean = {k: sum(vs) / len(vs) for k, vs in per_joint.items()}
    mpjpe_aln = sum(aln_pool) / len(aln_pool)
    reg_off = sum(offs) / len(offs)
    return mpjpe, pck30, pck50, per_joint_mean, mpjpe_aln, reg_off


def compute_jitter_variance(pairs, transform_fn, select=None):
    """Mean over joints of the temporal variance (mm^2) of the per-joint
    prediction error across matched frames — how much tracking wobbles around
    truth. Lower is better. NaN if too few frames.

    select: see mpjpe_pck_per_frame. Default _primary_body (2-cam); the 3-cam
    path passes a GT-matched selector so jitter reflects the real person, not
    frame-to-frame ghost swaps."""
    per_joint = {}
    for gt_f, pred_f in pairs:
        body = select(gt_f, pred_f) if select else _primary_body(pred_f)
        if body is None:
            continue
        for k, v in _frame_errors(gt_f["joints"], body["joints"], transform_fn).items():
            per_joint.setdefault(k, []).append(v)
    variances = []
    for series in per_joint.values():
        if len(series) >= 2:
            m = sum(series) / len(series)
            variances.append(sum((x - m) ** 2 for x in series) / (len(series) - 1))
    return (sum(variances) / len(variances)) if variances else NAN


def _is_ok(tracking_state):
    return "OK" in str(tracking_state)


def compute_id_drops(pred_frames):
    """(#distinct OK body-ids over the window - 1) + (#tracked->untracked transitions).
    Captures identity fragmentation and tracking gaps for the human."""
    ok_seq = []
    for fr in pred_frames:
        ok_seq.append({bid for bid, b in fr["bodies"].items() if _is_ok(b["tracking_state"])})
    distinct = set().union(*ok_seq) if ok_seq else set()
    id_switch = max(0, len(distinct) - 1)
    transitions = sum(1 for i in range(1, len(ok_seq)) if ok_seq[i - 1] and not ok_seq[i])
    return float(id_switch + transitions)


def compute_metrics(gt_csv, pred_csv, meta, h, r, rel_az_deg, cfg,
                    subject_pos=(0.0, 0.0, 0.0), mode="fusion",
                    conf_min=0.0, subject_pos_name="center", overhead_h=None,
                    cam_c_az=None, tag_detect_csv=None):
    """
    Full results.csv row (every column from CLAUDE.md).

    gt_csv   : ground_truth_<id>.csv (Isaac Z-up world)
    pred_csv : zed_pred_<id>.csv (fusion frame) or zed_single_<id>.csv (camera frame)
    meta     : dict from the receiver's _meta.json (frames_grabbed,
               frames_with_bodies) — or {} (coverage = NaN)
    mode     : "fusion" | "single"
    overhead_h : if set, a centered overhead (nadir) cam C is part of the layout;
                 adds cam-C geometry columns (accuracy columns are camera-count
                 agnostic — they come from the fused prediction either way).
    """
    aim_h = cfg.get("aim_height_m", 1.0)
    aim_point = [subject_pos[0], subject_pos[1], subject_pos[2] + aim_h]
    cam_a_az = cfg["cam_a"]["azimuth_deg"]
    pos_a = camera_rig.camera_position(cam_a_az, r, h, subject_pos)
    pos_b = camera_rig.camera_position(cam_a_az + rel_az_deg, r, h, subject_pos)
    center = cfg.get("workspace", {}).get("center", [0.0, 0.0])
    # cam C is either a RING camera at cam_c_az (unified tag run) or a centered overhead one.
    if cam_c_az is not None:
        pos_c = camera_rig.camera_position(cam_c_az, r, h, subject_pos)
    elif overhead_h is not None:
        pos_c = camera_rig.overhead_position(overhead_h, center)
    else:
        pos_c = None
    has_cam_c = pos_c is not None

    gt_avg, n_gt_frames = load_gt_average(gt_csv, joint_filter=set(joint_map.isaac_names()))
    pred_avg_raw, n_pred_rows = load_pred_average(pred_csv, conf_min=conf_min)

    grav_deg = _gravity_alignment_deg(pred_avg_raw)
    if not math.isnan(grav_deg):
        print(f"metrics: raw hips->neck axis vs +Y = {grav_deg:.1f} deg "
              f"(≈0 => gravity-aligned frame; ≈camera tilt => camera frame)")

    if mode == "fusion":
        transform_fn = fused_to_isaac
    elif mode == "single":
        transform_fn = lambda p: single_cam_to_isaac(p, pos_a, aim_point)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Motion path (animated character) vs static path (time-averaged pose).
    gt_frames = load_gt_per_frame(gt_csv, joint_filter=set(joint_map.isaac_names()))
    motion = _has_motion(gt_frames, joint_map.isaac_names())
    if motion:
        pred_frames = load_pred_per_frame(pred_csv, conf_min=conf_min)
        offset_s = float((cfg.get("metrics") or {}).get("frame_offset_s", 0.0))
        pairs = associate_frames(gt_frames, pred_frames, offset_s=offset_s)
        # 3-cam (overhead) adds ghost bodies -> score the GT-matched real person,
        # not the most-complete skeleton. 2-cam path keeps _primary_body (select=None).
        select = (lambda gt_f, pred_f: _gt_matched_body(gt_f["joints"], pred_f, transform_fn)) \
            if has_cam_c else None
        (mpjpe, pck30, pck50, per_joint,
         mpjpe_aligned, reg_offset) = mpjpe_pck_per_frame(pairs, transform_fn, select=select)
        jitter_var = compute_jitter_variance(pairs, transform_fn, select=select)
        id_drops = compute_id_drops(pred_frames)
    else:
        pred_iso = {n: transform_fn(p) for n, p in pred_avg_raw.items()}
        mpjpe, pck30, pck50, per_joint = mpjpe_pck(gt_avg, pred_iso)
        vecs = []
        for zed_name, isaac_name in joint_map.mapped_pairs():
            if zed_name in pred_iso and isaac_name in gt_avg:
                p, g = pred_iso[zed_name], gt_avg[isaac_name]
                vecs.append([p[k] - g[k] for k in range(3)])
        mpjpe_aligned, reg_offset = _aligned_and_offset(vecs)
        jitter_var = NAN
        id_drops = NAN

    # Visibility / triangulability on the REAL averaged GT joints (replaces the
    # CANONICAL_SKELETON placeholder per the Phase 6 plan).
    gt_joints = list(gt_avg.values())
    # Ring cam C aims at the subject (chest height); overhead cam C uses the default nadir aim.
    cam_c_aim = None
    if cam_c_az is not None:
        chest_h = (cfg.get("aruco", {}) or {}).get("chest_height_m", 1.3)
        cam_c_aim = [subject_pos[0], subject_pos[1], subject_pos[2] + chest_h]
    vis = geo_prescreener.prescreen(pos_a, pos_b, gt_joints or None, cfg,
                                    subject=subject_pos, cam_c_pos=pos_c, cam_c_aim=cam_c_aim)

    frames = meta.get("frames_grabbed") or 0
    coverage = (meta.get("frames_with_bodies", 0) / frames) if frames else NAN

    row = {
        "h_m": h,
        "r_m": r,
        "rel_az_deg": rel_az_deg,
        "tilt_deg": camera_rig.tilt_angle(h, r, aim_h),
        "convergence_angle_deg": vis["convergence_angle_deg"],
        "subject_pos_name": subject_pos_name,
        "mpjpe_mm": mpjpe,                       # absolute (includes frame-registration offset)
        "mpjpe_aligned_mm": mpjpe_aligned,       # translation-aligned = true pose accuracy
        "registration_offset_mm": reg_offset,    # global fused<->GT frame offset (diagnostic)
        "pck30": pck30,
        "pck50": pck50,
        "detection_coverage": coverage,
        "joint_visibility_cam_a": vis["joints_visible_cam_a"],
        "joint_visibility_cam_b": vis["joints_visible_cam_b"],
        "joint_visibility_either": vis["joints_visible_either"],
        "joint_visibility_both": vis["joints_visible_both_triangulable"],
        "unique_contribution_cam_b": vis["unique_contribution_cam_b"],
        "jitter_variance": jitter_var,   # live when the character moves; NaN if static
        "id_drops": id_drops,            # live when the character moves; NaN if static
        # extras (not in results.csv schema, useful for debugging)
        "_motion": motion,
        "_n_gt_frames": n_gt_frames,
        "_n_pred_rows": n_pred_rows,
        "_per_joint_mm": per_joint,
        "_gravity_axis_deg": grav_deg,
    }
    if has_cam_c:
        # 3-cam extras — appended columns, blank/NaN for 2-cam rows.
        row.update({
            "cam_c_h_m": overhead_h if overhead_h is not None else NAN,
            "cam_c_az_deg": cam_c_az if cam_c_az is not None else NAN,
            "joint_visibility_cam_c": vis.get("joints_visible_cam_c", NAN),
            "unique_contribution_cam_c": vis.get("unique_contribution_cam_c", NAN),
            "convergence_ab_deg": vis.get("convergence_ab_deg", NAN),
            "convergence_ac_deg": vis.get("convergence_ac_deg", NAN),
            "convergence_bc_deg": vis.get("convergence_bc_deg", NAN),
        })

    # Unified run: fold in the ArUco tag-visibility ratio from the detection log.
    if tag_detect_csv and os.path.exists(tag_detect_csv):
        import tag_metrics
        aru = cfg.get("aruco") or {}
        tm = tag_metrics.compute(tag_detect_csv, int(aru.get("marker_id_front", 23)),
                                 int(aru.get("marker_id_back", 42)))
        if tm:
            row.update({
                "tag_visibility_ratio": tm["tag_visibility_ratio"],
                "detect_rate_cam_a": tm["detect_rate_cam_a"],
                "detect_rate_cam_b": tm["detect_rate_cam_b"],
                "detect_rate_cam_c": tm["detect_rate_cam_c"],
                "detect_rate_front": tm["detect_rate_front"],
                "detect_rate_back": tm["detect_rate_back"],
                "longest_blind_gap_s": tm["longest_blind_gap_s"],
            })
    return row


RESULTS_COLUMNS = [
    "h_m", "r_m", "rel_az_deg", "tilt_deg", "convergence_angle_deg",
    "subject_pos_name", "mpjpe_mm", "mpjpe_aligned_mm", "registration_offset_mm",
    "pck30", "pck50",
    "detection_coverage", "joint_visibility_cam_a", "joint_visibility_cam_b",
    "joint_visibility_either", "joint_visibility_both",
    "unique_contribution_cam_b", "jitter_variance", "id_drops",
    # 3-cam columns — appended; blank for 2-cam rows.
    "cam_c_h_m", "cam_c_az_deg", "joint_visibility_cam_c", "unique_contribution_cam_c",
    "convergence_ab_deg", "convergence_ac_deg", "convergence_bc_deg",
    # unified tag-detection columns — blank for runs without --detect-tags.
    "tag_visibility_ratio", "detect_rate_cam_a", "detect_rate_cam_b", "detect_rate_cam_c",
    "detect_rate_front", "detect_rate_back", "longest_blind_gap_s",
]


def append_results_row(results_csv, metrics_dict):
    """Append one row to results.csv (header written if file is new).
    Never truncates — results/ is append-only per CLAUDE.md."""
    new = not os.path.exists(results_csv)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(RESULTS_COLUMNS)
        w.writerow([metrics_dict.get(c, NAN) for c in RESULTS_COLUMNS])
