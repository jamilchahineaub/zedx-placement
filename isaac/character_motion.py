# isaac/character_motion.py
#
# Deterministic, in-place articulated motion for the UsdSkel character.
# RUNS UNDER ISAAC PYTHON ONLY (imports pxr).
#
# Why baked UsdSkel.Animation (not Articulation.set_joint_positions):
#   - The character is a UsdSkel *skinned* skeleton, not a PhysX articulation, so
#     set_joint_positions does not apply; and a robot articulation would not be
#     detected by ZED's HUMAN_BODY model.
#   - A baked animation is DETERMINISTIC and replayable, so every sweep layout
#     runs the IDENTICAL motion (fair comparison), it deforms the skinned mesh so
#     ZED sees a moving person, and it is captured automatically by gt_logger
#     (which samples ComputeJointWorldTransforms per frame).
#
# We author a UsdSkel.Animation on OUR stage's root layer and bind it to the
# referenced skeleton (overriding any animationSource opinion from test.usd —
# test.usd itself is never modified). The motion is a sum of sinusoids on a
# subset of arm/leg/torso joints, authored over a long span so it does not depend
# on timeline looping.

import math


# Joint groups by LEAF name (matched against the rig's GetJointOrder leaves).
# (leaf, local_axis, amplitude_key, phase) — amplitude_key indexes the cfg motion block.
_GROUPS = [
    ("L_Upperarm",   (1.0, 0.0, 0.0), "arm_deg",   0.0),
    ("R_Upperarm",   (1.0, 0.0, 0.0), "arm_deg",   math.pi),
    ("L_Forearm",    (1.0, 0.0, 0.0), "arm_deg",   math.pi / 2),
    ("R_Forearm",    (1.0, 0.0, 0.0), "arm_deg",   3 * math.pi / 2),
    ("L_Thigh",      (1.0, 0.0, 0.0), "leg_deg",   math.pi),
    ("R_Thigh",      (1.0, 0.0, 0.0), "leg_deg",   0.0),
    ("L_Calf",       (1.0, 0.0, 0.0), "leg_deg",   0.0),
    ("R_Calf",       (1.0, 0.0, 0.0), "leg_deg",   math.pi),
    ("Spine01",      (0.0, 0.0, 1.0), "torso_deg", 0.0),
    ("Spine02",      (0.0, 0.0, 1.0), "torso_deg", 0.0),
    ("Spine03",      (0.0, 0.0, 1.0), "torso_deg", 0.0),
    ("NeckTwist01",  (0.0, 0.0, 1.0), "torso_deg", 0.0),
    ("Neck",         (0.0, 0.0, 1.0), "torso_deg", 0.0),
]

# Within-group amplitude scale (forearm/calf swing a bit less than the parent bone).
_AMP_SCALE = {"L_Forearm": 0.8, "R_Forearm": 0.8, "L_Calf": 0.8, "R_Calf": 0.8}


def _quatf(gf, quatd):
    """Gf.Quatd -> Gf.Quatf."""
    im = quatd.GetImaginary()
    return gf.Quatf(float(quatd.GetReal()),
                    gf.Vec3f(float(im[0]), float(im[1]), float(im[2])))


def apply_inplace_animation(stage, skel_prim, cfg, cover_s=None):
    """Author + bind a deterministic limb (walk-cycle / in-place) animation to
    `skel_prim` — joint ROTATIONS only; root translation (for walk) is handled
    separately by apply_walk_path on the character prim.

    Returns the animation prim path (str) on success, or None if it was skipped.
    Sets the stage's TimeCodesPerSecond and start/end timecodes so the existing
    run_episode timeline + gt_logger sample the motion correctly.
    """
    from pxr import Usd, UsdSkel, Gf, Vt, Sdf

    mcfg = (cfg.get("motion") or {})
    period_s = float(mcfg.get("period_s", 4.0))
    fps = float(mcfg.get("fps", 30))
    if cover_s is None:
        cover_s = float(mcfg.get("cover_s", 60.0))   # author this many seconds
    amps = {"arm_deg": float(mcfg.get("arm_deg", 35.0)),
            "torso_deg": float(mcfg.get("torso_deg", 20.0)),
            "leg_deg": float(mcfg.get("leg_deg", 12.0))}

    skel = UsdSkel.Skeleton(skel_prim)
    joints_attr = skel.GetJointsAttr().Get()
    rest_attr = skel.GetRestTransformsAttr().Get()
    if not joints_attr:
        print("character_motion: skeleton has no joints attr; skipping motion")
        return None
    if not rest_attr or len(rest_attr) != len(joints_attr):
        print("character_motion: skeleton missing/!= restTransforms; skipping motion "
              f"(joints={len(joints_attr)}, rest={0 if not rest_attr else len(rest_attr)})")
        return None

    joint_tokens = [str(j) for j in joints_attr]
    leaves = [t.split("/")[-1] for t in joint_tokens]
    leaf_to_idx = {ln: i for i, ln in enumerate(leaves)}

    # Build the animated-joint set (only joints that exist in this rig).
    animated = []   # (token, rest_quatf, T(Vec3f), S(Vec3h), axis, amp_deg, phase)
    for leaf, axis, amp_key, phase in _GROUPS:
        idx = leaf_to_idx.get(leaf)
        if idx is None:
            continue
        M = rest_attr[idx]
        xf = Gf.Transform(M)
        t = xf.GetTranslation()
        s = xf.GetScale()
        rest_q = _quatf(Gf, xf.GetRotation().GetQuat())
        amp = amps[amp_key] * _AMP_SCALE.get(leaf, 1.0)
        animated.append((joint_tokens[idx], rest_q,
                         Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])),
                         Gf.Vec3h(float(s[0]), float(s[1]), float(s[2])),
                         axis, amp, phase))

    if not animated:
        print(f"character_motion: none of the target joints found in rig "
              f"(leaves sample: {leaves[:8]}); skipping motion")
        return None

    anim_path = "/World/MotionAnim"
    anim = UsdSkel.Animation.Define(stage, Sdf.Path(anim_path))
    anim.CreateJointsAttr(Vt.TokenArray([a[0] for a in animated]))
    anim.CreateTranslationsAttr().Set(Vt.Vec3fArray([a[2] for a in animated]))
    anim.CreateScalesAttr().Set(Vt.Vec3hArray([a[3] for a in animated]))
    rot_attr = anim.CreateRotationsAttr()

    n_total = int(round(cover_s * fps))
    w = 2.0 * math.pi / period_s
    for f in range(n_total + 1):
        time_s = f / fps
        quats = []
        for (_tok, rest_q, _T, _S, axis, amp, phase) in animated:
            ang = amp * math.sin(w * time_s + phase)
            dq = _quatf(Gf, Gf.Rotation(Gf.Vec3d(*axis), ang).GetQuat())
            quats.append(dq * rest_q)
        rot_attr.Set(Vt.QuatfArray(quats), Usd.TimeCode(f))

    # Bind on OUR root layer (overrides any test.usd animationSource; never edits test.usd).
    binding = UsdSkel.BindingAPI.Apply(skel_prim)
    binding.CreateAnimationSourceRel().SetTargets([Sdf.Path(anim_path)])

    # Timecodes: run_episode reads TimeCodesPerSecond and loops [start,end].
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n_total)

    print(f"character_motion: animated {len(animated)} joints "
          f"{[a[0].split('/')[-1] for a in animated]} "
          f"period={period_s}s fps={fps:g} cover={cover_s:g}s -> {anim_path}")
    return anim_path


# ---------------------------------------------------------------------------
# Walk path (deterministic serpentine root translation across the floor)
# ---------------------------------------------------------------------------

def serpentine_polyline(cfg):
    """Waypoints [(x,y), ...] of a boustrophedon sweep inside the workspace box.
    Pure function (no pxr) so it's unit-testable."""
    ws = (cfg.get("workspace") or {})
    cx, cy = (list(ws.get("center", [0.0, 0.0])) + [0.0, 0.0])[:2]
    W, H = (list(ws.get("size_m", [5.0, 5.0])) + [5.0, 5.0])[:2]
    wk = (cfg.get("walk") or {})
    m = float(wk.get("margin_m", 0.4))
    rows = max(2, int(wk.get("serpentine_rows", 6)))
    x0, x1 = cx - W / 2 + m, cx + W / 2 - m
    y0, y1 = cy - H / 2 + m, cy + H / 2 - m
    pts = []
    for i in range(rows):
        y = y0 + (y1 - y0) * (i / (rows - 1))
        pts += [(x0, y), (x1, y)] if i % 2 == 0 else [(x1, y), (x0, y)]
    return pts


def polyline_pos(pts, s):
    """Point at arc length s along the polyline, looping (s mod total). Pure."""
    segs, total = [], 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        segs.append((a, b, L, total))
        total += L
    if total <= 1e-9:
        return pts[0]
    s = s % total
    for a, b, L, acc in segs:
        if s <= acc + L:
            t = (s - acc) / L if L > 0 else 0.0
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    return pts[-1]


def _root_to_pelvis_offset(stage, character_prim, skel_prim):
    """Horizontal (x,y) offset of the skeleton's pelvis from the character root at
    rest. We drive the ROOT along the path minus this offset so the PELVIS (what
    gt_logger bins and ZED tracks) traces the box centred on the aim. (0,0) if it
    can't be measured."""
    from pxr import Usd, UsdGeom, UsdSkel
    try:
        skel = UsdSkel.Skeleton(skel_prim)
        joints = [str(j).split("/")[-1] for j in (skel.GetJointsAttr().Get() or [])]
        # Same pelvis match as gt_logger / floor_coverage (note: this rig's joint is
        # "Hip", singular — earlier this was missed and a wrong joint got picked).
        pidx = next((i for i, n in enumerate(joints)
                     if "pelvis" in n.lower() or n.lower() in ("hips", "hip")), None)
        if pidx is None:
            print("character_motion: no pelvis/hip joint -> walk path NOT pelvis-centred "
                  f"(leaves sample: {joints[:8]})")
            return 0.0, 0.0
        cache = UsdSkel.Cache()
        for prim in stage.Traverse():
            if prim.IsA(UsdSkel.Root):
                cache.Populate(UsdSkel.Root(prim), Usd.PrimDefaultPredicate)
                break
        query = cache.GetSkelQuery(skel)
        xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
        world = query.ComputeJointWorldTransforms(xfc)
        if not world or pidx >= len(world):
            return 0.0, 0.0
        pt = world[pidx].ExtractTranslation()
        rt = xfc.GetLocalToWorldTransform(character_prim).ExtractTranslation()
        offx, offy = float(pt[0] - rt[0]), float(pt[1] - rt[1])
        # Safety: an implausible offset means we grabbed the wrong joint -> don't shift
        # (driving the root directly is never worse than over-shifting it off the box).
        if (offx * offx + offy * offy) ** 0.5 > 3.0:
            print(f"character_motion: pelvis offset ({offx:.2f},{offy:.2f}) implausibly "
                  f"large via joint '{joints[pidx]}' -> NOT centring (root drives directly)")
            return 0.0, 0.0
        print(f"character_motion: pelvis joint '{joints[pidx]}' offset "
              f"({offx:.2f},{offy:.2f})")
        return offx, offy
    except Exception as e:
        print(f"character_motion: pelvis-offset measurement failed ({e}); path not centred")
        return 0.0, 0.0


def apply_walk_path(stage, character_prim, skel_prim, cfg, cover_s=None):
    """Time-sample the CHARACTER PRIM translate along the serpentine so gt_logger's
    XformCache(timecode) logs the moving root. The path is shifted by the
    root->pelvis offset so the PELVIS traces the box (centred on the aim)."""
    from pxr import Usd, UsdGeom, Gf

    mcfg = (cfg.get("motion") or {})
    wk = (cfg.get("walk") or {})
    fps = float(mcfg.get("fps", 30))
    if cover_s is None:
        cover_s = float(wk.get("cover_s", 90.0))
    speed = float(wk.get("speed_m_s", 0.8))
    pts = serpentine_polyline(cfg)
    # Round-trip (there-and-back) so looping returns to the start smoothly instead of
    # teleporting from the top of the box back to the bottom. Covers the box twice/cycle.
    pts = pts + pts[-2::-1]

    # Measure BEFORE clearing the static translate (root still at its rest position).
    offx, offy = _root_to_pelvis_offset(stage, character_prim, skel_prim)

    xform = UsdGeom.Xformable(character_prim)
    # Pure time samples (no static default) — clear any static translate authored by
    # _load_character so we don't mix a default with the time samples.
    xform.ClearXformOpOrder()
    op = xform.AddTranslateOp()
    op.GetAttr().Clear()
    n_total = int(round(cover_s * fps))
    for f in range(n_total + 1):
        x, y = polyline_pos(pts, speed * (f / fps))
        op.Set(Gf.Vec3d(float(x - offx), float(y - offy), 0.0), Usd.TimeCode(f))

    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n_total)
    print(f"character_motion: walk serpentine {len(pts)} waypoints, speed {speed} m/s, "
          f"cover {cover_s:g}s ({n_total} samples), pelvis-offset=({offx:.2f},{offy:.2f})")
    return n_total


def apply_motion(stage, character_prim, skel_prim, cfg):
    """Dispatch by cfg['character_motion']: 'walk' = limb walk-cycle + serpentine
    root translation; 'inplace' = limbs only; 'none' = static. Returns the mode."""
    mode = cfg.get("character_motion", "inplace")
    if mode == "none":
        return "none"
    # Unified cover so limbs and the walk path span the same timeline.
    cover_s = (float((cfg.get("walk") or {}).get("cover_s", 90.0)) if mode == "walk"
               else float((cfg.get("motion") or {}).get("cover_s", 60.0)))
    apply_inplace_animation(stage, skel_prim, cfg, cover_s=cover_s)
    if mode == "walk":
        apply_walk_path(stage, character_prim, skel_prim, cfg, cover_s=cover_s)
    return mode
