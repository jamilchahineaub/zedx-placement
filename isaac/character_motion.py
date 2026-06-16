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


def apply_inplace_animation(stage, skel_prim, cfg):
    """Author + bind a deterministic in-place animation to `skel_prim`.

    Returns the animation prim path (str) on success, or None if it was skipped.
    Sets the stage's TimeCodesPerSecond and start/end timecodes so the existing
    run_episode timeline + gt_logger sample the motion correctly.
    """
    from pxr import Usd, UsdSkel, Gf, Vt, Sdf

    mcfg = (cfg.get("motion") or {})
    period_s = float(mcfg.get("period_s", 4.0))
    fps = float(mcfg.get("fps", 30))
    cover_s = float(mcfg.get("cover_s", 60.0))   # author this many seconds (no loop dependency)
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
