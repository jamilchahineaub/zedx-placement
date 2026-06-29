#!/usr/bin/env python3
# isaac/view_tag_scene.py
#
# INTERACTIVE viewport of one 3-camera layout with TWO ArUco tags (chest + back) on the
# character. RUNS UNDER ISAAC PYTHON ONLY (launched by scripts/view_tag.py).
#
# The character SPINS as it walks; the tags are pinned to the body each frame from the SKELETON
# (chest position + shoulder-perpendicular facing), so they ride flat on the chest/back and
# sweep through every facing — letting you eyeball whether the three cameras keep a tag in view.
# Visual sanity check only — no ZED streaming, no detection. Tag geometry is shared with the
# real episode via isaac/tag_utils.py. Close the window (or Ctrl-C) to exit.

import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import tag_utils as tu  # noqa: E402  (pxr imported lazily inside)


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def add_spin(stage, character_prim_path, spin_deg_s):
    """Append a time-sampled yaw so the character turns in place as it walks."""
    from pxr import UsdGeom, Usd
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(character_prim_path))
    rot = xf.AddRotateZOp()
    tcps = stage.GetTimeCodesPerSecond() or 30.0
    end = int(stage.GetEndTimeCode() or (tcps * 10))
    for f in range(0, end + 1):
        rot.Set(spin_deg_s * (f / tcps), Usd.TimeCode(f))
    print(f"view_tag: spinning character at {spin_deg_s:.0f}°/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--rel-az", type=float, required=True)
    ap.add_argument("--cam-c-az", type=float, required=True)
    ap.add_argument("--subject-name", default="center")
    ap.add_argument("--machine", default="4090")
    ap.add_argument("--marker-front", required=True, help="ArUco PNG for the chest (front) tag")
    ap.add_argument("--marker-back", required=True, help="ArUco PNG for the back tag")
    ap.add_argument("--tag-size", type=float, default=0.30)
    ap.add_argument("--tag-offset", type=float, default=0.30,
                    help="metres to push each tag off the chest joint to clear the torso")
    ap.add_argument("--spin-deg-s", type=float, default=30.0)
    ap.add_argument("--walk-speed", type=float, default=None,
                    help="override walk speed (m/s) for the viewer only (config is 1.5)")
    args = ap.parse_args()

    cfg = load_yaml(os.path.join(REPO, "config", "experiment.yaml"))
    if args.walk_speed is not None:
        cfg.setdefault("walk", {})["speed_m_s"] = args.walk_speed
        print(f"view_tag: walk speed overridden to {args.walk_speed} m/s (viewer only)", flush=True)
    machine_cfg = load_yaml(os.path.join(REPO, "config", f"machine.{args.machine}.yaml"))
    machine_cfg["headless"] = False

    # 1) Boot Isaac (windowed) + enable the ZED extension (for the ZED_X camera USD).
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": False})
    import omni.kit.app
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.add_path(machine_cfg["zed_ext_path"])
    mgr.set_extension_enabled_immediate("sl.sensor.camera", True)
    for _ in range(5):
        app.update()

    # 2) Stage (Z-up, metres) — mirror build_scene's setup.
    import omni.usd
    from pxr import UsdGeom, UsdPhysics, Usd, Gf
    from isaacsim.core.utils.prims import create_prim
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    create_prim("/World", "Xform")
    create_prim("/World/Looks", "Scope")
    create_prim("/World/Room", "Xform")
    create_prim("/World/Lights", "Xform")
    create_prim("/World/PhysicsScene", "PhysicsScene")
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").CreateGravityMagnitudeAttr(0.0)

    # 3) Scene + character + motion (reuse scene_builder helpers — no streaming).
    import scene_builder as sb
    import camera_rig
    import character_motion
    subject_pos = sb._subject_pos(args.subject_name, cfg)
    if cfg.get("scene_type") == "warehouse":
        sb._load_warehouse(stage, cfg, machine_cfg)
    else:
        sb._build_room(stage, cfg)
    sb._load_character(stage, cfg, machine_cfg, subject_pos)
    char_path = cfg.get("character_prim", "/World/biped_demo_meters")
    if cfg.get("character_motion", "inplace") != "none":
        skel = sb._find_skeleton_prim(stage)
        char = stage.GetPrimAtPath(char_path)
        if skel is not None:
            character_motion.apply_motion(stage, char, skel, cfg)
    if args.spin_deg_s:
        add_spin(stage, char_path, args.spin_deg_s)

    # 4) Tag prims + a skeleton query (reuse gt_logger) to pin them each frame.
    from gt_logger import GTLogger
    gl = GTLogger(stage)
    li, ri, ci = tu.find_joints(gl.joint_names)
    print(f"view_tag: chest joints -> L={li and gl.joint_names[li]} "
          f"R={ri and gl.joint_names[ri]} chest={ci and gl.joint_names[ci]}", flush=True)
    tags = (tu.create_tag_quad(stage, args.marker_front, args.tag_size, "Front"),
            tu.create_tag_quad(stage, args.marker_back, args.tag_size, "Back"))

    # 5) Three cameras: A,B on the ring (aim at hip), C on the ring at cam-c-az (aim at chest).
    aru = cfg.get("aruco") or {}
    chest_h = float(aru.get("chest_height_m", 1.30))
    aim_h = cfg.get("aim_height_m", 1.0)
    aim_ab = [subject_pos[0], subject_pos[1], subject_pos[2] + aim_h]
    aim_c = [subject_pos[0], subject_pos[1], subject_pos[2] + chest_h]
    zed_usd = sb._zed_x_usd_path(machine_cfg)
    a0 = cfg["cam_a"]["azimuth_deg"]
    for name, azc, aimpt in [("A", a0, aim_ab), ("B", a0 + args.rel_az, aim_ab),
                             ("C", args.cam_c_az, aim_c)]:
        pos = camera_rig.camera_position(azc, args.r, args.h, subject_pos)
        R = camera_rig.rotation_matrix_from_look_at(pos, aimpt)
        sb._place_camera(stage, f"/World/ZED_Camera_{name}", zed_usd, pos, R)
    app.update()

    # 6) Play + pin both tags to the body each frame until the window closes.
    import omni.timeline
    tl = omni.timeline.get_timeline_interface()
    tl.set_looping(True)
    tl.play()
    tcps = stage.GetTimeCodesPerSecond() or 30.0
    print(f"view_tag: VIEWPORT READY — h{args.h}/r{args.r}/az{int(args.rel_az)} + cam C @ "
          f"{int(args.cam_c_az)}°, spin {args.spin_deg_s:.0f}°/s. Close the window to exit.",
          flush=True)
    can_pin = None not in (li, ri, ci)
    if not can_pin:
        print("view_tag: WARNING could not find shoulder/chest joints — tags sit at origin", flush=True)
    while app.is_running():
        app.update()
        if can_pin:
            tu.pin_two_tags(gl.skel_query, li, ri, ci, tags, args.tag_offset,
                            tl.get_current_time() * tcps)
    app.close()


if __name__ == "__main__":
    main()
