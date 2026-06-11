# isaac/scene_builder.py
#
# Builds the experiment scene under Isaac Sim 5.1 and starts ZED streaming.
# RUNS UNDER ISAAC PYTHON ONLY:  /home/jimmy/isaacsim/python.sh
#
# Public entry point:
#   build_scene(h, r, rel_az, subject_name, cfg, machine_cfg)
#       -> (app, stage, annotator_a, annotator_b)
# run_episode.py (Phase 5) drives this.
#
# Coordinate system: Isaac side is RIGHT_HANDED_Z_UP, metres (see CLAUDE.md).
# The Z-up -> Y-up conversion lives ONLY in zed/make_fusion_config.py, not here.
#
# Streaming uses ZEDAnnotator (NOT a hand-built ActionGraph). One annotator per
# camera; it builds its own OGN graph internally. Verified signature in
# /home/jimmy/zed-isaac-sim/exts/sl.sensor.camera/sl/sensor/camera/annotators.py.

import math
import os
import sys

# camera_rig lives next to this file (isaac/). Pure math, no omni imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_rig  # noqa: E402


# Shipped UV-grid texture so no surface is untextured (ZED depth needs texture to
# stereo-match). Confirmed present in the Isaac install.
_CHECKER_TEXTURE = (
    "/home/jimmy/isaacsim/extscache/"
    "omni.kit.asset_converter-5.0.17+107.3.1.lx64.r.cp311.u353/"
    "data/references/textures/ov_uv_grids_basecolor_1024.png"
)


# ---------------------------------------------------------------------------
# Small helpers (Isaac-runtime; imported lazily after SimulationApp exists)
# ---------------------------------------------------------------------------

def _subject_pos(subject_name, cfg):
    for s in cfg["subject_positions"]:
        if s["name"] == subject_name:
            return list(s["pos"])
    raise ValueError(
        f"subject '{subject_name}' not in experiment.yaml subject_positions "
        f"({[s['name'] for s in cfg['subject_positions']]})"
    )


def _rotmat_to_wxyz(R):
    """
    Convert camera_rig look-at matrix (rows: right, up, forward) to Isaac wxyz quaternion.

    Isaac default camera looks down +X. To aim it toward an arbitrary target:
      1. yaw   around world-Z to face the horizontal direction of forward
      2. pitch around local-Y to tilt up/down

    We extract yaw and pitch from the forward vector directly and build the
    quaternion as Rz(yaw) * Ry(pitch) — no intermediate rot_matrix_to_quat needed.
    """
    import math
    forward = R[2]  # (fx, fy, fz) — unit vector toward target
    fx, fy, fz = forward

    yaw = math.atan2(fy, fx)  # horizontal angle from +X

    # Ry(pitch) applied to local +X gives z-component = -sin(pitch), so to get
    # forward.z == fz we need pitch = asin(-fz). (positive pitch = tilt down)
    pitch = math.asin(max(-1.0, min(1.0, -fz)))

    # Quaternion for Rz(yaw) * Ry(pitch) — intrinsic ZY rotations
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)

    # Rz then Ry: q = qz * qy, where qz=(cy,0,0,sy), qy=(cp,0,sp,0) in (w,x,y,z)
    qw = cy*cp
    qx = -sy*sp
    qy = cy*sp
    qz = sy*cp
    return (qw, qx, qy, qz)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def _make_checker_material(prim_path, name, scale=(4.0, 4.0)):
    import numpy as np
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    return OmniPBR(
        prim_path=prim_path,
        name=name,
        texture_path=_CHECKER_TEXTURE,
        texture_scale=np.array(scale, dtype=float),
    )


def _bind_material(stage, prim_path, material):
    """Bind a VisualMaterial to a prim via UsdShade."""
    from pxr import UsdShade
    prim = stage.GetPrimAtPath(prim_path)
    mat_prim = stage.GetPrimAtPath(material.prim_path)
    UsdShade.MaterialBindingAPI(prim).Bind(
        UsdShade.Material(mat_prim), UsdShade.Tokens.strongerThanDescendants
    )


def _build_room(stage, cfg):
    """Floor + 4 walls + 2 dome lights + table + pillar, all textured."""
    import numpy as np
    from pxr import UsdGeom, UsdLux, Gf
    from isaacsim.core.utils.prims import create_prim

    size = cfg.get("room_size_m", [6.0, 6.0, 6.0])
    sx, sy, sz = size
    hx, hy = sx / 2.0, sy / 2.0

    floor_mat = _make_checker_material("/World/Looks/FloorMat", "floor_mat", (6, 6))
    wall_mat = _make_checker_material("/World/Looks/WallMat", "wall_mat", (6, 6))

    # Floor (thin cuboid at z=0)
    create_prim("/World/Room/Floor", "Cube",
                translation=[0, 0, -0.01], scale=[hx, hy, 0.01])
    _bind_material(stage, "/World/Room/Floor", floor_mat)

    # 4 walls (thin cuboids). z spans 0..sz, centred at sz/2.
    wall_specs = [
        ("WallNorth", [0,  hy, sz/2], [hx, 0.05, sz/2]),
        ("WallSouth", [0, -hy, sz/2], [hx, 0.05, sz/2]),
        ("WallEast",  [hx, 0,  sz/2], [0.05, hy, sz/2]),
        ("WallWest",  [-hx, 0, sz/2], [0.05, hy, sz/2]),
    ]
    for name, t, s in wall_specs:
        p = f"/World/Room/{name}"
        create_prim(p, "Cube", translation=t, scale=s)
        _bind_material(stage, p, wall_mat)

    # 2 dome lights, intensity 1500
    for i, name in enumerate(["DomeLight0", "DomeLight1"]):
        lp = f"/World/Lights/{name}"
        create_prim(lp, "DomeLight", attributes={"inputs:intensity": 1500.0})

    # Occluders (togglable via experiment.yaml occluders section)
    occ = cfg.get("occluders", {"table": True, "pillar": True})
    if occ.get("table", True):
        # table: 1 x 2 x 0.8 m box, sitting on floor, offset from subject
        create_prim("/World/Room/Table", "Cube",
                    translation=[1.2, 0.0, 0.4], scale=[0.5, 1.0, 0.4])
        _bind_material(stage, "/World/Room/Table", wall_mat)
    if occ.get("pillar", True):
        # pillar: 0.4 m diameter, 2 m tall cylinder
        create_prim("/World/Room/Pillar", "Cylinder",
                    translation=[-1.2, 1.0, 1.0],
                    attributes={"radius": 0.2, "height": 2.0})
        _bind_material(stage, "/World/Room/Pillar", wall_mat)


def _load_character(stage, cfg, machine_cfg, subject_pos):
    """
    Reference /World/biped_demo_meters from the reference scene (test.usd) into the
    new stage. test.usd is USDC binary; Isaac resolves its nested character reference
    automatically. The reference-scene path comes from config, never hardcoded.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage

    ref_scene = machine_cfg.get("reference_scene") or cfg.get("reference_scene")
    char_src_prim = cfg.get("character_prim", "/World/biped_demo_meters")
    char_dst_prim = "/World/biped_demo_meters"

    if not ref_scene or not os.path.exists(ref_scene):
        print(f"RUN_FAILED reference_scene not found: {ref_scene!r} — "
              f"add 'reference_scene' to machine config. Using capsule placeholder.")
        from isaacsim.core.utils.prims import create_prim
        create_prim(char_dst_prim, "Capsule",
                    translation=[subject_pos[0], subject_pos[1], subject_pos[2] + 0.9],
                    attributes={"radius": 0.25, "height": 1.3})
        return char_dst_prim

    # add_reference_to_stage references the WHOLE test.usd at the dst prim, which
    # drags in everything authored in it (the character, plus whatever cameras /
    # ActionGraphs / environment geometry the scene was last saved with — the
    # user re-saves test.usd from their own sessions, so the content drifts).
    # Those are referenced opinions, so RemovePrim() won't work on the composed
    # stage. Instead: KEEP only the subtree(s) that contain a skeleton (the
    # character) and deactivate every other child on the session layer (session
    # opinions are strongest; the reference file is never modified).
    #
    # This is critical for streaming: a leftover ActionGraph with ZED helper
    # nodes inside test.usd would bind ports 30000/30002 and fight our two
    # ZEDAnnotators ("Error during zed streamer initialization" loop).
    add_reference_to_stage(usd_path=ref_scene, prim_path=char_dst_prim)

    from pxr import Usd as _Usd, UsdSkel as _UsdSkel
    root = stage.GetPrimAtPath(char_dst_prim)

    def _has_skeleton(prim):
        for p in _Usd.PrimRange(prim):
            if p.IsA(_UsdSkel.Root) or p.IsA(_UsdSkel.Skeleton):
                return True
        return False

    kept, dropped = [], []
    session_layer = stage.GetSessionLayer()
    for child in root.GetChildren():
        if _has_skeleton(child):
            kept.append(str(child.GetPath()))
        else:
            with _Usd.EditContext(stage, session_layer):
                child.SetActive(False)
            dropped.append(str(child.GetPath()))

    print(f"scene_builder: kept {kept}, deactivated {len(dropped)} test.usd prim(s): "
          f"{[p.split('/')[-1] for p in dropped]}")
    if not kept:
        print("RUN_FAILED no skeleton subtree under reference — "
              f"check {ref_scene} still contains the character")

    # Place at the subject floor position.
    from pxr import UsdGeom, Gf
    prim = stage.GetPrimAtPath(char_dst_prim)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in subject_pos]))
    return char_dst_prim


def _place_camera(stage, prim_path, usd_path, pos, R):
    """Reference a ZED_X.usdc at prim_path with world pos + orientation (Z-up)."""
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom, Gf

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    quat_wxyz = _rotmat_to_wxyz(R)  # (w,x,y,z)

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    w, x, y, z = [float(v) for v in quat_wxyz]
    xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return prim


def _zed_x_usd_path(machine_cfg):
    ext = machine_cfg["zed_ext_path"]
    return os.path.join(ext, "sl.sensor.camera", "data", "usd", "ZED_X.usdc")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_scene(h, r, rel_az, subject_name, cfg, machine_cfg, cams="both"):
    """
    Boot Isaac, build the room, load the character, place two ZED cameras, and start
    streaming via two ZEDAnnotators. Returns (app, stage, annotator_a, annotator_b).

    h, r        : camera height and radius (metres)
    rel_az      : cam B azimuth relative to cam A
    subject_name: name from experiment.yaml subject_positions
    cfg         : experiment.yaml dict
    machine_cfg : machine.<name>.yaml dict (headless, zed_ext_path, reference_scene)
    cams        : "both" | "a" | "b" — which annotators to start (diagnostic;
                  the skipped one is returned as None). Camera prims are always
                  placed so the scene geometry is identical either way.
    """
    # 1) SimulationApp FIRST, before any omni/isaacsim imports.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": bool(machine_cfg.get("headless", False))})

    # 2) Enable the ZED extension from the configured ext path.
    import omni.kit.app
    ext_path = machine_cfg["zed_ext_path"]
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.add_path(ext_path)
    mgr.set_extension_enabled_immediate("sl.sensor.camera", True)
    # Let the extension finish loading.
    for _ in range(5):
        app.update()

    # 3) New stage, Z-up, metres, /World.
    import omni.usd
    from pxr import UsdGeom, Usd
    from isaacsim.core.utils.prims import create_prim

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    create_prim("/World", "Xform")
    create_prim("/World/Looks", "Scope")
    create_prim("/World/Room", "Xform")
    create_prim("/World/Lights", "Xform")

    # Disable gravity so cameras and character don't fall when timeline plays.
    from pxr import UsdPhysics, Gf as _Gf
    scene_prim = create_prim("/World/PhysicsScene", "PhysicsScene")
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityMagnitudeAttr(0.0)

    subject_pos = _subject_pos(subject_name, cfg)

    # 4) Room.
    _build_room(stage, cfg)

    # 5) Character.
    _load_character(stage, cfg, machine_cfg, subject_pos)

    # 6) Two ZED_X cameras at ring positions, aimed at hip height (Z-up).
    aim_h = cfg.get("aim_height_m", 1.0)
    aim_point = [subject_pos[0], subject_pos[1], subject_pos[2] + aim_h]
    cam_a_az = cfg["cam_a"]["azimuth_deg"]
    zed_usd = _zed_x_usd_path(machine_cfg)

    pos_a = camera_rig.camera_position(cam_a_az, r, h, subject_pos)
    R_a = camera_rig.rotation_matrix_from_look_at(pos_a, aim_point)
    _place_camera(stage, "/World/ZED_Camera_A", zed_usd, pos_a, R_a)

    pos_b = camera_rig.camera_position(cam_a_az + rel_az, r, h, subject_pos)
    R_b = camera_rig.rotation_matrix_from_look_at(pos_b, aim_point)
    _place_camera(stage, "/World/ZED_Camera_B", zed_usd, pos_b, R_b)

    # 7) Render one frame so the camera render products exist before annotators.
    app.update()

    # 8) Two ZEDAnnotators. camera_prim is a LIST of Sdf.Path (uses .pathString).
    #    Serials 1001/1002 preserve the serial<->port invariant on the SDK side.
    from sl.sensor.camera.annotators import ZEDAnnotator
    zs = cfg.get("zed_stream", {})
    res = zs.get("resolution", "HD1080")
    fps = zs.get("fps", 30)
    transport = zs.get("transport", "BOTH")

    prim_a = stage.GetPrimAtPath("/World/ZED_Camera_A")
    prim_b = stage.GetPrimAtPath("/World/ZED_Camera_B")

    annotator_a = None
    annotator_b = None
    if cams in ("both", "a"):
        annotator_a = ZEDAnnotator(
            camera_prim=[prim_a.GetPath()],
            camera_model="ZED_X",
            streaming_port=cfg["cam_a"]["port"],          # 30000
            resolution=res,
            fps=fps,
            transport_layer_mode=transport,
            virtual_serial_number=str(cfg["cam_a"]["serial"]),   # "1001"
        )
    if cams in ("both", "b"):
        annotator_b = ZEDAnnotator(
            camera_prim=[prim_b.GetPath()],
            camera_model="ZED_X",
            streaming_port=cfg["cam_b"]["port"],          # 30002
            resolution=res,
            fps=fps,
            transport_layer_mode=transport,
            virtual_serial_number=str(cfg["cam_b"]["serial"]),   # "1002"
        )

    # 9) Sentinel for sweep.py.
    print("STREAMING_STARTED", flush=True)
    return app, stage, annotator_a, annotator_b
