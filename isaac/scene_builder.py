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
    """3x3 proper rotation (list of rows) -> (w,x,y,z) quaternion for create_prim."""
    import numpy as np
    from isaacsim.core.utils.rotations import rot_matrix_to_quat
    return rot_matrix_to_quat(np.array(R, dtype=float))


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

    # add_reference_to_stage references the WHOLE test.usd at the dst prim. To pull a
    # specific source prim we add the reference then (if needed) target its default
    # prim. test.usd's biped is the animated content; referencing the file root and
    # repathing to subject is sufficient for the experiment.
    add_reference_to_stage(usd_path=ref_scene, prim_path=char_dst_prim)

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

def build_scene(h, r, rel_az, subject_name, cfg, machine_cfg):
    """
    Boot Isaac, build the room, load the character, place two ZED cameras, and start
    streaming via two ZEDAnnotators. Returns (app, stage, annotator_a, annotator_b).

    h, r        : camera height and radius (metres)
    rel_az      : cam B azimuth relative to cam A
    subject_name: name from experiment.yaml subject_positions
    cfg         : experiment.yaml dict
    machine_cfg : machine.<name>.yaml dict (headless, zed_ext_path, reference_scene)
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

    annotator_a = ZEDAnnotator(
        camera_prim=[prim_a.GetPath()],
        camera_model="ZED_X",
        streaming_port=cfg["cam_a"]["port"],          # 30000
        resolution=res,
        fps=fps,
        transport_layer_mode=transport,
        virtual_serial_number=str(cfg["cam_a"]["serial"]),   # "1001"
    )
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
