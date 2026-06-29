# isaac/tag_utils.py
#
# Shared helpers for placing + pinning ArUco tag quads on the character's chest/back.
# Used by BOTH the interactive viewer (isaac/view_tag_scene.py) and the real episode
# (isaac/run_episode.py). RUNS UNDER ISAAC PYTHON (pxr imported lazily inside functions).
#
# A tag is a flat textured quad in canonical local frame (centred at origin, in the local
# Y-Z plane, outward normal = local +X). Its world pose is set every frame from the skeleton:
# centred on the chest joint, normal = horizontal perpendicular to the shoulder line, pushed
# `offset` m out to clear the torso. The two body faces (chest / back) are at +/- perp.

import math


def quat_from_cols(forward, left, up):
    """(w,x,y,z) for the rotation whose columns (images of local +X,+Y,+Z) are
    forward,left,up. Maps the tag's local +X (normal) -> forward, +Z -> up."""
    m00, m01, m02 = forward[0], left[0], up[0]
    m10, m11, m12 = forward[1], left[1], up[1]
    m20, m21, m22 = forward[2], left[2], up[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * S, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / S, 0.25 * S, (m01 + m10) / S, (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / S, (m01 + m10) / S, 0.25 * S, (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, 0.25 * S
    return w, x, y, z


def create_tag_quad(stage, marker_png, tag_size, name):
    """Create /World/Tag_<name> — a flat ArUco quad (local Y-Z plane, outward normal local +X)
    with an emissive marker texture so it reads regardless of lighting. Its world pose is set
    each frame by place_tag. Returns (translate_op, orient_op)."""
    from pxr import UsdGeom, UsdShade, Gf, Sdf, Vt
    half = tag_size / 2.0
    pts = [Gf.Vec3f(0, -half, -half), Gf.Vec3f(0, half, -half),
           Gf.Vec3f(0, half, half), Gf.Vec3f(0, -half, half)]
    path = "/World/Tag_" + name
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))

    mat = UsdShade.Material.Define(stage, path + "/Mat")
    surf = UsdShade.Shader.Define(stage, path + "/Mat/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, path + "/Mat/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, path + "/Mat/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(marker_png)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    surf.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(mat)

    xf = UsdGeom.Xformable(mesh)
    xf.ClearXformOpOrder()
    top = xf.AddTranslateOp()
    rop = xf.AddOrientOp()
    return top, rop


def find_joints(names):
    """Indices of (left shoulder, right shoulder, chest) from the skeleton joint names."""
    def f(pred):
        for i, n in enumerate(names):
            if pred(n.lower()):
                return i
        return None
    sh = lambda ln: ("upperarm" in ln or "shoulder" in ln or "clavicle" in ln)
    li = f(lambda ln: sh(ln) and (ln.startswith("l_") or "left" in ln))
    ri = f(lambda ln: sh(ln) and (ln.startswith("r_") or "right" in ln))
    spine = sorted((n.lower(), i) for i, n in enumerate(names) if "spine" in n.lower())
    ci = spine[-1][1] if spine else f(lambda ln: "chest" in ln or "neck" in ln)
    return li, ri, ci


def chest_pose(W, li, ri, ci):
    """From world joint transforms: (chest_center, perp_horizontal_unit, up). `perp` is the
    horizontal direction perpendicular to the shoulder line — the two body faces (chest/back)
    are at +/- perp."""
    from pxr import Gf
    L, R, C = (W[li].ExtractTranslation(), W[ri].ExtractTranslation(), W[ci].ExtractTranslation())
    rx, ry = float(R[0] - L[0]), float(R[1] - L[1])
    rn = math.hypot(rx, ry) or 1.0
    rx, ry = rx / rn, ry / rn
    perp = Gf.Vec3d(ry, -rx, 0.0)
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    center = Gf.Vec3d(float(C[0]), float(C[1]), float(C[2]))
    return center, perp, up


def place_tag(top, rop, center, fwd, up, offset):
    """Set a tag's translate+orient: centred on `center`, pushed `offset` along `fwd`, facing fwd."""
    from pxr import Gf
    left = Gf.Vec3d(up[1] * fwd[2] - up[2] * fwd[1],
                    up[2] * fwd[0] - up[0] * fwd[2],
                    up[0] * fwd[1] - up[1] * fwd[0])      # up x fwd
    pos = Gf.Vec3d(center[0] + fwd[0] * offset, center[1] + fwd[1] * offset, center[2])
    top.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    w, x, y, z = quat_from_cols(fwd, left, up)
    rop.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))


def pin_two_tags(skel_query, li, ri, ci, tags, offset, timecode):
    """Pin the two opposite-face tags from the skeleton at `timecode`. tags = (front_ops, back_ops),
    each (translate_op, orient_op). Returns True if pinned."""
    from pxr import UsdGeom, Usd, Gf
    W = skel_query.ComputeJointWorldTransforms(UsdGeom.XformCache(Usd.TimeCode(timecode)))
    if not W:
        return False
    center, perp, up = chest_pose(W, li, ri, ci)
    (top_f, rop_f), (top_b, rop_b) = tags
    place_tag(top_f, rop_f, center, perp, up, offset)
    place_tag(top_b, rop_b, center, Gf.Vec3d(-perp[0], -perp[1], 0.0), up, offset)
    return True
