# analysis/tag_visibility.py
#
# Geometric visibility of a directional chest ArUco tag — plain python3, no omni/pyzed.
#
# A chest tag is NOT like a body joint: it only DECODES when a camera is in front of it,
# within the decode-angle cone of the tag's outward normal, and close enough that the marker
# spans enough pixels. So a camera "sees" the tag iff ALL of:
#   1. tag is in the camera FOV cone + range and in front of it  (reuse geo_prescreener)
#   2. the tag is close enough to decode                          (distance <= max_range_m)
#   3. the camera lies within the decode cone of the tag normal   (angle(normal, tag->cam) <= decode_max_deg)
#
# The robot can face ANY direction (incl. rotating in place), so facing is a free variable:
# we sweep it 0..360 at every floor cell and ask whether AT LEAST ONE camera decodes the tag.
#
# Coordinate system: right-handed Z-up, metres (same as camera_rig / geo_prescreener).
# Reuses isaac/camera_rig.py and analysis/geo_prescreener.py + the grid from floor_coverage.py.

import math

import camera_rig as cr
import geo_prescreener as gp
import floor_coverage as fc


# ---------------------------------------------------------------------------
# Tag geometry
# ---------------------------------------------------------------------------

def facing_normal(theta_deg):
    """Horizontal outward unit normal of the chest tag for a robot facing theta (deg)."""
    t = math.radians(theta_deg)
    return [math.cos(t), math.sin(t), 0.0]


def make_cam(azimuth_deg, radius, height, aim):
    """A camera on the ring at (azimuth, radius, height), aimed at `aim`.
    Returns (pos, R) where R rows are [right, up, forward] (from rotation_matrix_from_look_at)."""
    pos = cr.camera_position(azimuth_deg, radius, height)
    R = cr.rotation_matrix_from_look_at(pos, aim)
    return (pos, R)


def tag_visible(cam_pos, R_cam, tag_pos, tag_normal, cfg_zed, decode_max_deg, max_range_m):
    """True if this camera can DECODE the tag (FOV+range+front, decode-range, decode-cone)."""
    # 1) FOV cone + sensor range + in-front-of-camera (reuse the joint test verbatim).
    if not gp._joint_in_view(cam_pos, R_cam, tag_pos, cfg_zed):
        return False
    # 2) decode-range gate: a far tag is too few pixels to read (stricter than sensor range).
    dx = cam_pos[0] - tag_pos[0]
    dy = cam_pos[1] - tag_pos[1]
    dz = cam_pos[2] - tag_pos[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist > max_range_m or dist < 1e-9:
        return False
    # 3) decode cone: angle between the tag's outward normal and the tag->camera ray.
    #    >90 deg => camera is behind the tag (can't see the face); decode_max_deg (~60) is stricter.
    vdot = (dx * tag_normal[0] + dy * tag_normal[1] + dz * tag_normal[2]) / dist
    vdot = max(-1.0, min(1.0, vdot))
    return math.degrees(math.acos(vdot)) <= decode_max_deg


# ---------------------------------------------------------------------------
# Coverage over (floor cell x facing)
# ---------------------------------------------------------------------------

def cell_coverage(cams, cell_xy, chest_h, facings, cfg_zed, decode_max_deg, max_range_m,
                  two_sided=False):
    """For one floor cell, return (fraction_of_facings_covered, covered_bools).
    A facing is covered if ANY camera decodes the tag; with two_sided=True the robot wears
    tags on BOTH the chest and the back, so a facing counts if a camera decodes the front
    normal OR the opposite (back) normal."""
    tag_pos = [cell_xy[0], cell_xy[1], chest_h]
    bools = []
    for theta in facings:
        n = facing_normal(theta)
        normals = [n, [-n[0], -n[1], 0.0]] if two_sided else [n]
        bools.append(any(
            tag_visible(cpos, R, tag_pos, nm, cfg_zed, decode_max_deg, max_range_m)
            for nm in normals for (cpos, R) in cams))
    frac = sum(1 for b in bools if b) / len(bools)
    return frac, bools


def largest_blind_wedge(bools, step_deg):
    """Largest contiguous arc (deg) of UNCOVERED facings, with wraparound."""
    n = len(bools)
    if all(bools):
        return 0.0
    if not any(bools):
        return 360.0
    best = cur = 0
    for b in bools * 2:                 # double for wraparound
        cur = 0 if b else cur + 1
        best = max(best, cur)
    return min(best, n) * step_deg


def aruco_params(cfg):
    """Pull the ArUco/coverage params from cfg with sensible defaults."""
    a = cfg.get("aruco") or {}
    return {
        "chest_height_m": float(a.get("chest_height_m", 1.30)),
        "decode_max_deg": float(a.get("decode_max_deg", 60.0)),
        "max_decode_range_m": float(a.get("max_decode_range_m", 6.0)),
        "facing_step_deg": float(a.get("facing_step_deg", 15)),
        "cam_c_azimuth_step_deg": float(a.get("cam_c_azimuth_step_deg", 15)),
    }


def coverage_map(cams, cfg, params=None, two_sided=False):
    """Per-cell angular coverage over the workspace grid.
    Returns dict: cells{(ix,iy)->frac}, grid, worst (min cell coverage), mean,
    blind_wedge_deg (largest blind facing arc over all cells), n_cells.
    two_sided=True models chest+back tags (a facing is covered if a camera decodes either face)."""
    if params is None:
        params = aruco_params(cfg)
    g = fc.grid_dims(cfg)
    cfg_zed = cfg["zed_x"]
    step = params["facing_step_deg"]
    facings = [i * step for i in range(int(round(360.0 / step)))]
    cells = {}
    worst = 1.0
    total = 0.0
    wedge = 0.0
    for iy in range(g["ny"]):
        for ix in range(g["nx"]):
            xc = g["x0"] + (ix + 0.5) * g["cell"]
            yc = g["y0"] + (iy + 0.5) * g["cell"]
            frac, bools = cell_coverage(cams, (xc, yc), params["chest_height_m"], facings,
                                        cfg_zed, params["decode_max_deg"],
                                        params["max_decode_range_m"], two_sided=two_sided)
            cells[(ix, iy)] = frac
            worst = min(worst, frac)
            total += frac
            wedge = max(wedge, largest_blind_wedge(bools, step))
    n = g["nx"] * g["ny"]
    return {"cells": cells, "grid": g, "worst": worst, "mean": total / n,
            "blind_wedge_deg": wedge, "n_cells": n}


def _circ_dist(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def best_cam_c_azimuths(h, r, rel_az, cfg, n=4, min_sep_deg=45.0, params=None):
    """Rank candidate RING cam-C azimuths for a (h, r, rel_az) layout by TWO-SIDED tag
    coverage, EXCLUDING angles within min_sep_deg of cam A or cam B (which would make cam C
    redundant with a ring camera — bad for both tags and pose triangulation). cam A/B aim at
    hip; cam C aims at chest, matching the real rig. Returns [(az, worst, mean), ...] best first."""
    if params is None:
        params = aruco_params(cfg)
    aim_h = cfg.get("aim_height_m", 1.0)
    chest = params["chest_height_m"]
    cx, cy = ((cfg.get("workspace") or {}).get("center", [0.0, 0.0]) + [0.0, 0.0])[:2]
    a0 = cfg["cam_a"]["azimuth_deg"]
    A = make_cam(a0, r, h, [cx, cy, aim_h])
    B = make_cam(a0 + rel_az, r, h, [cx, cy, aim_h])
    step = params["cam_c_azimuth_step_deg"]
    cands = []
    for i in range(int(round(360.0 / step))):
        azc = i * step
        if _circ_dist(azc, a0) < min_sep_deg or _circ_dist(azc, a0 + rel_az) < min_sep_deg:
            continue
        C = make_cam(azc, r, h, [cx, cy, chest])
        cm = coverage_map([A, B, C], cfg, params, two_sided=True)
        cands.append((cm["worst"], cm["mean"], azc))
    cands.sort(reverse=True)                       # by worst-cell coverage, then mean
    return [(az, w, m) for (w, m, az) in cands[:n]]


# ---------------------------------------------------------------------------
# Output (single-panel coverage heatmap; mirrors floor_coverage.save_png style)
# ---------------------------------------------------------------------------

def save_coverage_png(cmap, out_png, cams_xy=None, title=""):
    """Heatmap of per-cell angular coverage (0..1) with camera positions overlaid."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = cmap["grid"]
    arr = np.full((g["ny"], g["nx"]), np.nan)
    for (ix, iy), v in cmap["cells"].items():
        arr[iy, ix] = v
    extent = [g["x0"], g["x0"] + g["nx"] * g["cell"], g["y0"], g["y0"] + g["ny"] * g["cell"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, origin="lower", extent=extent, vmin=0, vmax=1,
                   cmap="RdYlGn", aspect="equal")
    ax.set_title(title or f"tag angular coverage (worst {cmap['worst']:.2f}, "
                          f"mean {cmap['mean']:.2f}, blind wedge {cmap['blind_wedge_deg']:.0f}°)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of facings a camera can decode")
    ax.plot(g["cx"], g["cy"], "k+", ms=12)
    if cams_xy:
        for (cx, cy), lbl in cams_xy:
            ax.plot(cx, cy, "b^", ms=9)
            ax.annotate(lbl, (cx, cy), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
