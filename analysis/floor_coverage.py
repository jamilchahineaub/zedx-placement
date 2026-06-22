#!/usr/bin/env python3
# analysis/floor_coverage.py
#
# Floor coverage / blind-spot map for one camera layout's WALK run. RUNS UNDER
# python3 (no omni/pyzed). READ-ONLY over results/; writes only to --out.
#
# Idea: during the walk we know, per frame, the GT pelvis (x,y) = where the person
# is standing (from gt_logger) and whether ZED detected a body (from the per-frame
# heartbeat zed_pred_<id>_frames.csv). Bin the floor into cells and, per cell:
#   detection_rate = frames with a body / frames the person was in that cell
#   mpjpe_mean     = mean per-frame aligned MPJPE when a body was tracked there
# -> a heatmap of where a placement keeps vs loses the skeleton, and how accurate.
#
# Reuses analysis/metrics.py (per-frame loaders, association, primary body, the
# fusion->Isaac transform) and analysis/joint_map.py.

import argparse
import bisect
import csv
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "isaac"), os.path.join(_REPO, "zed")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics       # noqa: E402
import joint_map     # noqa: E402
import camera_rig    # noqa: E402

NAN = float("nan")


# --------------------------------------------------------------------------- loaders

def load_gt_pelvis_per_frame(gt_csv):
    """Sorted list of (wall_clock, x, y) for the pelvis/hips joint, + its name."""
    pelvis_name = None
    frames = {}
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            name = row["joint_name"]
            ln = name.lower()
            if pelvis_name is None and ("pelvis" in ln or "hips" in ln or ln == "hip"):
                pelvis_name = name
            if name == pelvis_name:
                frames[float(row["wall_clock"])] = (float(row["x"]), float(row["y"]))
    out = [(w, frames[w][0], frames[w][1]) for w in sorted(frames)]
    return out, pelvis_name


def load_heartbeat(frames_csv):
    """Sorted list of (wall_clock, n_bodies) for every grabbed frame."""
    out = []
    with open(frames_csv) as f:
        for row in csv.DictReader(f):
            out.append((float(row["wall_clock"]), int(float(row["n_bodies"]))))
    out.sort()
    return out


# --------------------------------------------------------------------------- grid

def grid_dims(cfg):
    ws = cfg.get("workspace") or {}
    cx, cy = (list(ws.get("center", [0.0, 0.0])) + [0.0, 0.0])[:2]
    W, H = (list(ws.get("size_m", [5.0, 5.0])) + [5.0, 5.0])[:2]
    cell = float(cfg.get("grid_cell_m", 0.5))
    nx = max(1, int(round(W / cell)))
    ny = max(1, int(round(H / cell)))
    return {"nx": nx, "ny": ny, "cell": cell,
            "x0": cx - W / 2.0, "y0": cy - H / 2.0, "cx": cx, "cy": cy, "W": W, "H": H}


def cell_of(x, y, g):
    ix = int(math.floor((x - g["x0"]) / g["cell"]))
    iy = int(math.floor((y - g["y0"]) / g["cell"]))
    if 0 <= ix < g["nx"] and 0 <= iy < g["ny"]:
        return (ix, iy)
    return None


# --------------------------------------------------------------------------- compute

def _nearest_xy(pelvis, walls, w):
    if not pelvis:
        return None
    i = bisect.bisect_left(walls, w)
    cands = [c for c in (i - 1, i) if 0 <= c < len(pelvis)]
    k = min(cands, key=lambda k: abs(walls[k] - w))
    return (pelvis[k][1], pelvis[k][2])


def compute_floor_coverage(layout_id, cfg, results_dir, conf_min=20.0, offset_s=0.0,
                           tracked_radius_m=0.30):
    """Returns (cells, grid, info). cells[(ix,iy)] = {n, det, acc, trk, mpjpe:[...]}.
    n   = frames the person was in this cell (from heartbeat);
    det = frames with ANY body (>=1, ghosts included -> saturates at 1.0 in a 2-cam box);
    acc = frames where ZED gave a body we could measure;
    trk = of those, frames where the body's centroid sits within tracked_radius_m of the
          true pelvis (i.e. the REAL person, not a ghost) -> tracked_rate = trk/acc;
    mpjpe = aligned per-frame errors (mm) when tracked.
    offset_s shifts ZED frame wall_clocks before matching to GT (cancels render->detection
    latency so binning lands on the right cell)."""
    gt = os.path.join(results_dir, f"ground_truth_{layout_id}.csv")
    pred = os.path.join(results_dir, f"zed_pred_{layout_id}.csv")
    hb = os.path.join(results_dir, f"zed_pred_{layout_id}_frames.csv")

    pelvis, pelvis_name = load_gt_pelvis_per_frame(gt)
    walls = [p[0] for p in pelvis]
    g = grid_dims(cfg)
    cells = {}

    def cell(c):
        return cells.setdefault(c, {"n": 0, "det": 0, "acc": 0, "trk": 0, "mpjpe": []})

    info = {"pelvis_joint": pelvis_name, "gt_frames": len(pelvis),
            "heartbeat": os.path.exists(hb), "outside_box": 0}

    # 1) Detection map from the per-frame heartbeat (covers ALL frames incl. lost ones).
    if os.path.exists(hb):
        for w, nb in load_heartbeat(hb):
            pos = _nearest_xy(pelvis, walls, w + offset_s)
            if pos is None:
                continue
            c = cell_of(pos[0], pos[1], g)
            if c is None:
                info["outside_box"] += 1
                continue
            d = cell(c)
            d["n"] += 1
            d["det"] += 1 if nb > 0 else 0

    # 2) Accuracy map from per-frame aligned MPJPE, binned by pelvis position.
    gtf = metrics.load_gt_per_frame(gt, joint_filter=set(joint_map.isaac_names()))
    pf = metrics.load_pred_per_frame(pred, conf_min=conf_min)
    pairs = metrics.associate_frames(gtf, pf, offset_s=offset_s)
    tf = metrics.fused_to_isaac
    for gfr, pfr in pairs:
        b = metrics._primary_body(pfr)
        if not b:
            continue
        e = [[tf(b["joints"][z])[k] - gfr["joints"][i][k] for k in range(3)]
             for z, i in joint_map.mapped_pairs()
             if z in b["joints"] and i in gfr["joints"]]
        if not e:
            continue
        n = len(e)
        off = [sum(v[k] for v in e) / n for k in range(3)]
        off_m = math.sqrt(sum(o * o for o in off))   # centroid offset (m): real person vs ghost
        aln = 1000.0 * sum(math.sqrt(sum((v[k] - off[k]) ** 2 for k in range(3)))
                           for v in e) / n
        pos = _nearest_xy(pelvis, walls, pfr["wall"] + offset_s)
        if pos is None:
            continue
        c = cell_of(pos[0], pos[1], g)
        if c is not None:
            d = cell(c)
            d["mpjpe"].append(aln)
            d["acc"] += 1
            d["trk"] += 1 if off_m <= tracked_radius_m else 0
    return cells, g, info


# --------------------------------------------------------------------------- output

def _det_rate(d):
    return (d["det"] / d["n"]) if d["n"] else NAN


def _tracked_rate(d):
    """Fraction of frames the person was present where the detected body was actually them
    (within tracked_radius_m) — so BOTH ghost frames and lost frames count as untracked."""
    return min(1.0, d["trk"] / d["n"]) if d.get("n") else NAN


def ascii_map(cells, g):
    """Tracked rate: # >=.9, digit = rate*10, X = <.2 (ghost/lost), . = never visited.
    Top row = +y."""
    lines = []
    for iy in range(g["ny"] - 1, -1, -1):
        row = []
        for ix in range(g["nx"]):
            d = cells.get((ix, iy))
            if not d or d["n"] == 0:
                row.append(" .")
            else:
                r = _tracked_rate(d)
                row.append(" #" if r >= 0.9 else (" X" if r < 0.2 else f" {int(r * 10)}"))
        lines.append("".join(row))
    return "\n".join(lines)


def write_cell_csv(cells, g, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ix", "iy", "x_center", "y_center", "n_frames",
                    "detection_rate", "tracked_rate", "mpjpe_mean_mm"])
        for iy in range(g["ny"]):
            for ix in range(g["nx"]):
                d = cells.get((ix, iy), {"n": 0, "det": 0, "acc": 0, "trk": 0, "mpjpe": []})
                xc = g["x0"] + (ix + 0.5) * g["cell"]
                yc = g["y0"] + (iy + 0.5) * g["cell"]
                mp = (sum(d["mpjpe"]) / len(d["mpjpe"])) if d["mpjpe"] else NAN
                tr = _tracked_rate(d)
                w.writerow([ix, iy, f"{xc:.3f}", f"{yc:.3f}", d["n"],
                            f"{_det_rate(d):.4f}" if d["n"] else "nan",
                            f"{tr:.4f}" if not math.isnan(tr) else "nan",
                            f"{mp:.2f}" if not math.isnan(mp) else "nan"])


def save_png(cells, g, out_png, cams=None, layout_id="", overall=None, path=None,
             tracked_radius_m=0.30):
    """Three panels: occupancy (frames/cell), tracked rate, mean MPJPE. The middle
    panel is the ghost-aware TRACKED rate (fraction of frames where the detected body is
    the real person, within tracked_radius_m of the true pelvis) — not raw detection
    rate, which saturates at 1.0 because ghost bodies are always present. The actual GT
    walk trajectory (`path`) is overlaid so you can confirm the map matches the walk."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occ = np.full((g["ny"], g["nx"]), np.nan)
    trk = np.full((g["ny"], g["nx"]), np.nan)
    mp = np.full((g["ny"], g["nx"]), np.nan)
    for (ix, iy), d in cells.items():
        if d["n"]:
            occ[iy, ix] = d["n"]
            trk[iy, ix] = min(1.0, d["trk"] / d["n"])
        if d["mpjpe"]:
            mp[iy, ix] = sum(d["mpjpe"]) / len(d["mpjpe"])
    extent = [g["x0"], g["x0"] + g["nx"] * g["cell"],
              g["y0"], g["y0"] + g["ny"] * g["cell"]]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    t = layout_id + (f"  (overall tracked {overall:.2f})" if overall is not None else "")
    im0 = ax[0].imshow(occ, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    ax[0].set_title(f"occupancy (frames/cell) — {t}")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(trk, origin="lower", extent=extent, vmin=0, vmax=1,
                       cmap="RdYlGn", aspect="equal")
    ax[1].set_title(f"tracked rate (body <{tracked_radius_m*100:.0f}cm of GT)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(mp, origin="lower", extent=extent, cmap="RdYlGn_r", aspect="equal")
    ax[2].set_title("mean aligned MPJPE (mm)")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)
    if path:
        ax[1].plot([p[0] for p in path], [p[1] for p in path],
                   "-", color="black", lw=0.6, alpha=0.4)   # actual walk path
    for a in ax:
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")
        a.plot(g["cx"], g["cy"], "k+", ms=12, label="aim")
        if cams:
            for (cx, cy), lbl in zip(cams, ["A", "B"]):
                a.plot(cx, cy, "b^", ms=9)
                a.annotate(lbl, (cx, cy), textcoords="offset points", xytext=(4, 4))
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _parse_hra(layout_id):
    m = re.search(r"h([0-9.]+)_r([0-9.]+)_az([0-9]+)", layout_id)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None


def main():
    import yaml
    ap = argparse.ArgumentParser(description="Floor coverage / blind-spot map for a walk run")
    ap.add_argument("--layout-id", required=True)
    ap.add_argument("--results-dir", default=os.path.join(_REPO, "results", "layouts"))
    ap.add_argument("--config", default=os.path.join(_REPO, "config", "experiment.yaml"))
    ap.add_argument("--conf", type=float, default=20.0)
    ap.add_argument("--out", default=os.path.join(_REPO, "reports"))
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--rel-az", type=float, default=None)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--auto-frame", action="store_true",
                    help="frame the grid on the actual pelvis trajectory bbox (+margin) "
                         "instead of the config workspace box")
    ap.add_argument("--frame-offset", type=float, default=None,
                    help="seconds to shift ZED frames before matching GT (latency); "
                         "default = config metrics.frame_offset_s")
    ap.add_argument("--tracked-radius", type=float, default=None,
                    help="metres: a detected body counts as the real person (not a ghost) "
                         "if its centroid is within this of the true pelvis; "
                         "default = config metrics.tracked_radius_m or 0.30")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    gt = os.path.join(args.results_dir, f"ground_truth_{args.layout_id}.csv")
    if not os.path.exists(gt):
        print(f"floor_coverage: no ground truth for {args.layout_id} ({gt})")
        return

    # Always report where the pelvis actually walked (diagnoses off-center/clipped walks).
    _pel, _pn = load_gt_pelvis_per_frame(gt)
    if _pel:
        _xs = [p[1] for p in _pel]
        _ys = [p[2] for p in _pel]
        print(f"floor_coverage: pelvis '{_pn}' walked x[{min(_xs):.2f},{max(_xs):.2f}] "
              f"y[{min(_ys):.2f},{max(_ys):.2f}] over {len(_pel)} frames")

    if args.auto_frame:
        pel, _ = load_gt_pelvis_per_frame(gt)
        if pel:
            xs = [p[1] for p in pel]
            ys = [p[2] for p in pel]
            mrg = 0.5
            cfg["workspace"] = {
                "center": [(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0],
                "size_m": [max(max(xs) - min(xs), 1.0) + 2 * mrg,
                           max(max(ys) - min(ys), 1.0) + 2 * mrg]}
            print(f"floor_coverage: auto-framed grid to pelvis bbox "
                  f"center={cfg['workspace']['center']} size={cfg['workspace']['size_m']}")

    offset_s = (args.frame_offset if args.frame_offset is not None
                else float((cfg.get("metrics") or {}).get("frame_offset_s", 0.0)))
    if offset_s:
        print(f"floor_coverage: applying frame_offset_s={offset_s:+.2f}s (latency correction)")
    radius = (args.tracked_radius if args.tracked_radius is not None
              else float((cfg.get("metrics") or {}).get("tracked_radius_m", 0.30)))
    cells, g, info = compute_floor_coverage(args.layout_id, cfg, args.results_dir,
                                            args.conf, offset_s=offset_s,
                                            tracked_radius_m=radius)
    tot_n = sum(d["n"] for d in cells.values())
    tot_det = sum(d["det"] for d in cells.values())
    tot_acc = sum(d["acc"] for d in cells.values())
    tot_trk = sum(d["trk"] for d in cells.values())
    det_cov = (tot_det / tot_n) if tot_n else NAN
    overall = (tot_trk / tot_n) if tot_n else NAN   # ghost+lost-aware tracked coverage

    print(f"floor_coverage: {args.layout_id}  pelvis={info['pelvis_joint']}  "
          f"grid={g['nx']}x{g['ny']}@{g['cell']}m  frames_in_box={tot_n}  "
          f"detection={det_cov:.3f}  tracked(<{radius*100:.0f}cm)={overall:.3f}"
          + ("" if info["heartbeat"] else "  [WARN no heartbeat -> detection map empty]"))
    # CAPTURED region (cells the ZED actually recorded) vs the walked bbox printed above.
    # If CAPTURED is much smaller than walked, the ZED capture window was too short (in
    # sim-time) to see a full walk -> increase --capture-duration or speed up the walk.
    cov_cells = [(ix, iy) for (ix, iy), d in cells.items() if d["n"] > 0]
    if cov_cells:
        cxs = [g["x0"] + (ix + 0.5) * g["cell"] for ix, iy in cov_cells]
        cys = [g["y0"] + (iy + 0.5) * g["cell"] for ix, iy in cov_cells]
        print(f"floor_coverage: CAPTURED region x[{min(cxs):.2f},{max(cxs):.2f}] "
              f"y[{min(cys):.2f},{max(cys):.2f}] ({len(cov_cells)} cells)")
    print(ascii_map(cells, g))

    csv_out = os.path.join(args.out, f"floor_{args.layout_id}.csv")
    write_cell_csv(cells, g, csv_out)
    print(f"floor_coverage: wrote {csv_out}")

    if not args.no_png:
        hra = (args.h, args.r, args.rel_az) if args.h else _parse_hra(args.layout_id)
        cams = None
        if hra and all(v is not None for v in hra):
            h, r, az = hra
            a0 = cfg["cam_a"]["azimuth_deg"]
            cams = [tuple(camera_rig.camera_position(a0, r, h)[:2]),
                    tuple(camera_rig.camera_position(a0 + az, r, h)[:2])]
        pelvis, _ = load_gt_pelvis_per_frame(gt)
        path = [(p[1], p[2]) for p in pelvis]
        png = os.path.join(args.out, f"floor_{args.layout_id}.png")
        save_png(cells, g, png, cams=cams, layout_id=args.layout_id,
                 overall=overall, path=path, tracked_radius_m=radius)
        print(f"floor_coverage: wrote {png}")


if __name__ == "__main__":
    main()
