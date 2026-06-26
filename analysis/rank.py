#!/usr/bin/env python3
# analysis/rank.py  (v2)
#
# Ranks camera positions from results/results.csv. RUNS UNDER python3 (no omni,
# no pyzed). READ-ONLY over results.csv; only ever writes to an explicit --out
# (never into results/).
#
# A "camera position" is (h_m, r_m, rel_az_deg). Rows are grouped by it and
# aggregated across subject positions, then each layout is scored on THREE axes:
#   pose_fidelity      = aligned MPJPE  (true pose error, registration-invariant)
#   absolute_placement = absolute MPJPE (world position; includes the fusion offset)
#   stability          = id_drops + jitter (temporal tracking reliability)
# These come straight from config/experiment.yaml -> ranking (weights/bands/norms).
#
# Design notes (tuned to the full warehouse+motion sweep):
#   - Layouts are FLAGGED valid/invalid (registration offset / jitter / coverage),
#     but NONE are excluded — every layout is ranked and the flag is shown.
#   - pose_fidelity and absolute_placement are SEPARATE Pareto axes, so a layout
#     with great pose but a failed world-registration still surfaces on the
#     frontier (and is flagged). That tradeoff is the point.
#   - Heavy-tailed metrics (jitter) use RANK normalization (outlier-robust); the
#     MPJPE family uses interpretable clamped bands. Pareto uses the axis goodness
#     (monotonic), so it's robust to the exact normalization.

import argparse
import csv
import math
import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

NAN = float("nan")

# Columns aggregated across subject positions by (NaN-aware) mean.
_MEAN_COLS = [
    "mpjpe_aligned_mm", "registration_offset_mm",
    "pck30", "pck50", "detection_coverage",
    "joint_visibility_cam_a", "joint_visibility_cam_b",
    "joint_visibility_either", "joint_visibility_both",
    "unique_contribution_cam_b", "jitter_variance", "id_drops",
    "convergence_angle_deg", "tilt_deg",
]


# --------------------------------------------------------------------------- helpers

def _isnan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return NAN


def _nanmean(vals):
    xs = [v for v in vals if not _isnan(v)]
    return sum(xs) / len(xs) if xs else NAN


def _nanmax(vals):
    xs = [v for v in vals if not _isnan(v)]
    return max(xs) if xs else NAN


def _nanmin(vals):
    xs = [v for v in vals if not _isnan(v)]
    return min(xs) if xs else NAN


def _corr(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if not _isnan(x) and not _isnan(y)]
    n = len(pairs)
    if n < 3:
        return NAN
    ax = [p[0] for p in pairs]
    bx = [p[1] for p in pairs]
    ma, mb = sum(ax) / n, sum(bx) / n
    cov = sum((ax[i] - ma) * (bx[i] - mb) for i in range(n)) / n
    sa = (sum((x - ma) ** 2 for x in ax) / n) ** 0.5
    sb = (sum((y - mb) ** 2 for y in bx) / n) ** 0.5
    return cov / (sa * sb) if sa * sb else NAN


# --------------------------------------------------------------------------- load + aggregate

def load_results(path):
    """results.csv -> list of dicts (floats except subject_pos_name)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (v if k == "subject_pos_name" else _to_float(v))
                         for k, v in row.items()})
    return rows


def aggregate_by_layout(rows, subject=None):
    """Group by (h_m, r_m, rel_az_deg) — plus cam_c_h_m when present, so each
    overhead height ranks as its own 3-cam layout. Aggregate across subjects
    (NaN-aware mean; plus mpjpe_worst and coverage_min for worst-case checks)."""
    if subject is not None:
        rows = [r for r in rows if r.get("subject_pos_name") == subject]
    # Treat the overhead camera height as part of the layout identity only if the
    # results file actually carries it (2-cam CSVs have no cam_c_h_m -> behaves as before).
    has_cam_c = any(not _isnan(r.get("cam_c_h_m", NAN)) for r in rows)
    groups = {}
    for r in rows:
        key = (round(r["h_m"], 6), round(r["r_m"], 6), round(r["rel_az_deg"], 6))
        if has_cam_c:
            cc = r.get("cam_c_h_m", NAN)
            key = key + (round(cc, 6) if not _isnan(cc) else None,)
        groups.setdefault(key, []).append(r)
    layouts = []
    for key, grp in groups.items():
        agg = {"h_m": key[0], "r_m": key[1], "rel_az_deg": key[2], "n_subjects": len(grp)}
        if has_cam_c:
            agg["cam_c_h_m"] = key[3]
        agg["mpjpe_mm"] = _nanmean([g.get("mpjpe_mm", NAN) for g in grp])
        agg["mpjpe_worst"] = _nanmax([g.get("mpjpe_mm", NAN) for g in grp])
        agg["coverage_min"] = _nanmin([g.get("detection_coverage", NAN) for g in grp])
        for c in _MEAN_COLS:
            agg[c] = _nanmean([g.get(c, NAN) for g in grp])
        layouts.append(agg)
    layouts.sort(key=lambda a: (a["h_m"], a["r_m"], a["rel_az_deg"]))
    return layouts


# --------------------------------------------------------------------------- per-metric goodness

def _band(v, lo, hi):
    """Clamped linear band: v<=lo -> 1.0, v>=hi -> 0.0 (lower is better)."""
    if _isnan(v):
        return None
    span = (hi - lo) or 1.0
    return max(0.0, min(1.0, (hi - v) / span))


def _rank_goodness(raw, lower_better=True):
    """Within-sweep rank in [0,1] (best->1, worst->0). Outlier-robust: only the
    ORDER matters, so one huge value can't flatten the scale."""
    present = [v for v in raw if not _isnan(v)]
    m = len(present)
    out = []
    for v in raw:
        if _isnan(v):
            out.append(None)
        elif m <= 1:
            out.append(1.0)
        else:
            worse = (sum(1 for u in present if u > v) if lower_better
                     else sum(1 for u in present if u < v))
            out.append(worse / (m - 1))
    return out


def metric_goodness(raw_values, norm, cfg):
    """raw metric values -> [0,1] goodness per layout, per the `norm` method."""
    rk = cfg["ranking"]
    if norm == "pose_band":
        lo, hi = (float(x) for x in rk["pose_band_mm"])
        return [_band(v, lo, hi) for v in raw_values]
    if norm == "absolute_band":
        lo, hi = (float(x) for x in rk["absolute_band_mm"])
        return [_band(v, lo, hi) for v in raw_values]
    if norm == "rank":
        return _rank_goodness(raw_values, lower_better=True)
    if norm == "asis":
        return [None if _isnan(v) else max(0.0, min(1.0, v)) for v in raw_values]
    raise ValueError(f"unknown norm {norm!r}")


def axis_scores(layouts, cfg):
    """Return (per_layout_axes, axis_names, notes).
    per_layout_axes[i] = {axis_name: score in [0,1] or None}."""
    axes = cfg["ranking"]["axes"]
    axis_names = list(axes.keys())
    metric_goods, notes = {}, {}
    for adef in axes.values():
        for m, mdef in adef["metrics"].items():
            raw = [lay.get(m, NAN) for lay in layouts]
            present = [v for v in raw if not _isnan(v)]
            if not present:
                metric_goods[m] = [None] * len(layouts)
                notes[m] = "dropped (no data this run)"
                continue
            metric_goods[m] = metric_goodness(raw, mdef["norm"], cfg)
            if max(present) - min(present) < 1e-12:
                notes[m] = "present but constant (no differentiation)"

    per_layout = [{} for _ in layouts]
    for aname, adef in axes.items():
        for i in range(len(layouts)):
            num = den = 0.0
            for m, mdef in adef["metrics"].items():
                g = metric_goods[m][i]
                if g is None:
                    continue
                w = float(mdef["w"])
                num += w * g
                den += w
            per_layout[i][aname] = (num / den) if den > 0 else None
    return per_layout, axis_names, notes


# --------------------------------------------------------------------------- validity / composite / pareto

def validity(layouts, cfg):
    """Per layout (valid_bool, reason). Flags but does NOT exclude."""
    v = cfg["ranking"]["validity"]
    moff = float(v["max_registration_offset_mm"])
    mjit = float(v["max_jitter_variance"])
    cflo = float(v["coverage_floor"])
    out = []
    for lay in layouts:
        reasons = []
        off = lay.get("registration_offset_mm", NAN)
        if not _isnan(off) and off > moff:
            reasons.append(f"offset>{moff:g}")
        jit = lay.get("jitter_variance", NAN)
        if not _isnan(jit) and jit > mjit:
            reasons.append(f"jitter>{mjit:g}")
        cov = lay.get("coverage_min", lay.get("detection_coverage", NAN))
        if not _isnan(cov) and cov < cflo:
            reasons.append(f"coverage<{cflo:g}")
        out.append((not reasons, ",".join(reasons) if reasons else "ok"))
    return out


def composite(per_layout_axes, axis_weights):
    """Weighted sum of axis scores, renormalized over axes present per layout."""
    scores = []
    for axes in per_layout_axes:
        num = den = 0.0
        for aname, w in axis_weights.items():
            a = axes.get(aname)
            if a is None:
                continue
            num += w * a
            den += w
        scores.append((num / den) if den > 0 else NAN)
    return scores


def pareto_front(per_layout_axes, axis_names, eps=1e-9):
    """Flag layouts not dominated on ANY axis (weight-free), over axes defined for
    all layouts. Returns (flags, used_axes)."""
    used = [a for a in axis_names
            if all(ax.get(a) is not None for ax in per_layout_axes)]
    n = len(per_layout_axes)
    flags = [True] * n
    if not used:
        return flags, used
    for i in range(n):
        bi = per_layout_axes[i]
        for j in range(n):
            if i == j:
                continue
            aj = per_layout_axes[j]
            if (all(aj[a] >= bi[a] - eps for a in used)
                    and any(aj[a] > bi[a] + eps for a in used)):
                flags[i] = False
                break
    return flags, used


def rank(layouts, cfg, axis_weights):
    """Score, validity-flag, Pareto-flag, sort by composite desc. Returns
    (ranked, notes, pareto_axes)."""
    per_axis, axis_names, notes = axis_scores(layouts, cfg)
    scores = composite(per_axis, axis_weights)
    flags, pareto_axes = pareto_front(per_axis, axis_names)
    val = validity(layouts, cfg)
    ranked = []
    for i, lay in enumerate(layouts):
        ranked.append({**lay, "_axes": per_axis[i], "score": scores[i],
                       "pareto": flags[i], "valid": val[i][0],
                       "flag_reason": val[i][1]})
    ranked.sort(key=lambda r: (-1.0 if _isnan(r["score"]) else -r["score"]))
    for k, r in enumerate(ranked, 1):
        r["rank"] = k
    return ranked, notes, pareto_axes


# --------------------------------------------------------------------------- weights / presets

def default_axis_weights(cfg):
    return {n: float(a["weight"]) for n, a in cfg["ranking"]["axes"].items()}


def preset_weights(cfg, preset):
    names = list(cfg["ranking"]["axes"].keys())
    return {n: float(v) for n, v in zip(names, cfg["ranking"]["presets"][preset])}


def parse_weights(s, cfg):
    cw = default_axis_weights(cfg)
    for part in s.split(","):
        part = part.strip()
        if part:
            k, _, v = part.partition("=")
            cw[k.strip()] = float(v)
    return cw


# --------------------------------------------------------------------------- rendering

def _fmt(x):
    if _isnan(x):
        return "nan"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def layout_label(lay):
    base = f"h{_fmt(lay['h_m'])} r{_fmt(lay['r_m'])} az{int(round(lay['rel_az_deg']))}"
    cc = lay.get("cam_c_h_m")
    if cc is not None and not _isnan(cc):
        base += f" oh{_fmt(cc)}"
    return base


def _jit_std(r):
    j = r.get("jitter_variance")
    return _fmt(math.sqrt(j)) if not _isnan(j) else "nan"


def format_table(ranked, top_n):
    hdr = (f"{'#':>3} {'layout':<15} {'V':>1} {'score':>6}  "
           f"{'pose':>5} {'abs':>5} {'stab':>5}  "
           f"{'aligned':>7} {'absMPJPE':>9} {'offset':>7} {'idDrp':>5} {'jitStd':>6}  P")
    lines = [hdr, "-" * len(hdr)]
    for r in ranked[:top_n]:
        a = r["_axes"]
        def cs(name):
            v = a.get(name)
            return "  -  " if v is None else f"{v:5.3f}"
        lines.append(
            f"{r['rank']:>3} {layout_label(r):<15} {'Y' if r['valid'] else 'x':>1} "
            f"{r['score']:6.3f}  {cs('pose_fidelity')} {cs('absolute_placement')} "
            f"{cs('stability')}  {_fmt(r.get('mpjpe_aligned_mm')):>7} "
            f"{_fmt(r.get('mpjpe_mm')):>9} {_fmt(r.get('registration_offset_mm')):>7} "
            f"{_fmt(r.get('id_drops')):>5} {_jit_std(r):>6}  {'*' if r['pareto'] else ' '}")
    return "\n".join(lines)


def mounting_candidates(ranked, n=3):
    """Top n layouts that are BOTH valid and Pareto-optimal (fall back to valid)."""
    cands = [r for r in ranked if r["valid"] and r["pareto"]]
    if len(cands) < n:
        cands += [r for r in ranked if r["valid"] and r not in cands]
    return cands[:n]


def mounting_lines(ranked, n=3):
    out = []
    cands = mounting_candidates(ranked, n)
    if not cands:
        return ["  (no valid layouts — every layout was flagged; check the sweep / thresholds)"]
    for idx, r in enumerate(cands, 1):
        h, rr, az = r["h_m"], r["r_m"], int(round(r["rel_az_deg"]))
        cc = r.get("cam_c_h_m")
        oh = f", overhead cam at {cc:g} m" if (cc is not None and not _isnan(cc)) else ""
        out.append(f"  {idx}. h={h:g}m, r={rr:g}m, az={az}°{oh}   "
                   f"[composite {r['score']:.3f} | pose {_fmt(r.get('mpjpe_aligned_mm'))}mm | "
                   f"abs {_fmt(r.get('mpjpe_mm'))}mm | offset {_fmt(r.get('registration_offset_mm'))}mm]")
        mount = (f"     → mount both cameras at {h:g} m height, {rr:g} m from the subject, "
                 f"separated by {az}° around the subject")
        if cc is not None and not _isnan(cc):
            mount += f", plus an overhead camera at {cc:g} m looking straight down"
        out.append(mount)
    return out


def insight_line(layouts):
    aln = [l.get("mpjpe_aligned_mm", NAN) for l in layouts]
    rt = _corr(aln, [l.get("tilt_deg", NAN) for l in layouts])
    rh = _corr(aln, [l.get("h_m", NAN) for l in layouts])
    rr = _corr(aln, [l.get("r_m", NAN) for l in layouts])
    msg = []
    if not _isnan(rt) and rt > 0.2:
        msg.append("lower cameras track better (less tilt)")
    if not _isnan(rr) and abs(rr) < 0.2:
        msg.append("radius ~irrelevant to accuracy")
    tail = ("; " + "; ".join(msg)) if msg else ""
    return (f"Insight: aligned-MPJPE correlates tilt r={rt:+.2f}, height r={rh:+.2f}, "
            f"radius r={rr:+.2f}{tail}.")


def render(ranked, notes, pareto_axes, cfg, args):
    out = []
    n_valid = sum(1 for r in ranked if r["valid"])
    out.append(f"Ranked {len(ranked)} camera position(s) "
               f"(grouped by h/r/az{', subject=' + args.subject if args.subject else ''}); "
               f"{n_valid} valid, {len(ranked) - n_valid} flagged.")
    out.append("composite = " + " + ".join(f"{w:g}*{n}" for n, w in _active_weights.items())
               + "   (V=valid, P=Pareto-optimal)")
    out.append("")
    out.append(format_table(ranked, args.top))
    out.append("")

    # Pareto frontier members.
    pareto = [r for r in ranked if r["pareto"]]
    if pareto_axes:
        out.append(f"Pareto frontier over [{', '.join(pareto_axes)}] "
                   f"({len(pareto)} layouts):")
        for r in pareto:
            flag = "" if r["valid"] else f"  [flagged: {r['flag_reason']}]"
            out.append(f"  {layout_label(r):<15} valid={r['valid']!s:<5} "
                       f"pose={_fmt(r.get('mpjpe_aligned_mm'))}mm "
                       f"abs={_fmt(r.get('mpjpe_mm'))}mm{flag}")
    else:
        out.append("Pareto: not enough common axes to compute.")
    out.append("")

    # Preset sensitivity.
    out.append("Preset sensitivity (top-3 by composite):")
    tops = {}
    for pname in cfg["ranking"]["presets"]:
        r2, _, _ = rank([dict(l) for l in ranked], cfg, preset_weights(cfg, pname))
        tops[pname] = [layout_label(x) for x in r2[:3]]
        out.append(f"  {pname:<10} {'  >  '.join(tops[pname])}")
    winners = {v[0] for v in tops.values()}
    out.append(f"  => #1 is {'STABLE' if len(winners) == 1 else 'WEIGHT-SENSITIVE'} "
               f"across presets ({', '.join(sorted(winners))}).")
    out.append("")

    # The actionable output.
    out.append(insight_line(ranked))
    out.append("")
    out.append("Physical mounting candidates (top valid + Pareto-optimal):")
    out.extend(mounting_lines(ranked, 3))

    drop_notes = {m: n for m, n in notes.items() if "constant" in n or "dropped" in n}
    if drop_notes:
        out.append("")
        out.append("Metric notes (report-only columns / no signal this run):")
        for m, n in drop_notes.items():
            out.append(f"  {m}: {n}")
    return "\n".join(out)


def write_csv(path, ranked):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    axis_cols = ["pose_fidelity", "absolute_placement", "stability"]
    # Include cam_c_h_m only if any ranked layout carries it (3-cam results).
    cam_c_cols = ["cam_c_h_m"] if any(
        r.get("cam_c_h_m") is not None and not _isnan(r.get("cam_c_h_m")) for r in ranked) else []
    cols = (["rank", "h_m", "r_m", "rel_az_deg"] + cam_c_cols
            + ["score", "valid", "flag_reason", "pareto"]
            + axis_cols
            + ["mpjpe_aligned_mm", "mpjpe_mm", "registration_offset_mm",
               "id_drops", "jitter_variance", "joint_visibility_both",
               "convergence_angle_deg", "tilt_deg", "n_subjects"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in ranked:
            a = r["_axes"]
            w.writerow([a.get(c) if c in axis_cols else r.get(c) for c in cols])


_active_weights = {}


def main():
    ap = argparse.ArgumentParser(description="Rank camera positions from results.csv (v2)")
    ap.add_argument("--results", default=os.path.join(_REPO, "results", "results.csv"))
    ap.add_argument("--config", default=os.path.join(_REPO, "config", "experiment.yaml"))
    ap.add_argument("--preset", default="balanced",
                    help="axis-weight preset (experiment.yaml ranking.presets)")
    ap.add_argument("--weights", default=None,
                    help="override, e.g. pose_fidelity=0.6,absolute_placement=0.1,stability=0.3")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--subject", default=None,
                    help="rank within one subject_pos_name instead of aggregating")
    ap.add_argument("--out", default=None,
                    help="optional CSV output path (suggest reports/; never results/)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rk = cfg["ranking"]

    if not os.path.exists(args.results):
        print(f"rank: results file not found: {args.results}")
        return
    rows = load_results(args.results)
    if not rows:
        print("rank: results.csv has no rows.")
        return

    layouts = aggregate_by_layout(rows, subject=args.subject)
    if args.weights:
        cw = parse_weights(args.weights, cfg)
    elif args.preset in rk["presets"]:
        cw = preset_weights(cfg, args.preset)
    else:
        cw = default_axis_weights(cfg)

    global _active_weights
    _active_weights = cw

    ranked, notes, pareto_axes = rank(layouts, cfg, cw)
    print(render(ranked, notes, pareto_axes, cfg, args))

    if args.out:
        write_csv(args.out, ranked)
        print(f"\nrank: wrote {len(ranked)} rows to {args.out}")


if __name__ == "__main__":
    main()
