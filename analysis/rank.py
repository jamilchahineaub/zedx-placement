#!/usr/bin/env python3
# analysis/rank.py
#
# Ranks camera positions from results/results.csv. RUNS UNDER python3 (no omni,
# no pyzed). READ-ONLY over results.csv; only ever writes to an explicit --out
# (never into results/).
#
# A "camera position" is (h_m, r_m, rel_az_deg). Rows are grouped by it and
# aggregated across subject positions (so we rank the layout, not layout x
# subject). Each metric is mapped to a 0-1 "goodness", grouped into three
# categories (accuracy / reliability / geometry), and combined into a composite.
#
# Why this design (see README "Ranking"):
#   - Most metrics are already absolute 0-1 (pck, coverage, visibility,
#     unique_contribution) -> used as-is, NOT min-max normalized (that would
#     turn a 0.98->1.0 gap into 0->1). Only mpjpe needs a reference scale (the
#     experiment's 20-200mm acceptance band). jitter/id_drops have no absolute
#     scale -> within-sweep normalization, and are NaN (drop out) until motion.
#   - The composite weights are a judgment call, so the ranking is made robust to
#     them: a weight-free Pareto frontier over the 3 categories flags layouts that
#     are not dominated on ANY category, and preset sensitivity shows whether the
#     top picks move when the weights change.
#
# All weights / band / floor / presets live in config/experiment.yaml -> ranking:.

import argparse
import csv
import math
import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

NAN = float("nan")

LAYOUT_KEYS = ("h_m", "r_m", "rel_az_deg")

# Metrics whose smaller value is better.
LOWER_BETTER = {"mpjpe_mm", "jitter_variance", "id_drops"}
# Lower-better metrics with no natural absolute scale -> within-sweep normalize.
RELATIVE_LOWER = {"jitter_variance", "id_drops"}
# Columns aggregated across subject positions by (NaN-aware) mean.
_MEAN_COLS = [
    "pck30", "pck50", "detection_coverage",
    "joint_visibility_cam_a", "joint_visibility_cam_b",
    "joint_visibility_either", "joint_visibility_both",
    "unique_contribution_cam_b", "jitter_variance", "id_drops",
    "convergence_angle_deg", "tilt_deg",
]


# ---------------------------------------------------------------------------
# Small NaN-aware helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Load + aggregate
# ---------------------------------------------------------------------------

def load_results(path):
    """results.csv -> list of dicts. Every column float except subject_pos_name
    (NaN where unparseable / 'nan' / empty)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d = {}
            for k, v in row.items():
                d[k] = v if k == "subject_pos_name" else _to_float(v)
            rows.append(d)
    return rows


def aggregate_by_layout(rows, subject=None):
    """Group rows by (h_m, r_m, rel_az_deg) and aggregate across subjects.

    If `subject` is given, restrict to that subject_pos_name first (so each
    layout has a single row). Returns a list of layout dicts; mpjpe carries both
    mean (for scoring) and worst (max); coverage carries mean (score) and min
    (the gate uses worst-case)."""
    if subject is not None:
        rows = [r for r in rows if r.get("subject_pos_name") == subject]

    groups = {}
    for r in rows:
        key = (round(r["h_m"], 6), round(r["r_m"], 6), round(r["rel_az_deg"], 6))
        groups.setdefault(key, []).append(r)

    layouts = []
    for key, grp in groups.items():
        agg = {"h_m": key[0], "r_m": key[1], "rel_az_deg": key[2],
               "n_subjects": len(grp)}
        agg["mpjpe_mm"] = _nanmean([g.get("mpjpe_mm", NAN) for g in grp])
        agg["mpjpe_worst"] = _nanmax([g.get("mpjpe_mm", NAN) for g in grp])
        agg["coverage_min"] = _nanmin([g.get("detection_coverage", NAN) for g in grp])
        for c in _MEAN_COLS:
            agg[c] = _nanmean([g.get(c, NAN) for g in grp])
        layouts.append(agg)

    layouts.sort(key=lambda a: (a["h_m"], a["r_m"], a["rel_az_deg"]))
    return layouts


# ---------------------------------------------------------------------------
# Per-metric goodness in [0, 1]
# ---------------------------------------------------------------------------

def goodness_columns(layouts, cfg):
    """For every metric named in ranking.categories, return:
        goods : {metric: [goodness or None per layout]}
        notes : {metric: human note} for metrics that dropped or didn't vary.
    None = no data for that layout (NaN); a metric all-None has dropped out."""
    rk = cfg["ranking"]
    good_ref = float(rk.get("mpjpe_good_mm", 20.0))
    bad_ref = float(rk.get("mpjpe_bad_mm", 200.0))

    metric_names = []
    for cat in rk["categories"].values():
        metric_names.extend(cat["metrics"].keys())

    goods, notes = {}, {}
    for m in metric_names:
        raw = [lay.get(m, NAN) for lay in layouts]
        present = [v for v in raw if not _isnan(v)]
        if not present:
            goods[m] = [None] * len(layouts)
            notes[m] = "dropped (no data this run)"
            continue

        if m == "mpjpe_mm":
            span = (bad_ref - good_ref) or 1.0
            goods[m] = [None if _isnan(v)
                        else max(0.0, min(1.0, (bad_ref - v) / span)) for v in raw]
        elif m in RELATIVE_LOWER:
            lo, hi = min(present), max(present)
            if hi - lo < 1e-12:
                goods[m] = [None if _isnan(v) else 1.0 for v in raw]
                notes[m] = "present but constant (no differentiation)"
            else:
                goods[m] = [None if _isnan(v) else (hi - v) / (hi - lo) for v in raw]
        else:  # already absolute 0-1, higher is better
            goods[m] = [None if _isnan(v) else max(0.0, min(1.0, v)) for v in raw]
            if max(present) - min(present) < 1e-12:
                notes[m] = "present but constant (no differentiation)"

    return goods, notes


def category_scores(layouts, cfg):
    """Return (per_layout_cats, cat_names, notes).
    per_layout_cats[i] = {category_name: score in [0,1] or None}."""
    rk = cfg["ranking"]
    goods, notes = goodness_columns(layouts, cfg)
    cat_names = list(rk["categories"].keys())
    per_layout = [{} for _ in layouts]

    for cname, cdef in rk["categories"].items():
        mweights = cdef["metrics"]
        for i in range(len(layouts)):
            num = den = 0.0
            for m, w in mweights.items():
                g = goods.get(m, [None] * len(layouts))[i]
                if g is None:
                    continue
                num += w * g
                den += w
            per_layout[i][cname] = (num / den) if den > 0 else None

    return per_layout, cat_names, notes


# ---------------------------------------------------------------------------
# Composite + Pareto
# ---------------------------------------------------------------------------

def composite(per_layout_cats, category_weights):
    """Weighted sum of category scores, renormalized over the categories that
    are present (non-None) for each layout."""
    scores = []
    for cats in per_layout_cats:
        num = den = 0.0
        for cname, w in category_weights.items():
            c = cats.get(cname)
            if c is None:
                continue
            num += w * c
            den += w
        scores.append((num / den) if den > 0 else NAN)
    return scores


def pareto_front(per_layout_cats, cat_names, eps=1e-9):
    """Flag layouts not dominated on ANY category (weight-free). Dominance is
    computed over categories defined for ALL layouts. Returns (flags, used_cats)."""
    used = [c for c in cat_names
            if all(cats.get(c) is not None for cats in per_layout_cats)]
    n = len(per_layout_cats)
    flags = [True] * n
    if not used:
        return flags, used
    for i in range(n):
        bi = per_layout_cats[i]
        for j in range(n):
            if i == j:
                continue
            aj = per_layout_cats[j]
            ge_all = all(aj[c] >= bi[c] - eps for c in used)
            gt_any = any(aj[c] > bi[c] + eps for c in used)
            if ge_all and gt_any:
                flags[i] = False  # i is dominated by j
                break
    return flags, used


# ---------------------------------------------------------------------------
# Gate + rank
# ---------------------------------------------------------------------------

def gate(layouts, coverage_floor):
    """Drop layouts with NaN mpjpe or worst-case coverage below the floor."""
    kept, dropped = [], []
    for lay in layouts:
        if _isnan(lay.get("mpjpe_mm")):
            dropped.append((lay, "mpjpe NaN"))
            continue
        cov = lay.get("coverage_min", NAN)
        if not _isnan(cov) and cov < coverage_floor:
            dropped.append((lay, f"coverage {cov:.2f} < {coverage_floor}"))
            continue
        kept.append(lay)
    return kept, dropped


def rank(layouts, cfg, category_weights):
    """Score, Pareto-flag, and sort layouts by composite (desc). Returns
    (ranked, notes, pareto_cats)."""
    per_cats, cat_names, notes = category_scores(layouts, cfg)
    scores = composite(per_cats, category_weights)
    flags, pareto_cats = pareto_front(per_cats, cat_names)

    ranked = []
    for i, lay in enumerate(layouts):
        ranked.append({**lay, "_cats": per_cats[i],
                       "score": scores[i], "pareto": flags[i]})
    ranked.sort(key=lambda r: (-1.0 if _isnan(r["score"]) else -r["score"]))
    for k, r in enumerate(ranked, 1):
        r["rank"] = k
    return ranked, notes, pareto_cats


# ---------------------------------------------------------------------------
# Weights / presets
# ---------------------------------------------------------------------------

def default_category_weights(cfg):
    return {n: float(c["weight"]) for n, c in cfg["ranking"]["categories"].items()}


def preset_weights(cfg, preset):
    names = list(cfg["ranking"]["categories"].keys())
    vals = cfg["ranking"]["presets"][preset]
    return {n: float(v) for n, v in zip(names, vals)}


def parse_weights(s, cfg):
    """'accuracy=0.5,reliability=0.3,geometry=0.2' -> dict (missing -> default)."""
    cw = default_category_weights(cfg)
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        cw[k.strip()] = float(v)
    return cw


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(x):
    if _isnan(x):
        return "nan"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def layout_label(lay):
    return f"h{_fmt(lay['h_m'])} r{_fmt(lay['r_m'])} az{int(round(lay['rel_az_deg']))}"


def format_table(ranked, top_n):
    hdr = (f"{'#':>2}  {'layout':<16} {'score':>6}  {'acc':>5} {'rel':>5} "
           f"{'geo':>5}  {'mpjpe':>7} {'visB':>5} {'cov':>5} {'uniqB':>5}  P")
    lines = [hdr, "-" * len(hdr)]
    for r in ranked[:top_n]:
        c = r["_cats"]
        def cs(name):
            v = c.get(name)
            return "  -  " if v is None else f"{v:5.3f}"
        lines.append(
            f"{r['rank']:>2}  {layout_label(r):<16} {r['score']:6.3f}  "
            f"{cs('accuracy')} {cs('reliability')} {cs('geometry')}  "
            f"{_fmt(r['mpjpe_mm']):>7} "
            f"{_fmt(r.get('joint_visibility_both')):>5} "
            f"{_fmt(r.get('detection_coverage')):>5} "
            f"{_fmt(r.get('unique_contribution_cam_b')):>5}  "
            f"{'*' if r['pareto'] else ' '}")
    return "\n".join(lines)


def render(ranked, notes, pareto_cats, kept, dropped, cfg, args, cov_floor):
    out = []
    out.append(f"Ranked {len(kept)} camera position(s) "
               f"(grouped by h/r/az{', subject=' + args.subject if args.subject else ''}).")
    out.append(f"composite = " + " + ".join(
        f"{w:g}*{n}" for n, w in _active_weights.items()) + "   (P = Pareto-optimal)")
    out.append("")
    out.append(format_table(ranked, args.top))
    out.append("")

    pareto = [r for r in ranked if r["pareto"]]
    if pareto_cats:
        names = ", ".join(layout_label(r) for r in pareto)
        out.append(f"Pareto-optimal over [{', '.join(pareto_cats)}]: {names}")
        if len(pareto) == 1 and ranked[0]["pareto"]:
            out.append(f"  -> {layout_label(ranked[0])} is the SOLE Pareto winner: "
                       f"best under ANY weighting (weight-independent).")
    else:
        out.append("Pareto: not enough common categories to compute.")
    out.append("")

    # Preset sensitivity.
    out.append("Preset sensitivity (top-3 by composite):")
    tops = {}
    for pname in cfg["ranking"]["presets"]:
        r2, _, _ = rank(kept, cfg, preset_weights(cfg, pname))
        tops[pname] = [layout_label(x) for x in r2[:3]]
        out.append(f"  {pname:<11} {'  >  '.join(tops[pname])}")
    winners = {v[0] for v in tops.values()}
    out.append(f"  => #1 is {'STABLE' if len(winners) == 1 else 'WEIGHT-SENSITIVE'} "
               f"across presets ({', '.join(sorted(winners))}).")

    drop_notes = {m: n for m, n in notes.items()}
    if drop_notes:
        out.append("")
        out.append("Metric notes:")
        for m, n in drop_notes.items():
            out.append(f"  {m}: {n}")

    if dropped:
        out.append("")
        out.append(f"Gated out ({len(dropped)}, coverage_floor={cov_floor}):")
        for lay, why in dropped:
            out.append(f"  {layout_label(lay)}: {why}")
    return "\n".join(out)


def write_csv(path, ranked):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ["rank", "h_m", "r_m", "rel_az_deg", "score",
            "accuracy", "reliability", "geometry", "pareto",
            "mpjpe_mm", "mpjpe_worst", "detection_coverage", "coverage_min",
            "pck50", "joint_visibility_both", "unique_contribution_cam_b",
            "jitter_variance", "id_drops", "convergence_angle_deg", "tilt_deg",
            "n_subjects"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in ranked:
            c = r["_cats"]
            row = []
            for col in cols:
                if col in ("accuracy", "reliability", "geometry"):
                    row.append(c.get(col))
                else:
                    row.append(r.get(col))
            w.writerow(row)


# Set by main() so render() can print the active weights.
_active_weights = {}


def main():
    ap = argparse.ArgumentParser(description="Rank camera positions from results.csv")
    ap.add_argument("--results", default=os.path.join(_REPO, "results", "results.csv"))
    ap.add_argument("--config", default=os.path.join(_REPO, "config", "experiment.yaml"))
    ap.add_argument("--preset", default="balanced",
                    help="category-weight preset (from experiment.yaml ranking.presets)")
    ap.add_argument("--weights", default=None,
                    help="override, e.g. accuracy=0.5,reliability=0.3,geometry=0.2")
    ap.add_argument("--coverage-floor", type=float, default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--subject", default=None,
                    help="rank within one subject_pos_name instead of aggregating")
    ap.add_argument("--out", default=None,
                    help="optional CSV output path (suggest reports/; never results/)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rk = cfg["ranking"]
    cov_floor = (args.coverage_floor if args.coverage_floor is not None
                 else float(rk.get("coverage_floor", 0.8)))

    if not os.path.exists(args.results):
        print(f"rank: results file not found: {args.results}")
        return
    rows = load_results(args.results)
    if not rows:
        print("rank: results.csv has no rows.")
        return

    layouts = aggregate_by_layout(rows, subject=args.subject)
    kept, dropped = gate(layouts, cov_floor)
    if not kept:
        print(f"rank: no layouts pass the gate "
              f"(coverage_floor={cov_floor}, need non-NaN mpjpe).")
        return

    if args.weights:
        cw = parse_weights(args.weights, cfg)
    elif args.preset in rk["presets"]:
        cw = preset_weights(cfg, args.preset)
    else:
        cw = default_category_weights(cfg)

    global _active_weights
    _active_weights = cw

    ranked, notes, pareto_cats = rank(kept, cfg, cw)
    print(render(ranked, notes, pareto_cats, kept, dropped, cfg, args, cov_floor))

    if args.out:
        write_csv(args.out, ranked)
        print(f"\nrank: wrote {len(ranked)} rows to {args.out}")


if __name__ == "__main__":
    main()
