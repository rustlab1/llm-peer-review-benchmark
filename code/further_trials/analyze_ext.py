"""Analysis and figures for the trials published in 2026 (Figure 2 and Supplementary Figures 4 to 6).

Same definitions as the main study (code/analyze_primary.py and make_figures.py):
- each item score is the mean of the two original judges (Fable 5 = J1, Opus 5 = J2), 1 / 0.5 / 0;
- summed major (of 9) and minor (of 5) scores per review;
- per-error detection = mean item score over reviews, in percent (0.5 counts half);
- an error counts as detected (or an error type as flagged on a control) when the mean judge score is >= 0.5;
- sensitivity = share of the 14 inserted errors detected over a model's reviews of the error-seeded manuscripts;
  specificity = share of the 14 error types not flagged over its reviews of the unmodified manuscripts;
- 95% CIs from a cluster bootstrap that resamples manuscripts, then reviews (or item observations) within them;
- human comparator: untrained control group of Schroter et al. 2008 (2.40 major, 0.89 minor; ~27.0% detection).

    python3 code/further_trials/analyze_ext.py --trial-set new --tag new_trials   (see run_final_analysis.sh)
Reads $FT_WORK (made by prepare_workspace.py) and $SD1 (Supplementary Data 1). Writes
$FT_WORK/data/results_<tag>.json and results_<tag>_summary.md, and figures to $FT_WORK/figures.
"""
import argparse
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# SD1: the extracted Supplementary Data 1; FT_WORK: the working folder made by prepare_workspace.py
SD1 = os.environ.get("SD1", os.path.join(REPO, "Supplementary_Data_1"))
EXT = os.environ.get("FT_WORK", os.path.join(REPO, "work_further_trials"))
DATA = os.path.join(EXT, "data")
RNG = np.random.default_rng(20261027)
NBOOT = int(os.environ.get("NBOOT", 10000))   # 10,000 as in the main study; lower only for quick checks
MAJOR = [f"M{i}" for i in range(1, 10)]
MINOR = [f"m{i}" for i in range(1, 6)]
ITEMS = MAJOR + MINOR
ERR_SHORT = {"M1": "Poor justification", "M2": "Biased allocation", "M3": "No sample size calculation",
             "M4": "Outcome measure validity", "M5": "Not intention-to-treat", "M6": "Poor response rate",
             "M7": "Unjustified conclusions", "M8": "Abstract vs results", "M9": "Inconsistent denominators",
             "m1": "No ethics approval", "m2": "Non-enrolment unexplained", "m3": "Text vs table",
             "m4": "Comparison reversed", "m5": "Hawthorne effect"}
NAME = {"human": "Human reviewers", "haiku45": "Haiku 4.5", "sonnet5": "Sonnet 5", "opus5": "Opus 5",
        "fable5": "Fable 5", "gpt54mini": "GPT-5.4 mini", "gpt54": "GPT-5.4", "gpt55": "GPT-5.5", "opus55": "Opus 5.5",
        "gpt61sol": "GPT-6.1 Sol", "gpt6luna": "GPT-6 Luna", "gemini38fl": "Gemini 3.8 Flash", "kimik3": "Kimi K3",
        "dsv4pro": "DeepSeek V4 Pro", "qwen38_27b": "Qwen3.8-27B", "gptoss120b": "gpt-oss-120b"}
OPEN = {"kimik3", "dsv4pro", "qwen38_27b", "gptoss120b"}
# Schroter et al. 2008, untrained controls (as in the main study)
HUM_MAJ, HUM_MIN = 2.40, 0.89
HUM_MAJ_RANGE, HUM_MIN_RANGE = (2.13, 2.71), (0.71, 0.99)
HUM_SENS = 27.0
HUM_PER_ERR = None   # filled from the main study's results file when available

# model-type colours: muted slate (proprietary) and dusty rose (open-weight), kept apart from the PRGn colours of b and c
BLUE, RED, GREY = os.environ.get("COL_PROP", "#4F6375"), os.environ.get("COL_OPEN", "#C47885"), "#8A8A8A"
D_YMIN = 60   # panel d y-axis lower limit (specificity of all models lies well above this)
INK, INK2, RULE = "#1A1A1A", "#4D4D4D", "#D9D9D9"
# ColorBrewer PRGn (purple = missed, white = partial, green = found); colour-blind safe
CMAP = LinearSegmentedColormap.from_list("prgn", ["#762A83", "#AF8DC3", "#F7F7F7", "#7FBF7B", "#1B7837"])
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "legend.frameon": False, "axes.titlesize": 8})


def col(g):
    return GREY if g == "human" else (RED if g in OPEN else BLUE)


def text_on(v):
    r, g, b, _ = CMAP(v / 100)
    return "white" if 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2 < 0.22 else INK


# ------------------------------------------------------------------------------------------ loading
def load(min_judges):
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    scores = {}
    for m in man:
        js = {}
        for j in ("J1", "J2"):
            p = os.path.join(DATA, "judgements", f"{m['id']}__{j}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                js[j] = {s["id"]: float(s["score"]) for s in d["scores"]}
        if len(js) >= min_judges:
            scores[m["id"]] = {it: float(np.mean([js[j][it] for j in js])) for it in ITEMS}
            scores[m["id"]]["_judges"] = sorted(js)
            scores[m["id"]]["_pj"] = js
    return man, scores


ORIG_MODEL = {"haiku": "haiku45", "sonnet": "sonnet5", "opus": "opus5", "fable": "fable5"}
ORIG_TID = {1: "N2026", 2: "I2026", 3: "M2026"}


def load_original(root, min_judges, all_reps=False):
    """The three 2026 trials of the main study (papers N, I, M): reviews and controls with the same two judges.
    By default only the first of the four reviews per model and trial is used, so that every trial carries one
    error-seeded and one control review per model, as in the additional trials."""
    man, scores = [], {}
    for mf, jdir, kind in (("manifest.json", "judgements", "review"), ("control_manifest.json", os.path.join("controls", "judgements"), "control")):
        p = os.path.join(root, "data", mf)
        if not os.path.exists(p):
            continue
        for m in json.load(open(p)):
            rep = m.get("rep", 1)
            if kind == "review" and rep != 1 and not all_reps:
                continue
            js = {}
            for j in ("J1", "J2"):
                f = os.path.join(root, "data", jdir, f"{m['id']}__{j}.json")
                if os.path.exists(f):
                    js[j] = {x["id"]: float(x["score"]) for x in json.load(open(f))["scores"]}
            if len(js) < min_judges:
                continue
            mod = ORIG_MODEL.get(m["model"], m["model"])
            man.append({"id": "orig_" + m["id"], "tid": ORIG_TID[int(m["paper"])], "model": mod, "kind": kind, "rep": rep})
            sc = {it: float(np.mean([js[j][it] for j in js])) for it in ITEMS}
            sc["_judges"], sc["_pj"] = sorted(js), js
            scores["orig_" + m["id"]] = sc
    return man, scores


def collapse_reps(man, scores):
    """Average repeated reviews of the same model, trial and kind into one record, so that every trial carries equal
    weight (the main study ran four reviews per model on N, I and M; the additional trials have one)."""
    groups = collections.defaultdict(list)
    for m in man:
        if m["id"] in scores:
            groups[(m["model"], m["tid"], m["kind"])].append(m["id"])
    man2, sc2 = [], {}
    for (mod, tid, kind), ids in groups.items():
        nid = ids[0] if len(ids) == 1 else f"avg_{mod}_{tid}_{kind}"
        man2.append({"id": nid, "tid": tid, "model": mod, "kind": kind, "n_reps": len(ids)})
        sc = {it: float(np.mean([scores[i][it] for i in ids])) for it in ITEMS}
        # detection is decided per review (mean judge score >= 0.5) and then averaged over the repeated reviews,
        # so that sensitivity keeps its per-review meaning
        sc["_det"] = {it: float(np.mean([scores[i][it] >= 0.5 for i in ids])) for it in ITEMS}
        sc["_judges"] = scores[ids[0]].get("_judges"); sc["_pj"] = scores[ids[0]].get("_pj")
        sc2[nid] = sc
    return man2, sc2


SUFFIX = {}   # model -> label suffix (marks models evaluated on fewer trials)


def label(m):
    return NAME[m] + SUFFIX.get(m, "")


def cluster_boot(by_trial, stat=np.mean):
    trials = sorted(by_trial)
    allv = np.concatenate([by_trial[t] for t in trials])
    point = float(stat(allv))
    if len(trials) < 2:
        return point, point, point
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        pick = RNG.choice(len(trials), size=len(trials), replace=True)
        out[b] = stat(np.concatenate([RNG.choice(by_trial[trials[i]], size=len(by_trial[trials[i]]), replace=True)
                                      for i in pick]))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def summarize(man, scores):
    models = sorted({m["model"] for m in man if m["id"] in scores})
    res = {}
    for mod in models:
        r = {}
        for kind in ("review", "control"):
            ids = [m for m in man if m["model"] == mod and m["kind"] == kind and m["id"] in scores]
            if not ids:
                continue
            tot, maj, mi, det = (collections.defaultdict(list) for _ in range(4))
            for m in ids:
                s = scores[m["id"]]
                maj[m["tid"]].append(sum(s[i] for i in MAJOR))
                mi[m["tid"]].append(sum(s[i] for i in MINOR))
                tot[m["tid"]].append(sum(s[i] for i in ITEMS))
                det[m["tid"]].extend([s["_det"][i] if "_det" in s else (1.0 if s[i] >= 0.5 else 0.0) for i in ITEMS])
            arr = lambda d: {k: np.array(v, float) for k, v in d.items()}
            k = "s" if kind == "review" else "c"
            pmaj, lmaj, hmaj = cluster_boot(arr(maj))
            pmin, lmin, hmin = cluster_boot(arr(mi))
            ptot, ltot, htot = cluster_boot(arr(tot))
            pdet, ldet, hdet = cluster_boot(arr(det))
            r[kind] = {"n_reviews": len(ids), "n_trials": len(maj), "major_mean": pmaj, "major_ci": [lmaj, hmaj],
                       "minor_mean": pmin, "minor_ci": [lmin, hmin], "total_mean": ptot, "total_ci": [ltot, htot],
                       "per_error_pct": {i: 100 * float(np.mean([scores[m["id"]][i] for m in ids])) for i in ITEMS},
                       "per_error_detect_pct": {i: 100 * float(np.mean([scores[m["id"]]["_det"][i] if "_det" in scores[m["id"]]
                                                                         else scores[m["id"]][i] >= 0.5 for m in ids]))
                                                for i in ITEMS}}
            if kind == "review":
                r[kind]["sens"], r[kind]["sens_ci"] = 100 * pdet, [100 * ldet, 100 * hdet]
            else:
                r[kind]["spec"], r[kind]["spec_ci"] = 100 * (1 - pdet), [100 * (1 - hdet), 100 * (1 - ldet)]
        res[mod] = r
    return res


def per_trial(man, scores):
    """mean summed score (of 14) per model and trial on the error-seeded manuscripts"""
    d = collections.defaultdict(list)
    for m in man:
        if m["kind"] == "review" and m["id"] in scores:
            d[(m["model"], m["tid"])].append(sum(scores[m["id"]][i] for i in ITEMS))
    return {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in d.items()}


def agreement(scores):
    a = b = 0
    det_pairs = []
    for s in scores.values():
        pj = s["_pj"]
        if "J1" in pj and "J2" in pj:
            for i in ITEMS:
                a += 1
                b += pj["J1"][i] == pj["J2"][i]
                det_pairs.append((pj["J1"][i] > 0, pj["J2"][i] > 0))
    if not det_pairs:
        return None
    x = np.array(det_pairs, float)
    po = np.mean(x[:, 0] == x[:, 1])
    pe = np.mean(x[:, 0]) * np.mean(x[:, 1]) + (1 - np.mean(x[:, 0])) * (1 - np.mean(x[:, 1]))
    return {"exact_agreement_pct": 100 * b / a, "kappa_detected": float((po - pe) / (1 - pe)) if pe < 1 else None,
            "n_items": a}


# ------------------------------------------------------------------------------------------ figure
ZEBRA = "#F3F3F3"
GREEN, PURPLE = "#1B7837", "#9970AB"


def zebra(ax, n):
    for i in range(n):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color=ZEBRA, lw=0, zorder=0)


def colour_labels(ax, groups):
    for t, g in zip(ax.get_yticklabels(), groups):
        t.set_color(INK if g == "human" else col(g))
        if g == "human":
            t.set_fontweight("bold")


def row_axes(ax, groups, labels=True):
    n = len(groups)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label(g) if g != "human" else NAME[g] for g in groups] if labels else [""] * n)
    ax.tick_params(axis="y", length=0)
    ax.axhline(0.5, color=RULE, lw=0.6, ls=(0, (2, 2)))



def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


LV_A, LV_B, CROSS = (float(v) for v in os.environ.get("LABEL_COST", "1.2,0.3,5").split(","))


def _seg_cross(p1, p2, p3, p4):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = o(p3, p4, p1), o(p3, p4, p2), o(p1, p2, p3), o(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _seg_in_box(p1, p2, box, n=14):
    for t in np.linspace(0.08, 0.92, n):
        x, y = p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])
        if box[0] < x < box[2] and box[1] < y < box[3]:
            return True
    return False


def place_labels(ax, pts, fontsize=6.5, static=(), soft=(), tries=60):
    """Put each label next to its point, as close as possible, avoiding the other labels, markers, leader lines and
    static boxes (display coordinates), and preferably the soft boxes (confidence-interval lines). Candidates lie in
    16 directions at increasing distance; a thin leader line is drawn when a label is not directly beside its point.
    Several placement orders are tried and the layout with the lowest total cost is kept. Labels are coloured by
    model type, as in panel a, on a white background so that CI lines pass behind them."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    P = [tuple(ax.transData.transform((x, y))) for x, y, _ in pts]
    rad = 3.6 * pt
    markers = [(px - rad, py - rad, px + rad, py + rad) for px, py in P]
    axb = ax.get_window_extent(rend)
    size = {}
    for x, y, m in pts:
        t = ax.text(0, 0, label(m), fontsize=fontsize)
        bb = t.get_window_extent(rend); size[m] = (bb.width, bb.height); t.remove()
    cands = []
    for lvl, d in enumerate((4.5, 9, 14, 20, 27, 35, 45, 56)):
        for k in range(16):
            th = np.pi * 2 * k / 16
            c, s_ = np.cos(th), np.sin(th)
            ha = "left" if c > 0.3 else "right" if c < -0.3 else "center"
            va = "bottom" if s_ > 0.3 else "top" if s_ < -0.3 else "center"
            pref = 0.0 if abs(s_) < 0.3 else 0.15        # slight preference for left/right of the point
            cands.append((lvl, d * c, d * s_ * 0.8, ha, va, pref))

    def geom(i, cand):
        lvl, dx, dy, ha, va, pref = cand
        px, py = P[i]; w, h = size[pts[i][2]]
        ox, oy = px + dx * pt, py + dy * pt
        x0 = ox if ha == "left" else ox - w if ha == "right" else ox - w / 2
        y0 = oy if va == "bottom" else oy - h if va == "top" else oy - h / 2
        box = (x0 - 1.5, y0 - 1, x0 + w + 1.5, y0 + h + 1)
        cx = min(max(px, box[0] + 1.5), box[2] - 1.5); cy = min(max(py, box[1] + 1), box[3] - 1)
        seg = ((px, py), (cx, cy)) if lvl >= 1 and np.hypot(cx - px, cy - py) > rad * 1.6 else None
        return box, seg, (ox, oy)

    def layout(order):
        placed, total = {}, 0.0
        for i in order:
            w, h = size[pts[i][2]]
            best = None
            for cand in cands:
                box, seg, _ = geom(i, cand)
                lv = cand[0]
                cost = LV_A * lv + LV_B * lv ** 2 + cand[5]
                for j, (bj, sj, _c) in placed.items():
                    cost += 60 * _overlap(box, bj) / (w * h)
                    if sj is not None:
                        if _seg_in_box(sj[0], sj[1], box):
                            cost += CROSS
                        if seg is not None and _seg_cross(seg[0], seg[1], sj[0], sj[1]):
                            cost += CROSS
                    if seg is not None and _seg_in_box(seg[0], seg[1], bj):
                        cost += CROSS
                for j, mb in enumerate(markers):
                    if j != i:
                        cost += 60 * _overlap(box, mb) / (w * h)
                        if seg is not None and _seg_in_box(seg[0], seg[1], mb, n=8):
                            cost += 4
                cost += 30 * sum(_overlap(box, b) for b in static) / (w * h)
                cost += 1.5 * sum(_overlap(box, b) > 0 for b in soft)
                if box[0] < axb.x0 + 2 or box[2] > axb.x1 + 40 * pt or box[1] < axb.y0 + 2 or box[3] > axb.y1:
                    cost += 80
                if best is None or cost < best[0]:
                    best = (cost, box, seg, cand)
            placed[i] = (best[1], best[2], best[3])
            total += best[0]
        return total, placed

    crowd = [sum(np.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) < 45 * pt for j in range(len(P))) for i in range(len(P))]
    rng = np.random.default_rng(7)
    orders = [sorted(range(len(pts)), key=lambda i: -crowd[i]), sorted(range(len(pts)), key=lambda i: P[i][0]),
              sorted(range(len(pts)), key=lambda i: -P[i][1])]
    orders += [list(rng.permutation(len(pts))) for _ in range(tries)]
    best = min((layout(o) for o in orders), key=lambda r: r[0])
    for i, (box, seg, cand) in best[1].items():
        _, _, (ox, oy) = geom(i, cand)
        tx, ty = ax.transData.inverted().transform((ox, oy))
        ax.text(tx, ty, label(pts[i][2]), fontsize=fontsize, color=col(pts[i][2]), ha=cand[3], va=cand[4], zorder=6,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))
        if seg is not None:
            (px, py), (cx, cy) = seg
            vx, vy = cx - px, cy - py; L = np.hypot(vx, vy)
            a = ax.transData.inverted().transform([(px + vx / L * rad, py + vy / L * rad), (cx, cy)])
            ax.plot(a[:, 0], a[:, 1], color="#8C8C8C", lw=0.45, zorder=5)


def panel_d(ax, res, with_ctl, style="near_full"):
    """Sensitivity (inserted errors found) against specificity (error types not flagged on unmodified manuscripts).
    style: 'column' = full 0-100 axes, labels in a column right of the plot (v3);
           'near_zoom' = x 20-100 and y 60-100 (human reviewers' line visible), labels next to the points;
           'near_tight' = axes zoomed on the models, labels next to the points, human value marked at the edge."""
    pts, cis = [], []
    for mod in with_ctl:
        r, c = res[mod]["review"], res[mod]["control"]
        x, y = r["sens"], c["spec"]
        ax.plot(r["sens_ci"], [y, y], color=col(mod), lw=0.5, alpha=0.35)
        ax.plot([x, x], c["spec_ci"], color=col(mod), lw=0.5, alpha=0.35)
        ax.scatter(x, y, s=30, color=col(mod), edgecolors="white", linewidths=0.6, zorder=4)
        pts.append([x, y, mod])
        cis.append((r["sens_ci"][0], y, r["sens_ci"][1], y)); cis.append((x, c["spec_ci"][0], x, c["spec_ci"][1]))
    if style == "column":
        _p = ax.get_position(); ax.set_position([_p.x0, _p.y0, _p.width * 0.74, _p.height])
        lab_x = 109
        y_top, y_bot = 99, max(30, 99 - 5.0 * (len(pts) - 1))
        slots = np.linspace(y_top, y_bot, len(pts))
        ymid = (y_top + y_bot) / 2
        pts.sort(key=lambda p: -np.arctan2(p[1] - ymid, lab_x - p[0]))
        for (x, y, mod), ty in zip(pts, slots):
            ax.annotate(label(mod), xy=(x, y), xytext=(lab_x, ty), fontsize=7, color=INK, ha="left", va="center",
                        zorder=5, annotation_clip=False, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
                        arrowprops=dict(arrowstyle="-", color="#A6A6A6", lw=0.4, shrinkA=0, shrinkB=2.5,
                                        relpos=(0, 0.5)))
        ax.axvline(HUM_SENS, color=GREY, lw=0.8, ls=(0, (3, 2)))
        ax.text(HUM_SENS + 1.5, 2, "human reviewers\n2008 (untrained)", fontsize=7, color=INK2, va="bottom")
        ax.set_xlim(0, 104); ax.set_ylim(0, 104); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines["bottom"].set_bounds(0, 100); ax.spines["left"].set_bounds(0, 100)
    else:
        if style == "near_full":
            x0, x1, y0, y1 = 0, 104, 0, 104
            ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
            ax.spines["bottom"].set_bounds(0, 100); ax.spines["left"].set_bounds(0, 100)
        elif style == "near_zoom":
            x0, x1, y0, y1 = 20, 101, 64, 96
            ax.set_xticks([25, 50, 75, 100]); ax.set_yticks([70, 80, 90])
        else:
            x0, x1, y0, y1 = 40, 101, 66, 94
            ax.set_xticks([40, 50, 60, 70, 80, 90, 100]); ax.set_yticks([70, 80, 90])
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        static = []
        if style in ("near_zoom", "near_full"):
            ax.axvline(HUM_SENS, color=GREY, lw=0.8, ls=(0, (3, 2)))
            t = ax.text(HUM_SENS + 1.2, y0 + 1.0, "human reviewers\n2008 (untrained)", fontsize=6.5, color=INK2,
                        va="bottom")
            ax.figure.canvas.draw()
            bb = t.get_window_extent(ax.figure.canvas.get_renderer())
            static.append((bb.x0, bb.y0, bb.x1, bb.y1))
            lx = ax.transData.transform((HUM_SENS, y0))[0]
            static.append((lx - 1, ax.transData.transform((0, y0))[1], lx + 1, ax.transData.transform((0, y1))[1]))
        else:
            ax.annotate("human reviewers (27%)", xy=(x0, y0 + 2.2), xytext=(x0 + 7.5, y0 + 2.2), fontsize=6.5,
                        color=INK2, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.6, mutation_scale=6, shrinkA=1, shrinkB=0))
            static.append(tuple(ax.transData.transform((x0, y0)).tolist())
                          + tuple(ax.transData.transform((x0 + 30, y0 + 4.5)).tolist()))
        ax.figure.canvas.draw()
        soft = []
        for (a0, b0, a1, b1) in cis:
            (u0, v0), (u1, v1) = ax.transData.transform([(a0, b0), (a1, b1)])
            soft.append((min(u0, u1) - 1, min(v0, v1) - 1, max(u0, u1) + 1, max(v0, v1) + 1))
        place_labels(ax, pts, fontsize=6.5, static=static, soft=soft)
    ax.set_xlabel("Inserted errors found (% of 14 error types)")
    ax.set_ylabel("Error types not flagged on\nunmodified manuscripts (%)")
    ax.grid(color=RULE, lw=0.4, zorder=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=5, label="proprietary model"),
                       Line2D([], [], marker="o", ls="", color=RED, ms=5, label="open-weight model")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2, handletextpad=0.3)
    ax.text(-0.20, 1.03, "d", transform=ax.transAxes, fontsize=10, fontweight="bold")

def figure(res, order, out, n_trials, hum_per_err, name="Figure2_ext27"):
    fig = plt.figure(figsize=(7.2, 9.7))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 2.45], hspace=0.38)
    top = outer[0].subgridspec(1, 2, width_ratios=[0.95, 1.55], wspace=0.05)
    bot = outer[1].subgridspec(1, 2, width_ratios=[0.95, 1.55], wspace=0.30)
    groups = ["human"] + order
    # a
    ax = fig.add_subplot(top[0, 0])
    row_axes(ax, groups)
    zebra(ax, len(groups))
    for i, g in enumerate(groups):
        c = col(g)
        if g == "human":
            maj, (mlo, mhi) = HUM_MAJ / 9 * 100, [v / 9 * 100 for v in HUM_MAJ_RANGE]
            mi, (ilo, ihi) = HUM_MIN / 5 * 100, [v / 5 * 100 for v in HUM_MIN_RANGE]
        else:
            r = res[g]["review"]
            maj, (mlo, mhi) = r["major_mean"] / 9 * 100, [v / 9 * 100 for v in r["major_ci"]]
            mi, (ilo, ihi) = r["minor_mean"] / 5 * 100, [v / 5 * 100 for v in r["minor_ci"]]
        ax.plot([mlo, mhi], [i - 0.15] * 2, color=c, lw=1.3, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.plot([ilo, ihi], [i + 0.15] * 2, color=c, lw=1.3, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.scatter(maj, i - 0.15, s=34, color=c, edgecolors="white", linewidths=0.7, zorder=4)
        ax.scatter(mi, i + 0.15, s=34, facecolors="white", edgecolors=c, linewidths=1.1, zorder=4)
    ax.set_xlim(-2.5, 102.5); ax.set_xticks([0, 25, 50, 75, 100]); ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("Inserted errors found (%)")
    ax.grid(axis="x", color=RULE, lw=0.4, zorder=0); ax.spines["left"].set_visible(False)
    colour_labels(ax, groups)
    # legend: marker fill = error class, colour = model type (column-wise fill gives two rows)
    h = [Line2D([], [], marker="o", ls="", color=INK2, mec="white", ms=5.5, label="major errors (of 9)"),
         Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=5.5, label="proprietary model"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, mew=1.1, ms=5.5, label="minor errors (of 5)"),
         Line2D([], [], marker="o", ls="", color=RED, mec="white", ms=5.5, label="open-weight model")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, borderaxespad=0, handletextpad=0.3,
              labelspacing=0.25, columnspacing=0.9)
    ax.text(-0.62, 1.075, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")
    if SUFFIX:
        n_all = max(res[m]["review"]["n_trials"] for m in order)
        n_few = min(res[m]["review"]["n_trials"] for m in order)
        ax.text(-0.62, -0.115, f"\u2020 tested on a subset of three trials; other models on all {n_all} trials",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK2)
    # b
    ax = fig.add_subplot(top[0, 1])
    M = np.array([[hum_per_err.get(e, np.nan) for e in ITEMS]] +
                 [[res[g]["review"]["per_error_pct"][e] for e in ITEMS] for g in order])
    # tiles with thin white gaps and a slightly wider gap between major and minor errors
    xs = [j - 0.5 + (0.3 if j >= 9 else 0) for j in range(14)]
    ys = [i - 0.5 for i in range(M.shape[0])]
    tile_heatmap(ax, M, xs, ys, CMAP, pad=0.045)
    row_axes(ax, groups, labels=False)
    ax.set_xlim(xs[0] - 0.05, xs[-1] + 1.05)
    ax.set_xticks([x + 0.5 for x in xs]); ax.set_xticklabels([ERR_SHORT[e] for e in ITEMS], rotation=45, ha="right",
                                                         rotation_mode="anchor", fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(xs[j] + 0.5, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.8, color=text_on(M[i, j]))
    ax.axhline(0.5, color="white", lw=2.4)
    ax.text(np.mean([x + 0.5 for x in xs[:9]]), -0.95, "major errors", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.text(np.mean([x + 0.5 for x in xs[9:]]), -0.95, "minor errors", ha="center", va="bottom", fontsize=7.5, color=INK2)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    im = ScalarMappable(norm=Normalize(0, 100), cmap=CMAP)
    ax.text(-0.03, 1.10, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.canvas.draw()
    pb = ax.get_position()
    cax = fig.add_axes([pb.x1 + 0.012, pb.y0 + 0.30 * pb.height, 0.010, 0.40 * pb.height])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 50, 100]); cb.set_label("Detection (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2); cb.outline.set_linewidth(0.5)
    # c
    ax = fig.add_subplot(bot[0, 0])
    with_ctl = [m for m in order if "control" in res[m]]
    ins = {e: np.mean([res[m]["review"]["per_error_pct"][e] for m in order]) for e in ITEMS}
    un = {e: np.mean([res[m]["control"]["per_error_pct"][e] for m in with_ctl]) if with_ctl else np.nan for e in ITEMS}
    # major and minor errors as two blocks, each sorted by detection of the inserted error
    maj_o = sorted(MAJOR, key=lambda e: -ins[e]); min_o = sorted(MINOR, key=lambda e: -ins[e])
    eo = maj_o + min_o
    ypos = list(range(9)) + [i + 9.7 for i in range(5)]
    for k, y in enumerate(ypos):
        if k % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color=ZEBRA, lw=0, zorder=0)
    for y, e in zip(ypos, eo):
        ax.plot([un[e], ins[e]], [y, y], color="#C8C8C8", lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter(ins[e], y, s=36, color=GREEN, edgecolors="white", linewidths=0.7, zorder=4)
        ax.scatter(un[e], y, s=36, color=PURPLE, edgecolors="white", linewidths=0.7, zorder=4)
    ax.set_ylim(ypos[-1] + 0.6, -0.6); ax.set_yticks(ypos); ax.set_yticklabels([ERR_SHORT[e] for e in eo])
    ax.tick_params(axis="y", length=0)
    for y, lab in ((-0.5, "major errors"), (8.95, "minor errors")):
        ax.text(-0.03, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7,
                color=INK2, fontstyle="italic")
    ax.set_xlim(-2.5, 102.5); ax.set_xticks([0, 25, 50, 75, 100]); ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("Error type flagged (%)"); ax.grid(axis="x", color=RULE, lw=0.4, zorder=0)
    ax.spines["left"].set_visible(False)
    nrev = sum(res[m]["review"]["n_reviews"] for m in order)
    nctl = sum(res[m]["control"]["n_reviews"] for m in with_ctl)
    h = [Line2D([], [], marker="o", ls="", color=GREEN, mec="white", ms=5.5, label="error inserted"),
         Line2D([], [], marker="o", ls="", color=PURPLE, mec="white", ms=5.5, label="unmodified trial")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), borderaxespad=0, labelspacing=0.2,
              handletextpad=0.3)
    ax.text(-0.62, 1.10, "c", transform=ax.transAxes, fontsize=10, fontweight="bold")
    # d: sensitivity against specificity, one point per model
    ax = fig.add_subplot(bot[0, 1])
    panel_d(ax, res, with_ctl, os.environ.get("FIG2D_STYLE", "near_full"))
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def by_trial_figure(man, scores, order, out, trial_names, name="SupplFigure_ext27_by_trial"):
    pt = per_trial(man, scores)
    tids = sorted({m["tid"] for m in man if m["id"] in scores and m["kind"] == "review"})
    M = np.array([[pt.get(f"{m}|{t}", np.nan) / 14 * 100 for t in tids] for m in order])
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(order) + 1.8))
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([label(m) for m in order]); ax.tick_params(length=0)
    ax.set_xticks(range(len(tids))); ax.set_xticklabels([trial_names.get(t, t) for t in tids], rotation=60,
                                                        ha="right", rotation_mode="anchor", fontsize=6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color=text_on(M[i, j]))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("Summed judge score (% of 14)", fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def tile_heatmap(ax, M, xs, ys, cmap, pad=0.07, missing="#E6E6E6"):
    """Draw matrix M as separate square tiles with thin white gaps (RNA-seq heatmap style).
    xs, ys: left/top coordinate of each column/row (gaps between groups are built into them)."""
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    rects, cols = [], []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            v = M[i, j]
            rects.append(Rectangle((x + pad, y + pad), 1 - 2 * pad, 1 - 2 * pad))
            cols.append(missing if np.isnan(v) else cmap(v / 100))
    ax.add_collection(PatchCollection(rects, facecolors=cols, edgecolors="none"))
    ax.set_xlim(xs[0] - 0.05, xs[-1] + 1.05)
    ax.set_ylim(ys[-1] + 1.05, ys[0] - 0.05)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_facecolor("white")


def per_model_heatmaps(man, scores, order, out, trial_names, hum_per_err, trial_set_of, name="SupplFigure_per_model_heatmaps"):
    """One heatmap per reviewer model: error types (x) by trial (y), same colour scale as Figure 2b.
    Cell = judge score for that error in that model's review of the error-seeded version (mean of the two judges,
    and of repeat reviews where there are several), as %. Top row in every panel: human reviewers 2008.
    Cells are drawn as tiles with small white gaps, with wider gaps between major and minor errors, below the
    human row and between trial sets."""
    from collections import defaultdict
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    acc = defaultdict(list)
    for m in man:
        if m["kind"] == "review" and m["id"] in scores:
            acc[(m["model"], m["tid"])].append([scores[m["id"]][e] for e in ITEMS])
    cell = {k: np.mean(v, axis=0) * 100 for k, v in acc.items()}
    tids = sorted({t for (_, t) in cell})
    if not tids:
        return
    mean_t = {t: np.mean([cell[(mo, t)].mean() for mo in order if (mo, t) in cell]) for t in tids}
    set_rank = {"further": 0, "recent": 0, "phase3": 1, "main": 2}   # "further": the 55 trials as one block
    tids.sort(key=lambda t: (set_rank.get(trial_set_of(t), 3), -mean_t[t]))
    models = list(reversed(order))
    GAP_MM, GAP_HUM, GAP_SET = 0.45, 0.45, 0.35
    xs = [j + (GAP_MM if j >= 9 else 0) for j in range(14)]
    ys, y, prev = [0.0], 1 + GAP_HUM, None
    for t in tids:
        st = trial_set_of(t)
        if prev is not None and st != prev:
            y += GAP_SET
        ys.append(y); y += 1; prev = st
    height_units = ys[-1] + 1
    hum = np.array([hum_per_err.get(e, np.nan) for e in ITEMS])

    def draw(mods, fname, ncol, panel_w, lab_fs, title_fs, page=None):
        """Draw the heatmaps of the models in mods (ncol per row) and save as fname. page=(width, height) in inches
        draws one row of panels at its printed size (for a landscape page), so the fonts print at their set size."""
        nrow = int(np.ceil(len(mods) / ncol))
        ph = 0.105 * height_units + 0.5
        if page:
            fig = plt.figure(figsize=page)
            gs = fig.add_gridspec(1, ncol, left=0.245, right=0.925, top=0.93, bottom=0.165, wspace=0.07)
        else:
            fig = plt.figure(figsize=(panel_w * ncol + 1.6, ph * nrow + 0.9))
            gs = fig.add_gridspec(nrow, ncol, hspace=0.42 / ph, wspace=0.06)
        missing = False
        for k, mo in enumerate(mods):
            r, c = divmod(k, ncol)
            ax = fig.add_subplot(gs[r, c])
            M = np.full((len(tids) + 1, 14), np.nan)
            M[0] = hum
            for i, t in enumerate(tids):
                if (mo, t) in cell:
                    M[i + 1] = cell[(mo, t)]
            missing |= bool(np.isnan(M[1:]).any())
            tile_heatmap(ax, M, xs, ys, CMAP)
            n_sc = sum((mo, t) in cell for t in tids)
            maj = np.nanmean(M[1:, :9]) if n_sc else np.nan
            mnr = np.nanmean(M[1:, 9:]) if n_sc else np.nan
            ttl = (f"{label(mo)}\nmajor {maj:.0f}%, minor {mnr:.0f}%" if page else
                   f"{label(mo)}  (major {maj:.0f}%, minor {mnr:.0f}%)")
            ax.set_title(ttl, fontsize=title_fs, color=col(mo), pad=3)
            ax.set_xticks([x + 0.5 for x in xs])
            if r == nrow - 1 or k + ncol >= len(mods):
                ax.set_xticklabels([ERR_SHORT[e] for e in ITEMS], rotation=60, ha="right", rotation_mode="anchor",
                                   fontsize=lab_fs)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
            ax.set_yticks([yy + 0.5 for yy in ys])
            if c == 0:
                labs = [NAME["human"]] + [trial_names.get(t, t) for t in tids]
                ax.set_yticklabels(labs, fontsize=lab_fs)
                ax.get_yticklabels()[0].set_fontweight("bold")
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        fig.canvas.draw()
        right = max(a.get_position().x1 for a in fig.axes)
        cax = fig.add_axes([right + 0.015, 0.45, 0.008, 0.10])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 100), cmap=CMAP), cax=cax, ticks=[0, 50, 100])
        cb.set_label("Detection (%)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5, width=0.5, length=2); cb.outline.set_linewidth(0.5)
        if missing:
            fig.text(right + 0.015, 0.43, "grey: review missing", fontsize=6, color=INK2, va="top")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"{fname}.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # all models in one figure
    draw(models, name, 4 if len(models) > 9 else 3, 2.3, 5.6, 7.5)
    # the same split into figures of four models each (one row per figure, larger labels)
    if len(models) > 4:
        for k in range(0, len(models), 4):
            draw(models[k:k + 4], f"{name}_part{k // 4 + 1}", 4, 2.6, 6.5, 8, page=(9.7, 6.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-judges", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(EXT, "figures"))
    ap.add_argument("--human-per-error", default=os.path.join(SD1, "data_original", "results.json"))
    ap.add_argument("--with-original", action="store_true",
                    help="add the three 2026 trials of the main study (first review per model and trial)")
    ap.add_argument("--orig-root", default=SD1, help="folder holding the main study's data/ (Supplementary Data 1)")
    ap.add_argument("--models", nargs="*", help="restrict to these reviewer models")
    ap.add_argument("--trial-set", default="new", choices=["new", "rerun", "all"],
                    help="new: the additional trials; rerun: the main study's N, I, M rerun through the same API pipeline")
    ap.add_argument("--set", dest="tset", choices=["recent", "phase3"],
                    help="recent: T01-T27 (2026 trials of the first screen); phase3: P01-P36 (phase 3 since mid July)")
    ap.add_argument("--complete-only", action="store_true",
                    help="keep only trials for which every listed/active model has all reviews and controls scored")
    ap.add_argument("--tag", help="override the output tag")
    ap.add_argument("--exclude", nargs="*", default=[], help="trial ids to leave out (sensitivity analyses)")
    ap.add_argument("--exclude-noted", action="store_true",
                    help="leave out trials whose trials.json note flags caveats or a secondary report")
    a = ap.parse_args()
    man, scores = load(a.min_judges)
    tinfo = {t["tid"]: t for t in json.load(open(os.path.join(EXT, "trials.json")))["selected"]}
    man = [m for m in man if not tinfo.get(m["tid"], {}).get("excluded")]
    drop = set(a.exclude)
    if a.exclude_noted:
        drop |= {t for t, x in tinfo.items() if any(w in x.get("note", "") for w in ("caveat", "secondary report", "weak M4"))}
    if drop:
        print("excluding:", sorted(drop))
        man = [m for m in man if m["tid"] not in drop]
    if a.tset:
        want = "phase3_since_mid_july"
        man = [m for m in man if (tinfo.get(m["tid"], {}).get("set") == want) == (a.tset == "phase3")
               and tinfo.get(m["tid"], {}).get("source") != "main_study"]
    if a.complete_only:
        act = a.models or json.load(open(os.path.join(EXT, "runs", "active_models.json")))
        from collections import defaultdict
        need, have = defaultdict(int), defaultdict(int)
        for m in man:
            if m["model"] in act:
                need[m["tid"]] += 1
                have[m["tid"]] += m["id"] in scores
        done = {t for t in need if have[t] == need[t]}
        man = [m for m in man if m["tid"] in done]
    rerun = {t["tid"] for t in json.load(open(os.path.join(EXT, "trials.json")))["selected"] if t.get("source") == "main_study"}
    if a.trial_set == "new":
        man = [m for m in man if m["tid"] not in rerun]
    elif a.trial_set == "rerun":
        man = [m for m in man if m["tid"] in rerun]
    scores = {k: v for k, v in scores.items() if k in {m["id"] for m in man}}
    tag = {"new": "ext27", "rerun": "rerun_NIM", "all": "ext27_plus_rerun"}[a.trial_set]
    if a.tset:
        tag = {"recent": "recent2026", "phase3": "phase3"}[a.tset]
    if a.tag:
        tag = a.tag
    if a.with_original:
        m2, s2 = load_original(a.orig_root, a.min_judges, all_reps=True)
        man, scores = collapse_reps(man + m2, {**scores, **s2})
        tag = a.tag or "with_original"
        ntr = collections.Counter(m["model"] for m in man if m["kind"] == "review")
        top = max(ntr.values())
        for mod, n in ntr.items():
            if n < top:
                SUFFIX[mod] = " \u2020"
    if a.models:
        man = [m for m in man if m["model"] in a.models]
        scores = {k: v for k, v in scores.items() if k in {m["id"] for m in man}}
    res = summarize(man, scores)
    order = sorted([m for m in res if "review" in res[m]],
                   key=lambda m: res[m]["review"]["major_mean"] + res[m]["review"]["minor_mean"])
    hum = {}
    if os.path.exists(a.human_per_error):
        try:
            H = json.load(open(a.human_per_error))["human_per_paper_error_pct"]
            hum = {e: float(np.mean([H[p][e] for p in ("1", "2", "3")])) for e in ITEMS}
        except Exception:
            hum = {}
    trials = json.load(open(os.path.join(EXT, "trials.json")))["selected"]
    names = {t["tid"]: t["short"].split(" (")[0][:38] for t in trials}
    n_trials = len({m["tid"] for m in man if m["id"] in scores})
    rho = None
    xs = [(res[m]["review"]["sens"], res[m]["control"]["spec"]) for m in order if "control" in res[m]]
    if len(xs) >= 4:
        r1 = np.argsort(np.argsort([x for x, _ in xs])); r2 = np.argsort(np.argsort([y for _, y in xs]))
        rho = float(np.corrcoef(r1, r2)[0, 1])
    out = {"n_trials": n_trials, "n_scored_reviews": len(scores), "models": res, "order": order,
           "per_trial_total": per_trial(man, scores), "judge_agreement": agreement(scores),
           "spearman_sens_spec": rho}
    json.dump(out, open(os.path.join(DATA, f"results_{tag}.json"), "w"), indent=1)
    names.update({"N2026": "Tenecteplase (main study)", "I2026": "Guselkumab (main study)", "M2026": "Polypill (main study)"})
    lab_path = os.path.join(DATA, "trial_labels.json")
    if os.path.exists(lab_path):
        names.update(json.load(open(lab_path)))
    figure(res, order, a.out, n_trials, hum, name=f"Figure2_{tag}")
    by_trial_figure(man, scores, order, a.out, names, name=f"SupplFigure_{tag}_by_trial")

    def set_of(t):
        if t in rerun or t in ("N2026", "I2026", "M2026"):
            return "main"
        return "phase3" if tinfo.get(t, {}).get("set") == "phase3_since_mid_july" else "recent"
    per_model_heatmaps(man, scores, order, a.out, names, hum, set_of, name=f"SupplFigure_{tag}_per_model_heatmaps")
    lines = [f"# Additional 2026 trials: {n_trials} trials, {len(scores)} scored reviews", "",
             "| Model | Major (of 9) | Minor (of 5) | Total (of 14) | Sensitivity % | Specificity % | n reviews | n controls |",
             "|---|---|---|---|---|---|---|---|"]
    for m in reversed(order):
        r = res[m]["review"]; c = res[m].get("control", {})
        lines.append(f"| {NAME[m]} | {r['major_mean']:.2f} ({r['major_ci'][0]:.2f}-{r['major_ci'][1]:.2f}) | "
                     f"{r['minor_mean']:.2f} | {r['total_mean']:.2f} | {r['sens']:.1f} ({r['sens_ci'][0]:.1f}-{r['sens_ci'][1]:.1f}) | "
                     + (f"{c['spec']:.1f} ({c['spec_ci'][0]:.1f}-{c['spec_ci'][1]:.1f})" if c else "n/a")
                     + f" | {r['n_reviews']} | {c.get('n_reviews', 0)} |")
    lines += ["", f"Human reviewers 2008 (untrained): major 2.40, minor 0.89, total 3.29 of 14; detection about 27%.",
              f"Judge agreement: {out['judge_agreement']}", f"Spearman rho, sensitivity vs specificity: {rho}"]
    open(os.path.join(DATA, f"results_{tag}_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
