"""Figure 1 of the current manuscript: the 2004 test papers.

(a) design schematic (schematic_v3.py, layout C: four light grey cards with simple icons), (b) % of major and minor
inserted errors found per model, (c) detection of each error type (purple-white-green scale of Figure 2b). Slate,
proprietary models; rose, open-weight models.

Reads  $SD1/data_original/results.json and $SD1/data/results_v2.json (SD1 = extracted Supplementary Data 1)
Writes $FT_WORK/figures/Figure1_v2.png and .pdf
Run:   python code/further_trials/make_figure1_v2.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_ext as A  # noqa: E402  (shared colours, heatmap tiles and helpers)

ROOT = sys.argv[1] if len(sys.argv) > 1 else A.SD1
OUT = os.path.join(A.EXT, "figures")
OUT_DIR = os.environ.get("FIG1_OUT", OUT)
OUT_NAME = os.environ.get("FIG1_NAME", "Figure1_v2")
R1 = json.load(open(os.path.join(ROOT, "data_original", "results.json")))
V2 = json.load(open(os.path.join(ROOT, "data", "results_v2.json")))
S1, X1 = R1["summary"], V2["exp1"]
KEY = {"haiku": "haiku45", "sonnet": "sonnet5", "opus": "opus5", "fable": "fable5"}   # main study -> shared names
MODELS = list(X1)
ORDER = sorted(MODELS, key=lambda m: S1[m]["major_mean"] + S1[m]["minor_mean"])          # fewest to most found
H1 = R1["human_per_paper_error_pct"]
HUM_PER_ERR = {e: float(np.mean([H1[p][e] for p in ("1", "2", "3")])) for e in A.ITEMS}
nm = lambda m: A.NAME[KEY.get(m, m)]
cl = lambda m: A.col(KEY.get(m, m))

VAR = os.environ.get("FIG1A") or "C"   # A, B or C: schematic_v3 layouts; C is the published figure
import schematic_v3
DRAW_A, A_H = schematic_v3.DRAW[VAR]
B_H, GAP, TOP, BOT = 4.67, 0.40, 0.985, 0.17     # inches for panels b/c and the gap under a
fig = plt.figure(figsize=(7.2, (A_H + GAP + B_H) / (TOP - BOT)))
gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.55], height_ratios=[A_H, B_H], hspace=GAP / ((A_H + B_H) / 2),
                      wspace=0.05, top=TOP, bottom=BOT)
LAB_Y = 1.045
groups = ["human"] + ORDER

# b: % of major (filled) and minor (open) errors found
ax = fig.add_subplot(gs[1, 0])
n = len(groups)
ax.set_ylim(n - 0.5, -0.5); ax.set_yticks(range(n))
ax.set_yticklabels([A.NAME["human"]] + [nm(m) for m in ORDER]); ax.tick_params(axis="y", length=0)
ax.axhline(0.5, color=A.RULE, lw=0.6, ls=(0, (2, 2)))
A.zebra(ax, n)
for i, g in enumerate(groups):
    if g == "human":
        c = A.GREY
        maj, (mlo, mhi) = A.HUM_MAJ / 9 * 100, [v / 9 * 100 for v in A.HUM_MAJ_RANGE]
        mi, (ilo, ihi) = A.HUM_MIN / 5 * 100, [v / 5 * 100 for v in A.HUM_MIN_RANGE]
    else:
        c, r = cl(g), X1[g]
        maj, (mlo, mhi) = r["major_mean"] / 9 * 100, [v / 9 * 100 for v in r["major_ci_clustered"]]
        mi, (ilo, ihi) = r["minor_mean"] / 5 * 100, [v / 5 * 100 for v in r["minor_ci_clustered"]]
    ax.plot([mlo, mhi], [i - 0.15] * 2, color=c, lw=1.3, alpha=0.55, solid_capstyle="round", zorder=2)
    ax.plot([ilo, ihi], [i + 0.15] * 2, color=c, lw=1.3, alpha=0.55, solid_capstyle="round", zorder=2)
    ax.scatter(maj, i - 0.15, s=34, color=c, edgecolors="white", linewidths=0.7, zorder=4)
    ax.scatter(mi, i + 0.15, s=34, facecolors="white", edgecolors=c, linewidths=1.1, zorder=4)
ax.set_xlim(-2.5, 102.5); ax.set_xticks([0, 25, 50, 75, 100]); ax.spines["bottom"].set_bounds(0, 100)
ax.set_xlabel("Inserted errors found (%)")
ax.grid(axis="x", color=A.RULE, lw=0.4, zorder=0); ax.spines["left"].set_visible(False)
for t, g in zip(ax.get_yticklabels(), groups):
    t.set_color(A.INK if g == "human" else cl(g))
    if g == "human":
        t.set_fontweight("bold")
h = [Line2D([], [], marker="o", ls="", color=A.INK2, mec="white", ms=5.5, label="major errors (of 9)"),
     Line2D([], [], marker="o", ls="", color=A.BLUE, mec="white", ms=5.5, label="proprietary model"),
     Line2D([], [], marker="o", ls="", mfc="white", mec=A.INK2, mew=1.1, ms=5.5, label="minor errors (of 5)"),
     Line2D([], [], marker="o", ls="", color=A.RED, mec="white", ms=5.5, label="open-weight model")]
ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.66, -0.075), ncol=2, borderaxespad=0, handletextpad=0.3,
          labelspacing=0.25, columnspacing=0.9)
ax.text(-0.62, LAB_Y, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")
ax_b = ax

# c: detection of each error type, tiles as in Figure 2b
ax = fig.add_subplot(gs[1, 1])
M = np.array([[HUM_PER_ERR[e] for e in A.ITEMS]] + [[S1[m]["per_error_pct"][e] for e in A.ITEMS] for m in ORDER])
xs = [j - 0.5 + (0.3 if j >= 9 else 0) for j in range(14)]
ys = [i - 0.5 for i in range(M.shape[0])]
A.tile_heatmap(ax, M, xs, ys, A.CMAP, pad=0.045)
ax.set_ylim(n - 0.5, -0.5); ax.set_yticks(range(n)); ax.set_yticklabels([""] * n); ax.tick_params(axis="y", length=0)
ax.set_xlim(xs[0] - 0.05, xs[-1] + 1.05)
ax.set_xticks([x + 0.5 for x in xs])
ax.set_xticklabels([A.ERR_SHORT[e] for e in A.ITEMS], rotation=45, ha="right", rotation_mode="anchor", fontsize=7.5)
ax.tick_params(axis="x", length=0)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(xs[j] + 0.5, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.8, color=A.text_on(M[i, j]))
ax.axhline(0.5, color="white", lw=2.4)
for s in ax.spines.values():
    s.set_visible(False)
ax.text(np.mean([x + 0.5 for x in xs[:9]]), -0.95, "major errors", ha="center", va="bottom", fontsize=7.5, color=A.INK2)
ax.text(np.mean([x + 0.5 for x in xs[9:]]), -0.95, "minor errors", ha="center", va="bottom", fontsize=7.5, color=A.INK2)
ax.text(-0.03, LAB_Y, "c", transform=ax.transAxes, fontsize=10, fontweight="bold")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
fig.canvas.draw()
pc = ax.get_position()
cax = fig.add_axes([pc.x1 + 0.012, pc.y0 + 0.30 * pc.height, 0.010, 0.40 * pc.height])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 100), cmap=A.CMAP), cax=cax, ticks=[0, 50, 100])
cb.set_label("Detection (%)", fontsize=7.5); cb.ax.tick_params(labelsize=7, width=0.5, length=2)
cb.outline.set_linewidth(0.5)

# a: design schematic across the full width (schematic_v3.py; drawn in inch units so shapes keep their proportions)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
left = ax_b.get_tightbbox(rend).transformed(fig.transFigure.inverted()).x0
right = cax.get_tightbbox(rend).transformed(fig.transFigure.inverted()).x1
ax_a = fig.add_subplot(gs[0, :])
pa = ax_a.get_position()
ax_a.set_position([left, pa.y0, right - left, pa.height])
fw, fh = fig.get_size_inches()
print('schematic size (in):', round((right - left) * fw, 2), round(pa.height * fh, 2))
DRAW_A(ax_a, W=(right - left) * fw, H=pa.height * fh)
pb = ax_b.get_position()
fig.text(pb.x0 - 0.62 * pb.width, ax_a.get_position().y1 + 0.006, "a", fontsize=10, fontweight="bold",
         va="bottom", ha="left")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"{OUT_NAME}.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(OUT_DIR, f"{OUT_NAME}.png"))
