"""Generate Supplementary Figure 1 for the current manuscript.

Panels a–c compare the original judges, Fable 5 and Opus 5, with Kimi K3 and GPT-6.1 Sol.
Panel d compares Kimi K3 scores under original and reworded instructions.
Inputs are under data/, data_original/ and additional_judges/ in Supplementary Data 1.
Writes figures/SupplFigure1.png, SupplFigure1.pdf and SupplFigure1_facts.json.
Run: python code/make_judge_figure.py
"""
import json, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = os.environ.get("BMJ_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("BMJ_FIG_OUT", os.path.join(ROOT, "figures"))
os.makedirs(OUT, exist_ok=True)
ERR = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "m1", "m2", "m3", "m4", "m5"]
MODELS = ["haiku", "sonnet", "opus", "fable", "gpt54mini", "gpt54", "gpt55",
          "opus55", "gpt61sol", "gpt6luna", "gemini38fl", "kimik3", "dsv4pro", "qwen38_27b", "gptoss120b"]
OPEN = {"kimik3", "dsv4pro", "qwen38_27b", "gptoss120b"}
NAME = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 5", "opus": "Opus 5", "fable": "Fable 5", "gpt54mini": "GPT-5.4 mini",
        "gpt54": "GPT-5.4", "gpt55": "GPT-5.5", "opus55": "Opus 5.5", "gpt61sol": "GPT-6.1 Sol", "gpt6luna": "GPT-6 Luna",
        "gemini38fl": "Gemini 3.8 Flash", "kimik3": "Kimi K3", "dsv4pro": "DeepSeek V4 Pro", "qwen38_27b": "Qwen3.8-27B",
        "gptoss120b": "gpt-oss-120b"}
BLUE, RED = "#4F6375", "#C47885"   # slate (proprietary), rose (open-weight)
INK, INK2, RULE = "#1A1A1A", "#4D4D4D", "#D9D9D9"
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.direction": "out", "ytick.direction": "out", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "figure.dpi": 100,
                     "legend.frameon": False, "legend.handletextpad": 0.4})


def col(m):
    return RED if m in OPEN else BLUE


def load(p):
    return {s["id"]: float(s["score"]) for s in json.load(open(p))["scores"]}


R1 = json.load(open(os.path.join(ROOT, "data_original", "results.json")))
R2 = json.load(open(os.path.join(ROOT, "data", "results.json")))
SETS = [("2004 test papers", R1["rows"], "data_original/judgements"),
        ("2026 trials", R2["rows"], "data/judgements"),
        ("Unmodified 2026 trials", R2["control_rows"], "data/controls/judgements")]
J3DIR = os.path.join(ROOT, "additional_judges/kimi_k3", "kimik3_main", "judgements")
J3RDIR = os.path.join(ROOT, "additional_judges/kimi_k3", "kimik3_reworded", "judgements")
J4 = json.load(open(os.path.join(ROOT, "additional_judges/gpt61_sol", "gpt61sol_scores.json")))["scores"]
J4SET = {"2004 test papers": "original", "2026 trials": "2026_seeded", "Unmodified 2026 trials": "2026_controls"}
D = {}
for name, rows, jdir in SETS:
    D[name] = [{"id": r["id"], "model": r["model"], "J1": load(os.path.join(ROOT, jdir, f"{r['id']}__J1.json")),
                "J2": load(os.path.join(ROOT, jdir, f"{r['id']}__J2.json")),
                "J3": load(os.path.join(J3DIR, f"{r['id']}__J3.json")),
                 "J4": J4[J4SET[name]][r["id"]]} for r in rows]


def agreement(pairs):
    exact = np.mean([a == b for a, b in pairs])
    a1 = np.array([a >= 0.5 for a, b in pairs]); b1 = np.array([b >= 0.5 for a, b in pairs])
    po = np.mean(a1 == b1); pa, pb = a1.mean(), b1.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
    return 100 * exact, (po - pe) / (1 - pe)


def total(r, js):
    return sum(st.mean(r[j][e] for j in js) for e in ERR)


fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.4), gridspec_kw={"hspace": 0.40, "wspace": 0.62})
facts = {}

# a: agreement
ax = axs[0, 0]
PAIRS = [("J1", "J2", "Fable 5 vs Opus 5", dict(marker="o", color=INK, mfc=INK)),
         ("J3", "J1", "Kimi K3 vs Fable 5", dict(marker="o", color=INK2, mfc="white")),
         ("J3", "J2", "Kimi K3 vs Opus 5", dict(marker="s", color=INK2, mfc="white")),
         ("J4", "J1", "GPT-6.1 Sol vs Fable 5", dict(marker="^", color=INK2, mfc=INK2)),
         ("J4", "J2", "GPT-6.1 Sol vs Opus 5", dict(marker="v", color=INK2, mfc=INK2))]
ys = {name: i for i, (name, _, _) in enumerate(SETS)}
facts["agreement"] = {}
for k, (a, b, lab, sty) in enumerate(PAIRS):
    for name, _, _ in SETS:
        ex, kap = agreement([(r[a][e], r[b][e]) for r in D[name] for e in ERR])
        facts["agreement"][f"{name} {lab}"] = (round(ex, 1), round(kap, 3))
        y = ys[name] + (k - 2) * 0.17
        ax.plot(kap, y, ls="", ms=5, mew=0.9, zorder=3, **sty)
        ax.text(1.02, y, f"{ex:.1f}%", va="center", ha="left", fontsize=6.5, color=INK2, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(SETS))); ax.set_yticklabels([n for n, _, _ in SETS])
ax.set_ylim(len(SETS) - 0.5, -0.5)
ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xlabel("Agreement between judges (Cohen's κ)")
ax.text(1.02, -0.62, "identical\nscores", ha="left", va="bottom", fontsize=7, color=INK2, transform=ax.get_yaxis_transform())
ax.legend(handles=[Line2D([], [], ls="", ms=5, mew=0.9, label=lab, **sty) for _, _, lab, sty in PAIRS],
          loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1, fontsize=7)
ax.grid(axis="x", color=RULE, lw=0.4, zorder=0)

# b: per-model mean summed score, original judges vs third judge
ax = axs[0, 1]
xs, ys2 = {}, {}
for name, filled in (("2004 test papers", False), ("2026 trials", True)):
    x = [st.mean(total(r, ["J1", "J2"]) for r in D[name] if r["model"] == m) for m in MODELS]
    for J, mk, tag in (("J3", "o", "Kimi K3"), ("J4", "^", "GPT-6.1 Sol")):
        y = [st.mean(total(r, [J]) for r in D[name] if r["model"] == m) for m in MODELS]
        for m, xi, yi in zip(MODELS, x, y):
            ax.scatter(xi, yi, s=18, marker=mk, zorder=3, linewidths=0.9,
                       facecolors=col(m) if filled else "white", edgecolors=col(m))
        facts[f"rank_rho {tag} {name}"] = round(spearmanr(x, y)[0], 3)
ax.plot([0, 14], [0, 14], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.set_xticks(range(0, 15, 2)); ax.set_yticks(range(0, 15, 2))
ax.set_aspect("equal")
ax.set_xlabel("Errors found per review (of 14),\noriginal judges")
ax.set_ylabel("Errors found per review (of 14),\nother judge")
ax.text(0.97, 0.04, "Kimi K3: ρ = %.3f (2026), %.3f (2004)\nGPT-6.1 Sol: ρ = %.3f (2026), %.3f (2004)" % (
        facts["rank_rho Kimi K3 2026 trials"], facts["rank_rho Kimi K3 2004 test papers"],
        facts["rank_rho GPT-6.1 Sol 2026 trials"], facts["rank_rho GPT-6.1 Sol 2004 test papers"]),
        transform=ax.transAxes, va="bottom", ha="right", fontsize=6.5)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=5, label="Kimi K3"),
                   Line2D([], [], marker="^", ls="", color=INK2, ms=5, label="GPT-6.1 Sol"),
                   Line2D([], [], marker="s", ls="", color=INK2, ms=5, label="2026 trials"),
                   Line2D([], [], marker="s", ls="", color=INK2, mfc="white", ms=5, label="2004 test papers"),
                   Line2D([], [], marker="s", ls="", color=BLUE, ms=5, label="proprietary model"),
                   Line2D([], [], marker="s", ls="", color=RED, ms=5, label="open-weight model")],
          loc="upper left", fontsize=6.5, handletextpad=0.2, borderaxespad=0.1)

# c: per-model difference, third judge minus original judges, ordered by errors found on the 2026 trials (worst first)
ax = axs[1, 0]
order = sorted(MODELS, key=lambda m: st.mean(total(r, ["J1", "J2"]) for r in D["2026 trials"] if r["model"] == m))
rng = np.random.default_rng(15)
facts["diff_by_model"] = {}
for i, m in enumerate(order):
    for J, mk, off, tag in (("J3", "o", -0.18, "Kimi K3"), ("J4", "^", 0.18, "GPT-6.1 Sol")):
        d = np.array([total(r, [J]) - total(r, ["J1", "J2"]) for name in ("2004 test papers", "2026 trials")
                      for r in D[name] if r["model"] == m])
        boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        facts["diff_by_model"].setdefault(tag, {})[m] = (round(d.mean(), 2), round(lo, 2), round(hi, 2))
        ax.plot([lo, hi], [i + off, i + off], color=col(m), lw=0.8, zorder=2)
        ax.scatter(d.mean(), i + off, s=16, marker=mk, color=col(m), edgecolors="white", linewidths=0.5, zorder=3)
ax.axvline(0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([NAME[m] + (" (judge)" if m in ("kimik3", "gpt61sol") else "") for m in order], fontsize=7.5)
ax.set_ylim(len(order) - 0.5, -0.5)
ax.set_xlim(-1.5, 1.5); ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
ax.set_xlabel("Other judge minus original judges\n(errors found per review, of 14;\ncircles Kimi K3, triangles GPT-6.1 Sol)")
ax.grid(axis="x", color=RULE, lw=0.4, zorder=0)

# d: reworded vs original instructions for the third judge
ax = axs[1, 1]
ids = [f[:-len("__J3.json")] for f in os.listdir(J3RDIR)]
byid = {r["id"]: r for name in D for r in D[name]}
pairs, pts = [], []
for i in ids:
    a = load(os.path.join(J3DIR, f"{i}__J3.json")); b = load(os.path.join(J3RDIR, f"{i}__J3.json"))
    pairs += [(b[e], a[e]) for e in ERR]
    pts.append((sum(a.values()), sum(b.values()), "control" in i, byid[i]["model"]))
ex, kap = agreement(pairs)
facts["reworded"] = (len(ids), round(ex, 1), round(kap, 3))
jit = np.random.default_rng(3)
for x, y, ctl, m in pts:
    jx, jy = jit.uniform(-0.12, 0.12, 2)
    ax.scatter(x + jx, y + jy, s=16, zorder=3, linewidths=0.8,
               facecolors="white" if ctl else col(m), edgecolors=col(m), alpha=0.9)
ax.plot([0, 14], [0, 14], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.set_xticks(range(0, 15, 2)); ax.set_yticks(range(0, 15, 2))
ax.set_aspect("equal")
ax.set_xlabel("Score per review (of 14),\noriginal instructions")
ax.set_ylabel("Score per review (of 14),\nreworded instructions")
ax.text(0.04, 0.96, f"{ex:.1f}% identical items\nκ = {kap:.2f}, n = {len(ids)} reviews", transform=ax.transAxes,
        va="top", fontsize=7.5)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=5, label="error-seeded (45)"),
                   Line2D([], [], marker="o", ls="", color=INK2, mfc="white", ms=5, label="unmodified (45)")],
          loc="lower right", fontsize=7, handletextpad=0.2, borderaxespad=0.1)

fig.canvas.draw()
for ax, s in zip(axs.flat, "abcd"):
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    pos = ax.get_position()
    fig.text(bb.x0 - 0.005, pos.y1 + 0.025, s, fontsize=10, fontweight="bold", va="bottom", ha="left")
for a in fig.axes:
    for art in list(a.lines) + list(a.collections):
        art.set_clip_on(False)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"SupplFigure1.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
json.dump(facts, open(os.path.join(OUT, "SupplFigure1_facts.json"), "w"), indent=1)
print(json.dumps(facts, indent=1))
