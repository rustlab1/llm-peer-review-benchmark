"""Compare each reviewer model across three settings:
  paper  = main study, papers N/I/M, original harness (results_v2.json exp2; 4 reviews per paper)
  rerun  = the same N/I/M manuscripts rerun through this API pipeline (1 review per paper)
  new    = the additional trials (2026 trials + phase 3 trials), API pipeline (1 review per trial)
Writes data/compare_paper_rerun_new.md/.json and figures/Compare_paper_rerun_new.png/.pdf.
Run after analyze_ext.py has produced results_<tag>.json for the rerun and new sets."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SD1 = os.environ.get("SD1", os.path.join(REPO, "Supplementary_Data_1"))
EXT = os.environ.get("FT_WORK", os.path.join(REPO, "work_further_trials"))
DATA, FIG = os.path.join(EXT, "data"), os.path.join(EXT, "figures")
PAPER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SD1, "data", "results_v2.json")
NEW_TAG = sys.argv[2] if len(sys.argv) > 2 else "ext27"
RERUN_TAG = sys.argv[3] if len(sys.argv) > 3 else "rerun_NIM"

PMAP = {"haiku45": "haiku", "sonnet5": "sonnet", "opus5": "opus", "fable5": "fable"}
NAME = {"haiku45": "Haiku 4.5", "sonnet5": "Sonnet 5", "gpt54mini": "GPT-5.4 mini", "gpt54": "GPT-5.4", "gpt55": "GPT-5.5",
        "gpt61sol": "GPT-6.1 Sol", "gpt6luna": "GPT-6 Luna", "gemini38fl": "Gemini 3.8 Flash", "kimik3": "Kimi K3",
        "dsv4pro": "DeepSeek V4 Pro", "qwen38_27b": "Qwen3.8-27B", "gptoss120b": "gpt-oss-120b"}
INK, INK2, RULE = "#1A1A1A", "#4D4D4D", "#D9D9D9"
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.5, "legend.frameon": False})

paper = json.load(open(PAPER))["exp2"]
new = json.load(open(os.path.join(DATA, f"results_{NEW_TAG}.json")))
rerun = json.load(open(os.path.join(DATA, f"results_{RERUN_TAG}.json")))
models = [m for m in new["order"] if m in NAME]
rows = []
for m in models:
    p = paper.get(PMAP.get(m, m), {})
    r = rerun["models"].get(m, {}).get("review", {})
    n = new["models"][m]["review"]
    rows.append({"model": m, "name": NAME[m],
                 "paper_total": p.get("total_mean"), "paper_ci": p.get("total_ci_clustered"),
                 "paper_major": p.get("major_mean"), "paper_minor": p.get("minor_mean"),
                 "rerun_total": r.get("total_mean"), "rerun_major": r.get("major_mean"), "rerun_minor": r.get("minor_mean"),
                 "new_total": n["total_mean"], "new_ci": n["total_ci"],
                 "new_major": n["major_mean"], "new_minor": n["minor_mean"],
                 "new_spec": new["models"][m].get("control", {}).get("spec")})


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


ok = [r for r in rows if r["paper_total"] is not None]
rho_pn = spearman([r["paper_total"] for r in ok], [r["new_total"] for r in ok])
okr = [r for r in rows if r["rerun_total"] is not None and r["paper_total"] is not None]
rho_pr = spearman([r["paper_total"] for r in okr], [r["rerun_total"] for r in okr])
diff_pr = float(np.mean([r["rerun_total"] - r["paper_total"] for r in okr]))
diff_pn = float(np.mean([r["new_total"] - r["paper_total"] for r in ok]))
json.dump({"rows": rows, "spearman_paper_vs_new": rho_pn, "spearman_paper_vs_rerun": rho_pr,
           "mean_diff_rerun_minus_paper": diff_pr, "mean_diff_new_minus_paper": diff_pn,
           "n_new_trials": new["n_trials"]}, open(os.path.join(DATA, "compare_paper_rerun_new.json"), "w"), indent=1)

lines = ["# Paper (N, I, M; original harness) vs same trials rerun through the API pipeline vs additional trials", "",
         f"Additional trials: {new['n_trials']}. Errors found of 14 (mean).", "",
         "| Model | Paper N/I/M | Rerun N/I/M | Additional trials | Additional: major / minor |", "|---|---|---|---|---|"]
for r in rows:
    f = lambda v: "n/a" if v is None else f"{v:.1f}"
    lines.append(f"| {r['name']} | {f(r['paper_total'])} | {f(r['rerun_total'])} | {r['new_total']:.1f} | "
                 f"{r['new_major']:.1f} / {r['new_minor']:.1f} |")
lines += ["", f"Spearman rank correlation, paper vs additional trials: {rho_pn:.2f}; paper vs rerun: {rho_pr:.2f}.",
          f"Mean difference rerun minus paper: {diff_pr:+.2f}; additional minus paper: {diff_pn:+.2f} errors of 14."]
open(os.path.join(DATA, "compare_paper_rerun_new.md"), "w").write("\n".join(lines) + "\n")

fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(rows) + 1.0))
for i, r in enumerate(rows):
    vals = [v for v in (r["paper_total"], r["rerun_total"], r["new_total"]) if v is not None]
    ax.plot([min(vals), max(vals)], [i, i], color=RULE, lw=1.0, zorder=1)
    if r["paper_total"] is not None:
        ax.scatter(r["paper_total"], i, s=26, marker="s", facecolors="white", edgecolors=INK2, linewidths=0.9, zorder=3)
    if r["rerun_total"] is not None:
        ax.scatter(r["rerun_total"], i, s=24, marker="D", color="#9AA5B1", edgecolors="white", linewidths=0.5, zorder=3)
    ax.plot(r["new_ci"], [i, i], color="#4F6375", lw=0.8, zorder=2)
    ax.scatter(r["new_total"], i, s=30, color="#4F6375", edgecolors="white", linewidths=0.6, zorder=4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["name"] for r in rows]); ax.invert_yaxis()
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.axvline(3.29, color="#8A8A8A", lw=0.8, ls=(0, (3, 2)))
ax.text(3.45, len(rows) - 0.6, "human reviewers", fontsize=7, color=INK2, va="bottom")
ax.set_xlim(0, 14); ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14]); ax.set_xlabel("Inserted errors found per review (of 14)")
ax.grid(axis="x", color=RULE, lw=0.4, zorder=0)
ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc="white", mec=INK2, ms=5, label="papers N, I and M, main experiment"),
                   Line2D([], [], marker="D", ls="", color="#9AA5B1", ms=4.5, label="papers N, I and M, rerun as the further trials"),
                   Line2D([], [], marker="o", ls="-", color="#4F6375", ms=5, lw=0.8,
                          label=f"{new['n_trials']} further trials, with 95% CI")],
          loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7, borderaxespad=0.2)
os.makedirs(FIG, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"Compare_paper_rerun_new.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
print("\n".join(lines))
