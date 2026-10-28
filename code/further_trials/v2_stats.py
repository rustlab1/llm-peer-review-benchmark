"""Numbers in the text of the manuscript: the 58 trials (papers N, I, M of the main study, four reviews per model averaged
per trial, plus the 55 additional trials with one review per model). Run after run_final_analysis.sh.
Writes data/v2_stats.json and prints a readable summary."""
import collections
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_ext as A  # noqa: E402

EXT, DATA = A.EXT, A.DATA
ORIG_ROOT = A.SD1
ITEMS, MAJOR, MINOR = A.ITEMS, A.MAJOR, A.MINOR
HUM_TOTAL = 3.29
rng = np.random.default_rng(20261027)

# ---------------------------------------------------------------- data, as in analyze_ext --with-original
man, scores = A.load(2)
tinfo = {t["tid"]: t for t in json.load(open(os.path.join(EXT, "trials.json")))["selected"]}
man = [m for m in man if not tinfo.get(m["tid"], {}).get("excluded")]
rerun = {t for t, v in tinfo.items() if v.get("source") == "main_study"}
man = [m for m in man if m["tid"] not in rerun and m["id"] in scores]
m2, s2 = A.load_original(ORIG_ROOT, 2, all_reps=True)
man_u, sc_u = man + m2, {**scores, **s2}                    # every individual review
man_c, sc_c = A.collapse_reps(man_u, sc_u)                  # one record per model, trial and kind
models = sorted({m["model"] for m in man_c})
R = json.load(open(os.path.join(DATA, "results_fig2_combined.json")))
res, order = R["models"], R["order"]
out = {}

# ---------------------------------------------------------------- 1. lowest individual review
tot_u = collections.defaultdict(list)
for m in man_u:
    if m["kind"] == "review":
        tot_u[m["model"]].append(sum(sc_u[m["id"]][i] for i in ITEMS))
out["min_review_total"] = {mo: float(min(v)) for mo, v in tot_u.items()}
out["n_reviews_below_human_mean"] = {mo: int(sum(x < HUM_TOTAL for x in v)) for mo, v in tot_u.items()}
out["n_reviews_total"] = {mo: len(v) for mo, v in tot_u.items()}

# ---------------------------------------------------------------- 2. per-trial totals and paired permutation tests
pt = collections.defaultdict(dict)
for m in man_c:
    if m["kind"] == "review":
        pt[m["model"]][m["tid"]] = sum(sc_c[m["id"]][i] for i in ITEMS)


def perm(a, b, n=20000):
    """Two-sided paired permutation test: model labels shuffled within trial (sign flips of the differences)."""
    tids = sorted(set(pt[a]) & set(pt[b]))
    d = np.array([pt[a][t] - pt[b][t] for t in tids])
    obs = d.mean()
    flips = rng.choice([-1, 1], size=(n, len(d)))
    null = (flips * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n + 1)
    return {"a": a, "b": b, "n_trials": len(tids), "diff": float(obs), "p": float(p)}


pairs = [("gpt61sol", "gpt55"), ("kimik3", "gpt61sol"), ("sonnet5", "haiku45"), ("gpt55", "gpt54"),
         ("gpt61sol", "sonnet5"), ("gpt6luna", "gpt61sol"), ("gpt61sol", "opus55"), ("kimik3", "gpt55"),
         ("dsv4pro", "kimik3"), ("gpt54", "gpt54mini")]
out["perm"] = [perm(a, b) for a, b in pairs]
out["per_trial_sd"] = {mo: float(np.std(list(v.values()), ddof=1)) for mo, v in pt.items() if len(v) > 3}
out["per_trial_range"] = {mo: [float(min(v.values())), float(max(v.values()))] for mo, v in pt.items() if len(v) > 3}

# ---------------------------------------------------------------- 3. control manuscripts: flagged items
flag = collections.Counter()
n_ctl = 0
flag_by_model = collections.defaultdict(int)
for m in man_c:
    if m["kind"] == "control":
        n_ctl += 1
        s = sc_c[m["id"]]
        for i in ITEMS:
            f = s["_det"][i] if "_det" in s else float(s[i] >= 0.5)
            flag[i] += f
            flag_by_model[m["model"]] += f
out["controls"] = {"n_reviews": n_ctl, "n_items": n_ctl * 14, "flagged": float(sum(flag.values())),
                   "flagged_by_error": {i: float(flag[i]) for i in ITEMS},
                   "spec_overall_pct": 100 * (1 - sum(flag.values()) / (n_ctl * 14))}
top5 = sorted(ITEMS, key=lambda i: -flag[i])[:5]
out["controls"]["top5"] = top5
out["controls"]["top5_share"] = float(sum(flag[i] for i in top5))

# ---------------------------------------------------------------- 4. sensitivity vs specificity
sens = [res[m]["review"]["sens"] for m in order]
spec = [res[m]["control"]["spec"] for m in order]
rho, p = spearmanr(sens, spec)
out["spearman_sens_spec_15"] = {"rho": float(rho), "p": float(p)}
o12 = [m for m in order if res[m]["review"]["n_trials"] > 3]
rho, p = spearmanr([res[m]["review"]["sens"] for m in o12], [res[m]["control"]["spec"] for m in o12])
out["spearman_sens_spec_12"] = {"rho": float(rho), "p": float(p)}

# ---------------------------------------------------------------- 5. the same models on the 2004 test papers
R1 = json.load(open(os.path.join(ORIG_ROOT, "data_original", "results.json")))
S1 = R1["summary"]
key1 = {"haiku45": "haiku", "sonnet5": "sonnet", "opus5": "opus", "fable5": "fable"}
cmp = {}
for mo in order:
    k = key1.get(mo, mo)
    t04 = S1[k]["major_mean"] + S1[k]["minor_mean"]
    t26 = res[mo]["review"]["total_mean"]
    cmp[mo] = {"total_2004": t04, "total_2026": t26, "diff": t26 - t04,
               "per_error_2004": S1[k]["per_error_pct"]}
out["vs_2004"] = cmp

# ---------------------------------------------------------------- 6. per-error detection on the 58 trials
pe = {mo: res[mo]["review"]["per_error_pct"] for mo in order}
out["per_error"] = pe
out["per_error_mean_over_models"] = {i: float(np.mean([pe[m][i] for m in order])) for i in ITEMS}
out["control_per_error_mean_over_models"] = {i: float(np.mean([res[m]["control"]["per_error_pct"][i] for m in order]))
                                             for i in ITEMS}

# ---------------------------------------------------------------- 7. subsets: recent, phase 3, clean
sub = {}
for tag in ("recent2026", "phase3", "new_trials", "new_trials_sens_clean", "rerun_NIM"):
    p = os.path.join(DATA, f"results_{tag}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        sub[tag] = {"n_trials": d["n_trials"],
                    "total": {m: d["models"][m]["review"]["total_mean"] for m in d["models"] if "review" in d["models"][m]},
                    "sens": {m: d["models"][m]["review"]["sens"] for m in d["models"] if "review" in d["models"][m]},
                    "spec": {m: d["models"][m]["control"]["spec"] for m in d["models"] if "control" in d["models"][m]},
                    "judge_agreement": d.get("judge_agreement")}
out["subsets"] = sub
if "recent2026" in sub and "phase3" in sub:
    ms = sorted(set(sub["recent2026"]["total"]) & set(sub["phase3"]["total"]))
    a = [sub["recent2026"]["total"][m] for m in ms]; b = [sub["phase3"]["total"][m] for m in ms]
    out["recent_vs_phase3"] = {"rho": float(spearmanr(a, b)[0]), "max_abs_diff": float(max(abs(x - y) for x, y in zip(a, b))),
                               "mean_diff_phase3_minus_recent": float(np.mean(np.array(b) - np.array(a)))}
if "new_trials" in sub and "new_trials_sens_clean" in sub:
    ms = sorted(sub["new_trials"]["total"])
    a = [sub["new_trials"]["total"][m] for m in ms]; b = [sub["new_trials_sens_clean"]["total"][m] for m in ms]
    out["all55_vs_clean"] = {"rho": float(spearmanr(a, b)[0]), "max_abs_diff": float(max(abs(x - y) for x, y in zip(a, b)))}

json.dump(out, open(os.path.join(DATA, "v2_stats.json"), "w"), indent=1)

# ---------------------------------------------------------------- readable summary
N = A.NAME
print("model | n | major (CI) | minor | total (CI) | major% | sens (CI) | spec (CI) | min review | <3.29")
for m in reversed(order):
    r, c = res[m]["review"], res[m]["control"]
    print(f"{N[m]:16s} {r['n_trials']:2d} | {r['major_mean']:.2f} ({r['major_ci'][0]:.2f}-{r['major_ci'][1]:.2f}) | "
          f"{r['minor_mean']:.2f} ({r['minor_ci'][0]:.2f}-{r['minor_ci'][1]:.2f}) | {r['total_mean']:.2f} "
          f"({r['total_ci'][0]:.2f}-{r['total_ci'][1]:.2f}) | {100*r['major_mean']/9:.1f} | {r['sens']:.1f} "
          f"({r['sens_ci'][0]:.1f}-{r['sens_ci'][1]:.1f}) | {c['spec']:.1f} ({c['spec_ci'][0]:.1f}-{c['spec_ci'][1]:.1f}) | "
          f"{out['min_review_total'][m]:.2f} | {out['n_reviews_below_human_mean'][m]}/{out['n_reviews_total'][m]}")
print("perm:", *[f"{N[x['a']]} - {N[x['b']]}: {x['diff']:+.2f} (n={x['n_trials']}, P={x['p']:.4f})" for x in out["perm"]], sep="\n  ")
print("controls:", {k: v for k, v in out["controls"].items() if k != "flagged_by_error"})
print("  flagged by error:", {i: round(v, 2) for i, v in out["controls"]["flagged_by_error"].items()})
print("spearman 15:", out["spearman_sens_spec_15"], " 12:", out["spearman_sens_spec_12"])
print("vs 2004:", {N[m]: round(v["diff"], 2) for m, v in cmp.items()})
print("subsets:", out.get("recent_vs_phase3"), out.get("all55_vs_clean"))
print("per-trial SD:", {N[m]: round(v, 2) for m, v in out["per_trial_sd"].items()})
print("per-trial range:", {N[m]: v for m, v in out["per_trial_range"].items()})
