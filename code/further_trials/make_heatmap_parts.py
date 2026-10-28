"""Redraw only the per-model heatmaps (Supplementary Figs 4 to 6) without rerunning the bootstrap.
Data preparation as in analyze_ext.py --trial-set new --with-original --models <12 models>."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_ext as A
ROOT = A.SD1
ACT = ["haiku45", "sonnet5", "gpt54mini", "gpt54", "gpt55", "gpt61sol", "gpt6luna", "gemini38fl", "kimik3", "dsv4pro",
       "qwen38_27b", "gptoss120b"]
man, scores = A.load(2)
tinfo = {t["tid"]: t for t in json.load(open(os.path.join(A.EXT, "trials.json")))["selected"]}
man = [m for m in man if not tinfo.get(m["tid"], {}).get("excluded")]
rerun = {t for t, v in tinfo.items() if v.get("source") == "main_study"}
man = [m for m in man if m["tid"] not in rerun]
m2, s2 = A.load_original(ROOT, 2, all_reps=True)
man, scores = A.collapse_reps(man + m2, {**scores, **s2})
man = [m for m in man if m["model"] in ACT]
scores = {k: v for k, v in scores.items() if k in {m["id"] for m in man}}
order = json.load(open(os.path.join(A.DATA, "results_suppl_12models.json")))["order"]
H = json.load(open(os.path.join(ROOT, "data_original", "results.json")))["human_per_paper_error_pct"]
hum = {e: float(np.mean([H[p][e] for p in ("1", "2", "3")])) for e in A.ITEMS}
names = {t["tid"]: t["short"].split(" (")[0][:38] for t in tinfo.values()}
names.update(json.load(open(os.path.join(A.DATA, "trial_labels.json"))))
def set_of(t):
    if t in rerun or t in ("N2026", "I2026", "M2026"):
        return "main"
    return "further"          # the 55 further trials as one block (not split by search), then papers N, I and M
A.per_model_heatmaps(man, scores, order, os.path.join(A.EXT, "figures"), names, hum, set_of,
                     name="SupplFigure_suppl_12models_per_model_heatmaps")
print("ok")
