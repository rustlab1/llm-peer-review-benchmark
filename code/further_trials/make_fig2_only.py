"""Redraw Figure 2 from results_fig2_combined.json (no new bootstrap)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_ext as A
d = json.load(open(os.path.join(A.DATA, "results_fig2_combined.json")))
res, order = d["models"], d["order"]
for m in order:
    if res[m]["review"]["n_trials"] <= 3:
        A.SUFFIX[m] = " †"
H = json.load(open(os.path.join(A.SD1, "data_original", "results.json")))["human_per_paper_error_pct"]
hum = {e: float(np.mean([H[p][e] for p in ("1", "2", "3")])) for e in A.ITEMS}
A.figure(res, order, os.path.join(A.EXT, "figures"), d["n_trials"], hum, name="Figure2_fig2_combined")
print("ok")
