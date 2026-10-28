"""Validate the further trials in Supplementary Data 1 (folder further_trials/).

In Supplementary Data 1: python further_trials/code/validate_further_trials.py
From this repository:    python code/further_trials/validate_further_trials.py /path/to/Supplementary_Data_1/further_trials
Checks
 1. Every included trial (55 further trials and papers N, I and M rerun as N26, I26, M26) has one error-seeded and one
    unmodified review from each of the 12 models run on these trials, apart from the one documented missing review.
 2. Every review has two judge outputs (J1 = Fable 5, J2 = Opus 5) with 14 item scores of 0, 0.5 or 1.
 3. The per-model results on the 55 further trials (mean major, minor and total score, sensitivity and specificity)
    recomputed from the judge outputs agree with results/results_new_trials.json.
 4. The counts stated in further_trials/README.md (reviews and judge outputs).
Writes further_trials/validation/further_trials_validation.json and exits with status 1 if any check fails.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

FT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
ITEMS = [f"M{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 6)]
MAJOR, MINOR = ITEMS[:9], ITEMS[9:]
MODELS12 = ["haiku45", "sonnet5", "gpt54mini", "gpt54", "gpt55", "gpt61sol", "gpt6luna", "gemini38fl", "kimik3",
            "dsv4pro", "qwen38_27b", "gptoss120b"]
KNOWN_MISSING = {("gptoss120b", "P37", "control")}
errors, report = [], {}

man = json.loads((FT / "manifest.json").read_text())
trials = list(csv.DictReader((FT / "trials" / "trials.csv").open()))
included = {t["trial_id"] for t in trials if t["status"] == "included"}
report["included_further_trials"] = len(included)
if len(included) != 55:
    errors.append(f"expected 55 included trials, found {len(included)}")
tids = included | {"N26", "I26", "M26"}

reviews = {p.stem for p in (FT / "reviews").glob("*.md")}
by_key = {(m["model"], m["tid"], m["kind"]): m["id"] for m in man}
missing = []
for mod in MODELS12:
    for tid in sorted(tids):
        for kind in ("review", "control"):
            rid = by_key.get((mod, tid, kind))
            if rid is None or rid not in reviews:
                missing.append((mod, tid, kind))
report["missing_reviews"] = [list(k) for k in missing]
if set(missing) != KNOWN_MISSING:
    errors.append(f"unexpected missing reviews: {sorted(set(missing) ^ KNOWN_MISSING)}")

scores, bad = {}, []
for rid in sorted(reviews):
    js = {}
    for j in ("J1", "J2"):
        p = FT / "judgements" / f"{rid}__{j}.json"
        if not p.exists():
            bad.append(f"{rid}: {j} missing"); continue
        s = {x["id"]: float(x["score"]) for x in json.loads(p.read_text())["scores"]}
        if sorted(s) != sorted(ITEMS) or any(v not in (0.0, 0.5, 1.0) for v in s.values()):
            bad.append(f"{rid}: {j} scores malformed")
        js[j] = s
    if len(js) == 2:
        scores[rid] = {it: (js["J1"][it] + js["J2"][it]) / 2 for it in ITEMS}
if bad:
    errors += bad[:20]
n_rev = sum(1 for m in man if m["id"] in reviews and m["kind"] == "review")
n_ctl = sum(1 for m in man if m["id"] in reviews and m["kind"] == "control")
n_judge = len(list((FT / "judgements").glob("*.json")))
report.update({"error_seeded_reviews": n_rev, "unmodified_reviews": n_ctl, "judge_outputs": n_judge})
for name, got, want in (("error-seeded reviews", n_rev, 696), ("unmodified reviews", n_ctl, 695),
                        ("judge outputs", n_judge, 2782)):
    if got != want:
        errors.append(f"{name}: {got}, README states {want}")

ref = json.loads((FT / "results" / "results_new_trials.json").read_text())["models"]
checked = 0
for mod in MODELS12:
    for kind in ("review", "control"):
        ids = [m["id"] for m in man if m["model"] == mod and m["kind"] == kind and m["tid"] in included
               and m["id"] in scores]
        S = [scores[i] for i in ids]
        det = np.mean([[s[it] >= 0.5 for it in ITEMS] for s in S])
        got = {"n_reviews": len(ids), "major_mean": np.mean([sum(s[i] for i in MAJOR) for s in S]),
               "minor_mean": np.mean([sum(s[i] for i in MINOR) for s in S]),
               "total_mean": np.mean([sum(s[i] for i in ITEMS) for s in S])}
        if kind == "review":
            got["sens"] = 100 * det
        else:
            got["spec"] = 100 * (1 - det)
        for k, v in got.items():
            r = ref[mod][kind][k]
            if abs(float(v) - float(r)) > 1e-6:
                errors.append(f"{mod} {kind} {k}: recomputed {v:.4f}, results file {r:.4f}")
            checked += 1
report["result_values_checked"] = checked
report["errors"] = errors
report["status"] = "PASS" if not errors else "FAIL"
out = FT / "validation"
out.mkdir(exist_ok=True)
(out / "further_trials_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "missing_reviews"}, indent=2))
sys.exit(1 if errors else 0)
