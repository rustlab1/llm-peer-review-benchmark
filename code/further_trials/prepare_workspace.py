"""Build the working folder that the further-trials scripts read, from an extracted Supplementary Data 1.

The scripts were written against the study's working layout (data/manifest.json, data/judgements/, trials.json,
runs/active_models.json). This script recreates that layout from further_trials/ in Supplementary Data 1, so the
published reviews and judge outputs can be analysed without any model calls.

Usage (from the repository root):
    python code/further_trials/prepare_workspace.py /path/to/Supplementary_Data_1 [work_further_trials]
    export SD1=/path/to/Supplementary_Data_1 FT_WORK=work_further_trials
    bash code/further_trials/run_final_analysis.sh
"""
import csv
import json
import os
import shutil
import sys

ACTIVE = ["haiku45", "sonnet5", "gpt54mini", "gpt54", "gpt55", "gpt61sol", "gpt6luna", "gemini38fl", "kimik3",
          "gptoss120b", "dsv4pro", "qwen38_27b"]
RERUN = {"N26": "Tenecteplase vs low-dose alteplase before thrombectomy (main study paper N)",
         "I26": "Guselkumab in psoriatic arthritis after one TNF inhibitor (main study paper I)",
         "M26": "Polypill vs usual care in heart failure (main study paper M)"}

sd1 = os.path.abspath(sys.argv[1])
work = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "work_further_trials")
ft = os.path.join(sd1, "further_trials")
if not os.path.exists(os.path.join(ft, "manifest.json")):
    raise SystemExit(f"not an extracted Supplementary Data 1 (no further_trials/manifest.json): {sd1}")
for sub in ("data", "runs", "figures"):
    os.makedirs(os.path.join(work, sub), exist_ok=True)
shutil.copy(os.path.join(ft, "manifest.json"), os.path.join(work, "data", "manifest.json"))
shutil.copy(os.path.join(ft, "results", "trial_labels.json"), os.path.join(work, "data", "trial_labels.json"))
link = os.path.join(work, "data", "judgements")
if not os.path.exists(link):
    try:
        os.symlink(os.path.join(ft, "judgements"), link)
    except OSError:                                  # no symlinks (e.g. Windows without developer mode): copy
        shutil.copytree(os.path.join(ft, "judgements"), link)
selected = []
with open(os.path.join(ft, "trials", "trials.csv"), newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        t = {"tid": r["trial_id"], "short": r["label"], "journal": r["journal"], "pmc": r["pmcid"], "pmid": r["pmid"],
             "doi": r["doi"], "note": r["preparation_note"]}
        if r["status"] == "excluded":
            t["excluded"] = r["reason_excluded"] or True
        if r["screen"].startswith("phase 3"):
            t["set"] = "phase3_since_mid_july"
        selected.append(t)
selected += [{"tid": tid, "short": short, "source": "main_study", "journal": "", "pmc": ""} for tid, short in RERUN.items()]
json.dump({"selected": selected}, open(os.path.join(work, "trials.json"), "w"), indent=1)
json.dump(ACTIVE, open(os.path.join(work, "runs", "active_models.json"), "w"))
print(f"workspace ready: {work} ({len(selected)} trials)")
print(f"export SD1={sd1} FT_WORK={work}")
