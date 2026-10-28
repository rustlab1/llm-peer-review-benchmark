"""Recorded time, tokens and billed cost for the API reviews and Opus 5.5 sessions.

OpenRouter models: cost is OpenRouter's billed cost per request (usage.cost in each log); time is the wall-clock
duration of the request as printed by run_api_reviews.py (run_logs/*.log). Requests that needed a retry are left out
of the timing (their printed time includes the failed attempts and back-off), but their cost is the billed cost of the
successful request. Requests that failed in one pass and succeeded at the first attempt of a later pass are timed from
that successful request. Opus 5.5: cost at list price and duration as reported by Claude Code for each session
(logs/claude_code_runs_opus55/exp*/manifest.json, field "audit").
Statistics are over the 12 reviews of the error-seeded 2026 trials, as in Supplementary Table 4.
Writes data/cost_time_new_models.json.   Run: python3 code/summarize_session_costs.py
"""
import glob, json, os, re, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OR = os.path.join(ROOT, "logs", "open_model_runs")
CL = os.path.join(ROOT, "logs", "claude_code_runs_opus55")

# wall-clock minutes per successful request, from the runner's own log lines
minutes = {}
# read the logs in the order the runs happened, so that a later successful run of the same request (the one whose
# output was kept) overrides an earlier one (e.g. the two unpinned pilot requests that were set aside)
LOG_ORDER = ["pilot.log", "or_exp1.log", "or_exp2.log", "or_ds.log", "or_retry.log"]
for lf in [os.path.join(OR, "run_logs", f) for f in LOG_ORDER]:
    for line in open(lf):
        m = re.match(r"\d\d:\d\d:\d\d (\S+) \S+ p\d ok \(finish=stop.*\) ([\d.]+) min", line)
        if m:
            minutes[m.group(1)] = float(m.group(2))

out = {}
man = json.load(open(os.path.join(OR, "exp2", "manifest.json")))
for model in sorted({j["model"] for j in man}):
    ids = [j["id"] for j in man if j["model"] == model and j["kind"] == "review"]
    cost, tin, tout, treas, mins, words = [], [], [], [], [], []
    for i in ids:
        r = json.load(open(os.path.join(OR, "exp2", "logs", i + ".json")))
        u = r["usage"]
        cost.append(u["cost"]); tin.append(u["prompt_tokens"]); tout.append(u["completion_tokens"])
        treas.append((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        words.append(len(open(os.path.join(OR, "exp2", "out", i + ".md")).read().split()))
        retried = bool(glob.glob(os.path.join(OR, "exp2", "logs", i + ".attempt*.json")))
        if i in minutes and not retried:
            mins.append(minutes[i])
    out[model] = {"n": len(ids), "cost_mean": st.mean(cost), "cost_min": min(cost), "cost_max": max(cost),
                  "min_mean": st.mean(mins), "min_min": min(mins), "min_max": max(mins), "n_timed": len(mins),
                  "input_tokens_mean": st.mean(tin), "output_tokens_mean": st.mean(tout),
                  "reasoning_tokens_mean": st.mean(treas), "words_mean": st.mean(words),
                  "cost_basis": "OpenRouter billed cost", "providers": sorted({json.load(open(os.path.join(OR, "exp2", "logs", i + ".json"))).get("provider") for i in ids})}

cm = json.load(open(os.path.join(CL, "exp2", "manifest.json")))
R = [j for j in cm if j["kind"] == "review"]
cost = [j["audit"]["cost_usd_list"] for j in R]; mins = [j["audit"]["duration_s"] / 60 for j in R]
u = [j["audit"]["usage"] for j in R]
tin = [x["input_tokens"] + x.get("cache_creation_input_tokens", 0) + x.get("cache_read_input_tokens", 0) for x in u]
tout = [x["output_tokens"] for x in u]
treas = [(x.get("output_tokens_details") or {}).get("thinking_tokens", 0) for x in u]
words = [len(open(os.path.join(CL, "exp2", "out", j["id"] + ".md")).read().split()) for j in R]
out["opus55"] = {"n": len(R), "cost_mean": st.mean(cost), "cost_min": min(cost), "cost_max": max(cost),
                 "min_mean": st.mean(mins), "min_min": min(mins), "min_max": max(mins), "n_timed": len(mins),
                 "input_tokens_mean": st.mean(tin), "output_tokens_mean": st.mean(tout),
                 "reasoning_tokens_mean": st.mean(treas), "words_mean": st.mean(words),
                 "cost_basis": "Claude Code report at list price", "providers": ["Anthropic (Claude Code)"]}
json.dump(out, open(os.path.join(ROOT, "data", "cost_time_new_models.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:12s} ${v['cost_mean']:.4f} ({v['cost_min']:.4f}-{v['cost_max']:.4f})  {v['min_mean']:.1f} min ({v['min_min']:.1f}-{v['min_max']:.1f}, n={v['n_timed']})  in {v['input_tokens_mean']/1000:.1f}k out {v['output_tokens_mean']/1000:.1f}k (reas {v['reasoning_tokens_mean']/1000:.1f}k) words {v['words_mean']:.0f} {v['providers']}")
