"""Estimated cost of one review for all 15 models, at list prices (Supplementary Table 4).

One uniform estimate for every model, so that models run through different software can be compared:
  cost = 10,000 input tokens x input price + output tokens x output price
- 10,000 input tokens is the size of a 2026 manuscript with the reviewer instructions (the measured prompt was
  8,400 to 11,800 tokens depending on the paper).
- Output tokens include the model's reasoning. They are the recorded means over the 12 error-seeded 2026 reviews
  (data/cost_time_new_models.json, models run through OpenRouter and Opus 5.5) or over the three timed sessions
  (data/timing_run/timing_summary.json, Haiku 4.5, Sonnet 5, Opus 5 and Fable 5). The three GPT-5 models ran through
  Codex, which did not record tokens: their output is estimated from review length (1.3 tokens per word) times the
  median ratio of total output to visible review tokens among the 12 recorded models.
- Prices are the developers' list prices of September 2026 (USD per million tokens). For the open-weight models the
  developer's own price is used where the developer sells access (Moonshot AI, DeepSeek, Alibaba); OpenAI does not
  host gpt-oss-120b, so a typical host price is used.
Minutes per review are the recorded means (not recorded for the GPT-5 models). Words per review are the main-experiment
means for every model (the ratio above uses the word counts of the sessions whose tokens were recorded).
Writes data/cost_estimates_v8.json.   Run: python3 code/estimate_costs.py
"""
import json, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_TOKENS = 10_000
TOKENS_PER_WORD = 1.3

# USD per million tokens (input, output), September 2026
PRICE = {"haiku": (1.0, 5.0), "sonnet": (2.0, 10.0), "opus": (5.0, 25.0), "fable": (10.0, 50.0), "opus55": (4.0, 20.0),
         "gpt54mini": (0.75, 4.5), "gpt54": (2.5, 15.0), "gpt55": (5.0, 30.0), "gpt61sol": (2.0, 10.0), "gpt6luna": (0.1, 0.5),
         "gemini38fl": (0.75, 3.75), "kimik3": (3.0, 15.0), "dsv4pro": (0.66, 1.98), "qwen38_27b": (0.425, 2.55),
         "gptoss120b": (0.15, 0.60)}
PRICE_SOURCE = {"gptoss120b": "typical host price (OpenAI does not host this model)",
                "kimik3": "developer's own price", "dsv4pro": "developer's own price", "qwen38_27b": "developer's own price"}
OPEN = {"kimik3", "dsv4pro", "qwen38_27b", "gptoss120b"}

R2 = json.load(open(os.path.join(ROOT, "data", "results.json")))["summary"]
CT = json.load(open(os.path.join(ROOT, "data", "cost_time_new_models.json")))
TR = json.load(open(os.path.join(ROOT, "data", "timing_run", "timing_summary.json")))

rec = {}                                  # model -> (output tokens, minutes, words, recorded?)
for m, v in list(TR.items()) + list(CT.items()):
    rec[m] = {"output_tokens": v["output_tokens_mean"], "reasoning_tokens": v["reasoning_tokens_mean"],
              "minutes": v["min_mean"], "minutes_range": [v["min_min"], v["min_max"]], "words": v["words_mean"], "recorded": True}
# ratio of total output to visible review tokens among the recorded models
ratios = [rec[m]["output_tokens"] / (rec[m]["words"] * TOKENS_PER_WORD) for m in rec]
RATIO = round(st.median(ratios), 1)
for m in ("gpt54mini", "gpt54", "gpt55"):
    words = R2[m]["words_mean"]
    rec[m] = {"output_tokens": words * TOKENS_PER_WORD * RATIO, "reasoning_tokens": None, "minutes": None,
              "minutes_range": None, "words": words, "recorded": False}

out = {"_method": {"input_tokens": INPUT_TOKENS, "tokens_per_word": TOKENS_PER_WORD, "output_to_visible_ratio_median": RATIO,
                   "ratios_recorded_models": {m: round(r, 2) for m, r in zip(rec, ratios)}}}
for m in PRICE:
    pi, po = PRICE[m]; r = rec[m]
    cost = INPUT_TOKENS * pi / 1e6 + r["output_tokens"] * po / 1e6
    found = R2[m]["major_mean"] + R2[m]["minor_mean"]
    out[m] = {"open_weight": m in OPEN, "price_in": pi, "price_out": po, "price_source": PRICE_SOURCE.get(m, "developer's list price"),
              "output_tokens": r["output_tokens"], "reasoning_tokens": r["reasoning_tokens"], "output_recorded": r["recorded"],
              "cost": cost, "errors_found": found, "cost_per_error": cost / found,
              "minutes": r["minutes"], "minutes_range": r["minutes_range"],
              "words": R2[m]["words_mean"]}          # words per review in the main experiment, for every model
json.dump(out, open(os.path.join(ROOT, "data", "cost_estimates_v8.json"), "w"), indent=1)
print(f"output/visible ratio median {RATIO} ({min(ratios):.1f} to {max(ratios):.1f})")
for m in sorted(PRICE, key=lambda m: out[m]["errors_found"]):
    o = out[m]
    print(f"{m:11s} out {o['output_tokens']/1000:6.1f}k {'rec' if o['output_recorded'] else 'est'}  cost ${o['cost']:.4f}  "
          f"found {o['errors_found']:.2f}  per error ${o['cost_per_error']:.4f}  min {o['minutes'] if o['minutes'] is None else round(o['minutes'],1)}")
