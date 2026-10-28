"""Reviews and judging for the 55 further trials published in 2026, through the providers' programming interfaces.

This is the runner that produced the reviews and judge outputs in Supplementary Data 1 (further_trials/). Re-running
it needs the control and error-seeded manuscripts in $FT_WORK/trials/<id>/, which are not distributed because of
the licences of the published articles. Every command is short-lived and resume-safe: batches run on the providers'
servers, and state lives in runs/state.json.

    python3 code/further_trials/ext_runner.py plan                 # create the blinded review manifest
    python3 code/further_trials/ext_runner.py submit-reviews [--models ...] [--tids ...] [--limit-batches N]
    python3 code/further_trials/ext_runner.py poll                 # check batches, collect finished results
    python3 code/further_trials/ext_runner.py submit-judges [--judges J1 J2]
    python3 code/further_trials/ext_runner.py status               # counts and estimated spend
    python3 code/further_trials/ext_runner.py sync-reviews --models dsv4pro qwen38_27b [--max-seconds 0]
                                                                   # synchronous OpenRouter calls for models without batch

Design follows the 2026 manuscripts of the main study: same reviewer instructions and notes, same models and reasoning
settings, a new independent request per review, the manuscript inside the prompt and no tools; the same two judges
(Fable 5 = J1, Opus 5 = J2), blinded to the reviewing model, scoring batches of up to five reviews of the same manuscript
against the trial's answer key with the 1 / 0.5 / 0 scheme.

API keys are read from the environment or a local .env (OPENROUTER_API_KEY, ANTHROPIC_API_KEY), or anthropic_key.txt.
They are never printed or written anywhere else.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_PROMPTS = os.path.join(REPO, "prompts", "further_trials")
EXT = os.environ.get("FT_WORK", os.path.join(REPO, "work_further_trials"))   # working folder with trials/<id>/
ROOT = os.path.dirname(EXT)
RUNS = os.path.join(EXT, "runs")
DATA = os.path.join(EXT, "data")
STATE = os.path.join(RUNS, "state.json")
MANIFEST = os.path.join(DATA, "manifest.json")
PROMPT_REVIEW = os.path.join(REPO_PROMPTS, "reviewer_prompt.txt")
PROMPT_JUDGE = os.path.join(REPO_PROMPTS, "judge_prompt.txt")
NOTES = ("this is a text-only version. The reference list and citation markers have been omitted for length. Tables "
         "are included in the text; figures and supplementary material are not included (figure legends are).")

# manifest key -> settings. Claude models: Anthropic Message Batches API with the highest reasoning setting
# (effort "max" as in the Claude Code runs; Haiku 4.5 has no effort parameter and uses extended thinking).
# Other models: OpenRouter, same model ids and effort as the main study. Models with a platform-billed batch offer go
# through OpenRouter's batch API; the others (no batch offer, a batch host capped at 16k output tokens, or batch only
# with an own OpenAI key) run as ordinary synchronous requests pinned to the main study's provider.
MODELS = {
    "haiku45":    {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001", "label": "Haiku 4.5",
                   "params": {"max_tokens": 64000, "thinking": {"type": "enabled", "budget_tokens": 32000}}},
    "sonnet5":    {"provider": "anthropic", "model_id": "claude-sonnet-5", "label": "Sonnet 5",
                   "params": {"max_tokens": 128000, "output_config": {"effort": "max"}}},
    "opus5":      {"provider": "anthropic", "model_id": "claude-opus-5", "label": "Opus 5",
                   "params": {"max_tokens": 128000, "output_config": {"effort": "max"}}},
    "fable5":     {"provider": "anthropic", "model_id": "claude-fable-5", "label": "Fable 5",
                   "params": {"max_tokens": 128000, "output_config": {"effort": "max"}}},
    "opus55":     {"provider": "anthropic", "model_id": "claude-opus-5-5", "label": "Opus 5.5",
                   "params": {"max_tokens": 128000, "output_config": {"effort": "max"}}},
    "gpt54mini": {"provider": "openrouter_sync", "model_id": "openai/gpt-5.4-mini", "label": "GPT-5.4 mini", "effort": "xhigh",
                   "routing": {"order": ["openai"], "allow_fallbacks": False}},
    "gpt54": {"provider": "openrouter_sync", "model_id": "openai/gpt-5.4", "label": "GPT-5.4", "effort": "xhigh",
                   "routing": {"order": ["openai"], "allow_fallbacks": False}},
    "gpt55": {"provider": "openrouter_sync", "model_id": "openai/gpt-5.5", "label": "GPT-5.5", "effort": "xhigh",
                   "routing": {"order": ["openai"], "allow_fallbacks": False}},
    "gpt61sol":   {"provider": "openrouter_sync", "model_id": "openai/gpt-6.1-sol", "label": "GPT-6.1 Sol",
                   "effort": "xhigh", "routing": {"order": ["openai"], "allow_fallbacks": False}},
    "gpt6luna":   {"provider": "openrouter_sync", "model_id": "openai/gpt-6-luna", "label": "GPT-6 Luna",
                   "effort": "xhigh", "routing": {"order": ["openai"], "allow_fallbacks": False}},
    "gemini38fl": {"provider": "openrouter_sync", "model_id": "google/gemini-3.8-flash", "label": "Gemini 3.8 Flash",
                   "effort": "high", "routing": {"allow_fallbacks": True}},
    "kimik3":     {"provider": "openrouter_sync", "model_id": "moonshotai/kimi-k3", "label": "Kimi K3",
                   "effort": "high", "routing": {"order": ["moonshotai"], "allow_fallbacks": False}},
    "gptoss120b": {"provider": "openrouter_sync", "model_id": "openai/gpt-oss-120b", "label": "gpt-oss-120b",
                   "effort": "high", "routing": {"order": ["deepinfra"], "quantizations": ["bf16"], "allow_fallbacks": False}},
    "dsv4pro":    {"provider": "openrouter_sync", "model_id": "deepseek/deepseek-v4-pro-0813", "label": "DeepSeek V4 Pro",
                   "effort": "high", "routing": {"order": ["baidu"], "quantizations": ["fp8"], "allow_fallbacks": False}},
    "qwen38_27b": {"provider": "openrouter_sync", "model_id": "qwen/qwen3.8-27b", "label": "Qwen3.8-27B",
                   "effort": "high", "routing": {"order": ["alibaba"], "allow_fallbacks": False}},
}
# Batch-API routing accepts only "only" and "allow_fallbacks"; pin the developer's own endpoint where the main study did.
BATCH_ROUTING = {"gpt61sol": {"only": ["openai"], "allow_fallbacks": False},
                 "gpt6luna": {"only": ["openai"], "allow_fallbacks": False}}
JUDGES = {"J1": "claude-fable-5", "J2": "claude-opus-5"}
JUDGE_MAX_TOKENS = 64000
ERROR_IDS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "m1", "m2", "m3", "m4", "m5"]


# ----------------------------------------------------------------------------------------------- helpers
def load_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_key(name):
    """Read an API key from the environment, the project .env or anthropic_key.txt; never print it."""
    if os.environ.get(name):
        return os.environ[name].strip()
    kd = os.environ.get("KEYS_DIR", ROOT)
    env = os.path.join(kd, ".env")
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    if name == "ANTHROPIC_API_KEY":
        p = os.path.join(kd, "anthropic_key.txt")
        if os.path.exists(p):
            return open(p).read().strip()
    sys.exit(f"{name} not found")


def http(method, url, headers, body=None, timeout=120, tries=4, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
                return payload if raw else json.loads(payload)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:800]
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < tries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code} {url}: {msg}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < tries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def anthropic_headers():
    return {"x-api-key": read_key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
            "content-type": "application/json"}


def openrouter_headers():
    return {"Authorization": "Bearer " + read_key("OPENROUTER_API_KEY"), "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rustlab1/llm-peer-review-benchmark", "X-Title": "inserted-error replication"}


def trials():
    return [t for t in json.load(open(os.path.join(EXT, "trials.json")))["selected"] if not t.get("excluded")]


def built_trials():
    """Trials whose clean, error-seeded and key files exist and passed the build."""
    out = []
    for t in trials():
        d = os.path.join(EXT, "trials", t["tid"])
        if all(os.path.exists(os.path.join(d, f)) for f in ("clean.md", "errors.md", "key.json", "READY")):
            out.append(t["tid"])
    return out


def review_prompt(job):
    ms = os.path.join(EXT, "trials", job["tid"], "errors.md" if job["kind"] == "review" else "clean.md")
    tpl = open(PROMPT_REVIEW).read()
    return tpl.replace("{NOTES}", NOTES).replace("{MANUSCRIPT}", open(ms).read().rstrip("\n"))


# ----------------------------------------------------------------------------------------------- plan
def cmd_plan(args):
    jobs = load_json(MANIFEST, [])
    have = {(j["tid"], j["model"], j["kind"], j["rep"]) for j in jobs}
    new = []
    for tid in built_trials():
        for m in MODELS:
            for rep in range(1, args.reps + 1):
                if (tid, m, "review", rep) not in have:
                    new.append({"tid": tid, "model": m, "kind": "review", "rep": rep})
            if (tid, m, "control", 1) not in have:
                new.append({"tid": tid, "model": m, "kind": "control", "rep": 1})
    random.Random(2026 + len(jobs)).shuffle(new)
    nr = sum(j["kind"] == "review" for j in jobs)
    nc = sum(j["kind"] == "control" for j in jobs)
    for j in new:
        if j["kind"] == "review":
            nr += 1
            j["id"] = f"ereview_{nr:04d}"
        else:
            nc += 1
            j["id"] = f"econtrol_{nc:04d}"
        j["model_id"] = MODELS[j["model"]]["model_id"]
        j["provider"] = MODELS[j["model"]]["provider"]
    jobs += new
    save_json(MANIFEST, jobs)
    print(f"manifest: {len(jobs)} jobs ({len(new)} new) for {len(built_trials())} built trials")


# ----------------------------------------------------------------------------------------------- state
def state():
    return load_json(STATE, {"batches": []})


def review_path(jid):
    return os.path.join(DATA, "reviews", jid + ".md")


def pending_review_jobs(models=None, tids=None, max_attempts=3):
    st = state()
    in_flight = set()
    tries = {}
    for b in st["batches"]:
        if b["kind"] != "review":
            continue
        for i in b["ids"]:
            tries[i] = tries.get(i, 0) + 1
        if b["status"] not in ("collected", "failed", "expired", "cancelled", "canceled", "abandoned"):
            in_flight.update(b["ids"])
    out = []
    for j in load_json(MANIFEST, []):
        if models and j["model"] not in models:
            continue
        if tids and j["tid"] not in tids:
            continue
        if os.path.exists(review_path(j["id"])) or j["id"] in in_flight or tries.get(j["id"], 0) >= max_attempts:
            continue
        out.append(j)
    return out


# ----------------------------------------------------------------------------------------------- submit reviews
def ms_sha(job):
    import hashlib
    ms = os.path.join(EXT, "trials", job["tid"], "errors.md" if job["kind"] == "review" else "clean.md")
    return hashlib.sha256(open(ms, "rb").read()).hexdigest()


def anthropic_review_request(job):
    m = MODELS[job["model"]]
    body = {"model": m["model_id"], "messages": [{"role": "user", "content": review_prompt(job)}]}
    body.update(json.loads(json.dumps(m["params"])))
    return {"custom_id": job["id"], "params": body}


def openrouter_review_body(job):
    m = MODELS[job["model"]]
    return {"model": m["model_id"], "messages": [{"role": "user", "content": review_prompt(job)}],
            "max_tokens": 64000, "reasoning": {"effort": m["effort"]}, "usage": {"include": True}}


def submit_anthropic(requests, kind, label, ids):
    res = http("POST", "https://api.anthropic.com/v1/messages/batches", anthropic_headers(), {"requests": requests},
               timeout=170)
    rec = {"provider": "anthropic", "kind": kind, "label": label, "batch_id": res["id"], "ids": ids,
           "status": res.get("processing_status", "in_progress"), "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
    if kind == "review":
        man = {j["id"]: j for j in load_json(MANIFEST, [])}
        rec["ms_sha"] = {i: ms_sha(man[i]) for i in ids}
    st = state()
    st["batches"].append(rec)
    save_json(STATE, st)
    return rec


def submit_openrouter(model_key, jobs):
    m = MODELS[model_key]
    body = {"endpoint": "/v1/chat/completions", "model": m["model_id"]}
    if model_key in BATCH_ROUTING:
        body["provider"] = BATCH_ROUTING[model_key]
    body["completion_window"] = "24h"
    body["requests"] = [{"custom_id": j["id"], "body": openrouter_review_body(j)} for j in jobs]
    res = http("POST", "https://openrouter.ai/api/v1/batches", openrouter_headers(), body, timeout=170)
    rec = {"provider": "openrouter", "kind": "review", "label": model_key, "batch_id": res["id"],
           "ids": [j["id"] for j in jobs], "status": res.get("status", "validating"),
           "submitted": time.strftime("%Y-%m-%d %H:%M:%S"), "ms_sha": {j["id"]: ms_sha(j) for j in jobs}}
    st = state()
    st["batches"].append(rec)
    save_json(STATE, st)
    return rec


def cmd_submit_reviews(args):
    jobs = pending_review_jobs(args.models, args.tids)
    if args.dry_run:
        from collections import Counter
        print(Counter(j["model"] for j in jobs))
        return
    by_model = {}
    for j in jobs:
        by_model.setdefault(j["model"], []).append(j)
    n = 0
    capped = {"anthropic": over_cap("anthropic"), "openrouter": over_cap("openrouter")}
    for mk, js in sorted(by_model.items()):
        prov = MODELS[mk]["provider"]
        if prov == "openrouter_sync":
            continue
        if capped["anthropic" if prov == "anthropic" else "openrouter"]:
            print(f"SPEND CAP reached, not submitting {mk}")
            continue
        for i in range(0, len(js), args.chunk):
            chunk = js[i:i + args.chunk]
            try:
                if prov == "anthropic":
                    rec = submit_anthropic([anthropic_review_request(j) for j in chunk], "review", mk,
                                           [j["id"] for j in chunk])
                else:
                    rec = submit_openrouter(mk, chunk)
            except Exception as e:
                print(f"SUBMIT FAILED {mk}: {str(e)[:300]}")
                break
            n += 1
            print(f"submitted {mk}: {len(chunk)} reviews -> batch {rec['batch_id']}")
            if args.limit_batches and n >= args.limit_batches:
                return


# ----------------------------------------------------------------------------------------------- poll / collect
def extract_anthropic_text(msg):
    return "".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text").strip()


def extract_openrouter_text(body):
    try:
        return (body["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def finish_review(jid, text, raw, ok_reason):
    os.makedirs(os.path.join(DATA, "reviews"), exist_ok=True)
    os.makedirs(os.path.join(RUNS, "raw"), exist_ok=True)
    save_json(os.path.join(RUNS, "raw", jid + ".json"), raw)
    if text and ok_reason:
        open(review_path(jid), "w", encoding="utf-8").write(text + "\n")
        return True
    return False


def poll_anthropic(b):
    h = anthropic_headers()
    info = http("GET", f"https://api.anthropic.com/v1/messages/batches/{b['batch_id']}", h)
    b["status"] = info.get("processing_status")
    b["counts"] = info.get("request_counts")
    if b["status"] != "ended":
        return
    lines = http("GET", info["results_url"], h, timeout=170, raw=True).decode().splitlines()
    done, bad = 0, []
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        cid = r["custom_id"]
        res = r.get("result", {})
        if b["kind"] == "review":
            if res.get("type") == "succeeded":
                msg = res["message"]
                text = extract_anthropic_text(msg)
                ok = msg.get("stop_reason") in ("end_turn", "stop_sequence") and len(text.split()) > 150
                if finish_review(cid, text, r, ok):
                    done += 1
                else:
                    bad.append(cid)
            else:
                save_json(os.path.join(RUNS, "raw", cid + ".error.json"), r)
                bad.append(cid)
        else:
            if res.get("type") == "succeeded":
                if store_judgement_batch(b, cid, res["message"]):
                    done += 1
                else:
                    bad.append(cid)
            else:
                save_json(os.path.join(RUNS, "judge_raw", cid + ".error.json"), r)
                bad.append(cid)
    b["status"] = "collected"
    b["n_ok"], b["failed_ids"] = done, bad


def poll_openrouter(b):
    info = http("GET", f"https://openrouter.ai/api/v1/batches/{b['batch_id']}", openrouter_headers(), timeout=170)
    b["status"] = info.get("status")
    b["counts"] = info.get("request_counts")
    if b["status"] in ("failed", "expired", "cancelled"):
        b["error"] = str(info.get("errors") or info.get("error"))[:500]
        return
    if b["status"] != "completed" or info.get("results") is None:
        return
    done, bad = 0, []
    for r in info["results"]:
        cid = r["custom_id"]
        resp = (r.get("response") or {})
        body = resp.get("body") or {}
        text = extract_openrouter_text(body)
        fin = ""
        try:
            fin = body["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            pass
        ok = resp.get("status_code") == 200 and fin == "stop" and len(text.split()) > 150
        if finish_review(cid, text, r, ok):
            done += 1
        else:
            bad.append(cid)
    b["status"] = "collected"
    b["n_ok"], b["failed_ids"] = done, bad


def cmd_poll(args):
    st = state()
    for b in st["batches"]:
        if b["status"] in ("collected", "failed", "expired", "cancelled", "canceled", "abandoned"):
            continue
        try:
            if b["provider"] == "anthropic":
                poll_anthropic(b)
            else:
                poll_openrouter(b)
        except Exception as e:
            print(f"poll error {b['label']} {b['batch_id']}: {e}")
        save_json(STATE, st)
        print(f"{b['provider']:10} {b['kind']:6} {b['label']:12} {b['status']:12} {b.get('counts')}"
              + (f" ok={b.get('n_ok')} failed={len(b.get('failed_ids', []))}" if b["status"] == "collected" else ""))


# ----------------------------------------------------------------------------------------------- judges
def judge_out(jid, judge):
    return os.path.join(DATA, "judgements", f"{jid}__{judge}.json")


def valid_judgement(path):
    try:
        d = json.load(open(path))
        return [s["id"] for s in d["scores"]] == ERROR_IDS and all(s["score"] in (0, 0.5, 1) for s in d["scores"])
    except Exception:
        return False


def judge_prompt(key_json, reviews):
    """reviews: list of (label, text). The batch prompt of the main study with the files placed inline."""
    tpl = open(PROMPT_JUDGE).read()
    blocks = []
    for lab, text in reviews:
        blocks.append(f"<<<REVIEW {lab}\n{text.strip()}\nREVIEW {lab}>>>")
    return (tpl.replace("{N_REVIEWS}", str(len(reviews))).replace("{KEY_JSON}", key_json)
            .replace("{REVIEWS}", "\n\n".join(blocks)).replace("{LABELS}", ", ".join(l for l, _ in reviews)))


def store_judgement_batch(b, cid, msg):
    """Parse one judge reply (a JSON object keyed by review label) into per-review judgement files."""
    os.makedirs(os.path.join(RUNS, "judge_raw"), exist_ok=True)
    save_json(os.path.join(RUNS, "judge_raw", cid + ".json"), msg)
    text = extract_anthropic_text(msg)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    meta = b["requests"][cid]
    ok_all = True
    for lab, jid in meta["labels"].items():
        rec = obj.get(lab)
        if not isinstance(rec, dict):
            ok_all = False
            continue
        rec = dict(rec)
        rec["judge"], rec["judge_model"], rec["review_id"] = meta["judge"], msg.get("model"), jid
        p = judge_out(jid, meta["judge"])
        save_json(p, rec)
        if not valid_judgement(p):
            os.replace(p, p + ".invalid")
            ok_all = False
    return ok_all


def cmd_submit_judges(args):
    jobs = {j["id"]: j for j in load_json(MANIFEST, [])}
    st = state()
    in_flight = set()
    for b in st["batches"]:
        if b["kind"] == "judge" and b["status"] not in ("collected", "failed", "expired", "canceled", "cancelled", "abandoned"):
            for r in b["requests"].values():
                in_flight.update((jid, r["judge"]) for jid in r["labels"].values())
    review_in_flight = set()
    for b in st["batches"]:
        if b["kind"] == "review" and b["status"] not in ("collected", "failed", "expired", "canceled", "cancelled", "abandoned"):
            review_in_flight.update(b["ids"])
    busy = {(jobs[i]["tid"], jobs[i]["kind"]) for i in review_in_flight if i in jobs}
    active = load_json(os.path.join(RUNS, "active_models.json"), None)
    if active:
        sync_tries = load_json(os.path.join(RUNS, "sync_attempts.json"), {})
        for j in pending_review_jobs(active):
            if MODELS[j["model"]]["provider"] != "openrouter_sync" or sync_tries.get(j["id"], 0) < MAX_SYNC_ATTEMPTS:
                busy.add((j["tid"], j["kind"]))
    import glob as _g
    for f in _g.glob(os.path.join(RUNS, "sync_busy*.json")):
        pid = os.path.basename(f)[len("sync_busy_"):-len(".json")]
        if pid.isdigit() and not os.path.exists(f"/proc/{pid}"):
            continue  # worker process is gone; its in-flight list is stale
        busy |= set(tuple(x) for x in load_json(f, []))
    rng = random.Random(77 + len(st["batches"]))
    for judge in args.judges:
        groups = {}
        for jid, j in jobs.items():
            if not os.path.exists(review_path(jid)) or (j["tid"], j["kind"]) in busy:
                continue
            if valid_judgement(judge_out(jid, judge)) or (jid, judge) in in_flight:
                continue
            groups.setdefault((j["tid"], j["kind"]), []).append(jid)
        requests, reqmeta = [], {}
        for (tid, kind), ids in sorted(groups.items()):
            rng.shuffle(ids)
            key_json = open(os.path.join(EXT, "trials", tid, "key.json")).read()
            for i in range(0, len(ids), 5):
                chunk = ids[i:i + 5]
                labels = {f"R{k + 1}": jid for k, jid in enumerate(chunk)}
                reviews = [(lab, open(review_path(jid)).read()) for lab, jid in labels.items()]
                cid = f"{judge}_{tid}_{kind}_{len(st['batches']):03d}_{i // 5:02d}"
                requests.append({"custom_id": cid, "params": {
                    "model": JUDGES[judge], "max_tokens": JUDGE_MAX_TOKENS,
                    "messages": [{"role": "user", "content": judge_prompt(key_json, reviews)}]}})
                reqmeta[cid] = {"judge": judge, "tid": tid, "kind": kind, "labels": labels}
        if not requests:
            print(f"{judge}: nothing to judge")
            continue
        if over_cap("anthropic"):
            print("SPEND CAP reached, not submitting judges")
            return
        for i in range(0, len(requests), args.chunk):
            part = requests[i:i + args.chunk]
            res = http("POST", "https://api.anthropic.com/v1/messages/batches", anthropic_headers(),
                       {"requests": part}, timeout=170)
            st = state()
            st["batches"].append({"provider": "anthropic", "kind": "judge", "label": judge, "batch_id": res["id"],
                                  "requests": {r["custom_id"]: reqmeta[r["custom_id"]] for r in part},
                                  "ids": [r["custom_id"] for r in part], "status": res.get("processing_status"),
                                  "submitted": time.strftime("%Y-%m-%d %H:%M:%S")})
            save_json(STATE, st)
            print(f"submitted judge {judge}: {len(part)} requests -> batch {res['id']}")


# ----------------------------------------------------------------------------------------------- synchronous models
def cmd_sync_reviews(args):
    """Synchronous OpenRouter reviews. With --follow, keep picking up jobs for newly ready trials until all trials in
    trials.json are ready and no job is left (used for the models without a batch offer; run from a terminal)."""
    args._t_stop = time.time() + args.stop_after if getattr(args, "stop_after", 0) else None
    while True:
        _sync_once(args)
        if not args.follow:
            return
        if args._t_stop and time.time() > args._t_stop:
            print(time.strftime("%H:%M:%S"), "stop-after reached, exiting", flush=True)
            return
        n_ready = len(built_trials())
        left = [j for j in pending_review_jobs(args.models, args.tids)
                if MODELS[j["model"]]["provider"] == "openrouter_sync"]
        if n_ready >= len(trials()) and not left:
            print("all sync reviews done", flush=True)
            return
        if not left:
            print(time.strftime("%H:%M:%S"), f"waiting for more trials ({n_ready}/{len(trials())} ready)", flush=True)
            time.sleep(120)


SYNC_ATTEMPTS = os.path.join(RUNS, "sync_attempts.json")
MAX_SYNC_ATTEMPTS = 5  # failed requests (no complete review, e.g. reasoning used all output tokens) are re-sent as new requests


def _sync_once(args):
    import threading
    attempts = load_json(SYNC_ATTEMPTS, {})
    lock = threading.Lock()
    jobs = [j for j in pending_review_jobs(args.models, args.tids)
            if MODELS[j["model"]]["provider"] == "openrouter_sync" and attempts.get(j["id"], 0) < MAX_SYNC_ATTEMPTS]
    if not jobs:
        return
    if over_cap("openrouter"):
        print("SPEND CAP reached, not running sync reviews", flush=True)
        time.sleep(600)
        return
    t_end = time.time() + args.max_seconds if args.max_seconds else None
    from concurrent.futures import ThreadPoolExecutor

    inflight = set()

    lockdir = os.path.join(RUNS, "locks")
    os.makedirs(lockdir, exist_ok=True)

    def one(j):
        # Graceful stop: do not start new requests after --stop-after; in-flight ones finish normally.
        t_stop = getattr(args, "_t_stop", None)
        if t_stop and time.time() > t_stop:
            return j["id"], "skipped (stop-after)"
        if os.path.exists(review_path(j["id"])):
            return j["id"], "skipped (done elsewhere)"
        # Cross-process lock so overlapping workers (rolling restarts) never run the same job twice.
        lp = os.path.join(lockdir, j["id"] + ".lock")
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
        except FileExistsError:
            if time.time() - os.path.getmtime(lp) < 2400:
                return j["id"], "skipped (locked)"
            os.utime(lp, None)
        try:
            return _one_counted(j)
        finally:
            try:
                os.remove(lp)
            except OSError:
                pass

    def _one_counted(j):
        with lock:
            attempts.update(load_json(SYNC_ATTEMPTS, {}))
            attempts[j["id"]] = attempts.get(j["id"], 0) + 1
            save_json(SYNC_ATTEMPTS, attempts)
            inflight.add((j["tid"], j["kind"], j["id"]))
            save_json(busy_file, sorted({(a, b) for a, b, _ in inflight}))
        try:
            return _one(j)
        finally:
            with lock:
                inflight.discard((j["tid"], j["kind"], j["id"]))
                save_json(busy_file, sorted({(a, b) for a, b, _ in inflight}))

    def _one(j):
        m = MODELS[j["model"]]
        body = openrouter_review_body(j)
        body["provider"] = m["routing"]
        t0 = time.time()
        timeout = max(30, int(t_end - time.time())) if t_end else 1800
        try:
            r = http("POST", "https://openrouter.ai/api/v1/chat/completions", openrouter_headers(), body,
                     timeout=timeout, tries=1 if t_end else 3)
        except Exception as e:
            return j["id"], f"FAILED {str(e)[:120]}"
        text = extract_openrouter_text(r)
        fin = (r.get("choices") or [{}])[0].get("finish_reason") if isinstance(r, dict) else None
        r["_elapsed_s"] = round(time.time() - t0, 1)
        r["_ms_sha"] = ms_sha(j)
        ok = finish_review(j["id"], text, r, fin == "stop" and len(text.split()) > 150)
        return j["id"], "ok" if ok else f"bad finish={fin}"

    busy_file = os.path.join(RUNS, f"sync_busy_{os.getpid()}.json")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for jid, res in ex.map(one, jobs):
                print(time.strftime("%H:%M:%S"), jid, res, flush=True)
    finally:
        save_json(busy_file, [])


def cmd_cycle(args):
    """One pass: add jobs for newly ready trials, submit reviews for the chosen models, collect, submit judges."""
    cmd_plan(argparse.Namespace(reps=1))
    cmd_submit_reviews(argparse.Namespace(models=args.models, tids=None, chunk=200, limit_batches=0, dry_run=False))
    cmd_poll(args)
    if not args.no_judges:
        cmd_submit_judges(argparse.Namespace(judges=["J1", "J2"], chunk=200))
    cmd_status(args)


# ----------------------------------------------------------------------------------------------- status
def cmd_status(args):
    jobs = load_json(MANIFEST, [])
    from collections import Counter
    have = Counter((j["model"], j["kind"]) for j in jobs if os.path.exists(review_path(j["id"])))
    tot = Counter((j["model"], j["kind"]) for j in jobs)
    print("model        reviews     controls")
    for m in MODELS:
        print(f"{m:12} {have[(m, 'review')]:3}/{tot[(m, 'review')]:<5} {have[(m, 'control')]:3}/{tot[(m, 'control')]:<5}")
    jd = Counter()
    for j in jobs:
        for J in JUDGES:
            if valid_judgement(judge_out(j["id"], J)):
                jd[J] += 1
    print("judged:", dict(jd))
    st = state()
    active = [b for b in st["batches"] if b["status"] not in ("collected", "failed", "expired", "cancelled", "canceled", "abandoned")]
    print(f"batches: {len(st['batches'])} total, {len(active)} active")
    print("estimated spend so far (USD):", spend())


# Message Batches prices per million tokens (input, output), 50% of list price, October 2026
BATCH_PRICE = {"claude-haiku-4-5-20251001": (0.5, 2.5), "claude-sonnet-5": (1.0, 5.0), "claude-opus-5": (2.5, 12.5),
               "claude-fable-5": (5.0, 25.0), "claude-opus-5-5": (2.0, 10.0)}


def openrouter_usage():
    try:
        return float(http("GET", "https://openrouter.ai/api/v1/key", openrouter_headers(), timeout=60)["data"]["usage"])
    except Exception:
        return None


def spend():
    """Anthropic: usage of collected requests at batch prices. OpenRouter: the key's cumulative usage since the
    baseline recorded before the first submission of this extension."""
    tot = {"anthropic_reviews": 0.0, "anthropic_judges": 0.0, "openrouter": 0.0}
    base = load_json(os.path.join(RUNS, "baseline.json"), {})
    u = openrouter_usage()
    if u is not None:
        if "openrouter_usage_start" not in base:
            base["openrouter_usage_start"] = u
            base["recorded"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_json(os.path.join(RUNS, "baseline.json"), base)
        tot["openrouter_since_start"] = u - base["openrouter_usage_start"]
    for sub, k in (("raw", "reviews"), ("judge_raw", "judges")):
        d = os.path.join(RUNS, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".json") or f.endswith(".error.json"):
                continue
            try:
                r = json.load(open(os.path.join(d, f)))
            except Exception:
                continue
            msg = r.get("result", {}).get("message") if "result" in r else (r if r.get("type") == "message" else None)
            if msg and msg.get("usage"):
                pi, po = BATCH_PRICE.get(msg.get("model"), (0, 0))
                u = msg["usage"]
                tot["anthropic_" + k] += (u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po) / 1e6
                continue
            body = ((r.get("response") or {}).get("body")) if "response" in r else r
            if isinstance(body, dict) and isinstance(body.get("usage"), dict):
                tot["openrouter"] += float(body["usage"].get("cost") or 0)
    tot["anthropic_total"] = tot["anthropic_reviews"] + tot["anthropic_judges"]
    return {k: round(v, 2) for k, v in tot.items()}


CAPS = {"anthropic": float(os.environ.get("ANTHROPIC_CAP", "60")),
        "openrouter": float(os.environ.get("OPENROUTER_CAP", "60"))}
CAPS.update(load_json(os.path.join(RUNS, "caps.json"), {}))   # runs/caps.json overrides the environment


def over_cap(provider):
    sp = spend()
    if provider == "anthropic":
        return sp["anthropic_total"] >= CAPS["anthropic"]
    return sp.get("openrouter_since_start", 0) >= CAPS["openrouter"]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--reps", type=int, default=1)
    p = sub.add_parser("submit-reviews"); p.add_argument("--models", nargs="*"); p.add_argument("--tids", nargs="*")
    p.add_argument("--chunk", type=int, default=200); p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("poll")
    p = sub.add_parser("submit-judges"); p.add_argument("--judges", nargs="+", default=["J1", "J2"])
    p.add_argument("--chunk", type=int, default=200)
    sub.add_parser("status")
    p = sub.add_parser("cycle"); p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--no-judges", action="store_true")
    p = sub.add_parser("sync-reviews"); p.add_argument("--models", nargs="+", default=["kimik3", "dsv4pro", "qwen38_27b"])
    p.add_argument("--tids", nargs="*"); p.add_argument("--max-seconds", type=int, default=0)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--follow", action="store_true"); p.add_argument("--stop-after", type=int, default=0)
    a = ap.parse_args()
    {"plan": cmd_plan, "submit-reviews": cmd_submit_reviews, "poll": cmd_poll, "submit-judges": cmd_submit_judges,
     "status": cmd_status, "sync-reviews": cmd_sync_reviews, "cycle": cmd_cycle}[a.cmd](a)


if __name__ == "__main__":
    main()
