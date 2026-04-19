#!/usr/bin/env python3
"""Continuous loop: process pending prompts from raw_prompts.jsonl.
Auto-resumes on restart. Runs until all pending are done.
Uses socket timeout for API calls. No retry. No hang."""
import json, hashlib, os, time, sys, socket
from datetime import datetime, timezone

socket.setdefaulttimeout(15)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from survival import SurvivalEngine, SurvivalConfig

cfg = SurvivalConfig(
    zhipu_api_key="85c159bed2ad4cd6a36ec3110c842c65.Xx3EKUOkOCEf1jHo",
    provider="zhipu",
    request_delay=0.1,
    n_perturbations=2,
    n_contexts=3,
)

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(BASE, "logs", "shadow_eval_live.jsonl")
RAW = os.path.join(BASE, "data", "raw_prompts.jsonl")
METRICS = os.path.join(BASE, "logs", "daily_metrics.jsonl")

engine = SurvivalEngine(cfg)
CYCLE_TIME = 100  # seconds per cycle

while True:
    # Reload evaluated qids each cycle (resilient to crashes)
    evaluated = set()
    with open(LIVE) as f:
        for l in f:
            r = json.loads(l.strip())
            evaluated.add(r.get("query_id", ""))

    pending = []
    with open(RAW) as f:
        for l in f:
            r = json.loads(l.strip())
            p = r.get("prompt", "").strip()
            if p:
                qid = hashlib.sha256(p.encode()).hexdigest()[:12]
                if qid not in evaluated:
                    pending.append((qid, p, r))

    if not pending:
        print(f"[{datetime.now(timezone.utc).isoformat()}] ALL DONE", flush=True)
        break

    print(f"[{datetime.now(timezone.utc).isoformat()}] {len(pending)} pending", flush=True)
    done = 0
    skip = 0
    batch = []
    t0 = time.time()

    for idx, (qid, p, rec) in enumerate(pending):
        elapsed = time.time() - t0
        if elapsed > CYCLE_TIME:
            print(f"  cycle limit ({CYCLE_TIME}s), pausing", flush=True)
            break

        print(f"  [{idx+1}/{len(pending)}] ", end="", flush=True)
        try:
            result = engine.evaluate_shadow(p)
            result["source"] = rec.get("source", "g")
            result["source_class"] = rec.get("class", "unknown")
            result["persona"] = rec.get("persona", "")
            result["category"] = rec.get("category", "")
            with open(LIVE, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            s = result.get("v4", {}).get("S", 0)
            d = result.get("v4", {}).get("decision", "?")
            div = "DIV" if result.get("divergence") else "ok"
            print(f"S={s:.3f} {d} [{div}]", flush=True)
            batch.append(result)
            done += 1
        except Exception as e:
            print(f"FAIL:{type(e).__name__[:15]} — SKIP", flush=True)
            skip += 1

        time.sleep(0.05)

    # End-of-cycle metrics
    if batch:
        n = len(batch)
        dv = sum(1 for r in batch if r.get("divergence"))
        ba = sum(1 for r in batch if r.get("v4",{}).get("decision") == "accept" and r.get("source_class") == "bad")
        gr = sum(1 for r in batch if r.get("v4",{}).get("decision") == "reject" and r.get("source_class") == "good")
        m = {
            "chunk": done, "size": n,
            "bad_accepted": ba, "good_rejected": gr,
            "div_rate": round(dv / n, 4) if n else 0,
            "done": done, "skip": skip,
            "remaining": len(pending) - done - skip,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with open(METRICS, "a") as f:
            f.write(json.dumps(m) + "\n")

    print(f"  cycle done: +{done} skip={skip}", flush=True)

    # If nothing was processed, stop
    if done == 0 and skip == 0:
        break

print(f"[{datetime.now(timezone.utc).isoformat()}] EXIT", flush=True)
