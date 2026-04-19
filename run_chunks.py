#!/usr/bin/env python3
"""Run pending prompts via subprocess isolation. Resilient to hangs/crashes."""
import json, hashlib, os, sys, time, subprocess
from datetime import datetime, timezone

LIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "shadow_eval_live.jsonl")
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_prompts.jsonl")
METRICS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "daily_metrics.jsonl")

EVAL_SCRIPT = '''
import json, sys, hashlib
sys.path.insert(0, '.')
from survival import SurvivalEngine, SurvivalConfig
cfg = SurvivalConfig(
    zhipu_api_key="85c159bed2ad4cd6a36ec3110c842c65.Xx3EKUOkOCEf1jHo",
    provider="zhipu", request_delay=0.15, n_perturbations=2, n_contexts=3)
engine = SurvivalEngine(cfg)
rec = json.loads(sys.argv[1])
result = engine.evaluate_shadow(rec["prompt"])
result["source"] = rec.get("source","generated")
result["source_class"] = rec.get("class","unknown")
result["persona"] = rec.get("persona","")
result["category"] = rec.get("category","")
print(json.dumps(result, ensure_ascii=False))
'''

def main():
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

    total = len(pending)
    done = 0
    skip = 0
    batch_results = []
    print(f"[{datetime.now(timezone.utc).isoformat()}] START: {total} pending", flush=True)

    for idx, (qid, p, rec) in enumerate(pending):
        encoded = json.dumps({"prompt": p, "source": rec.get("source","g"),
                              "class": rec.get("class","unknown"),
                              "persona": rec.get("persona",""),
                              "category": rec.get("category","")})
        print(f"[{idx+1}/{total}] ", end="", flush=True)
        try:
            result = subprocess.run(
                ["timeout", "90", "python3", "-u", "-c", EVAL_SCRIPT, encoded],
                capture_output=True, text=True, timeout=100,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    entry = json.loads(output)
                    with open(LIVE, "a") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    s = entry.get("v4",{}).get("S",0)
                    d = entry.get("v4",{}).get("decision","?")
                    div = "DIV" if entry.get("divergence") else "ok"
                    print(f"S={s:.3f} {d} [{div}]", flush=True)
                    batch_results.append(entry)
                    done += 1
                else:
                    print("EMPTY — SKIP", flush=True)
                    skip += 1
            else:
                print(f"FAIL(rc={result.returncode}) — SKIP", flush=True)
                skip += 1
        except subprocess.TimeoutExpired:
            print("TIMEOUT — SKIP", flush=True)
            skip += 1
        except Exception as e:
            print(f"ERR — SKIP", flush=True)
            skip += 1

        evaluated.add(qid)
        time.sleep(0.1)

        # Metrics every 20
        if (idx + 1) % 20 == 0 and batch_results:
            n = len(batch_results)
            dv = sum(1 for r in batch_results if r.get("divergence"))
            ba = sum(1 for r in batch_results if r.get("v4",{}).get("decision")=="accept" and r.get("source_class")=="bad")
            gr = sum(1 for r in batch_results if r.get("v4",{}).get("decision")=="reject" and r.get("source_class")=="good")
            m = {"chunk": (idx+1)//20, "size": n,
                 "bad_accepted": ba, "good_rejected": gr,
                 "divergence_rate": round(dv/n,4) if n else 0,
                 "done": done, "skip": skip, "rem": total-done-skip,
                 "timestamp": datetime.now(timezone.utc).isoformat()}
            with open(METRICS, "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"  === done={done} skip={skip} rem={total-done-skip} ===", flush=True)

    # Final
    if batch_results:
        n = len(batch_results)
        dv = sum(1 for r in batch_results if r.get("divergence"))
        ba = sum(1 for r in batch_results if r.get("v4",{}).get("decision")=="accept" and r.get("source_class")=="bad")
        gr = sum(1 for r in batch_results if r.get("v4",{}).get("decision")=="reject" and r.get("source_class")=="good")
        m = {"final": True, "size": n,
             "bad_accepted": ba, "good_rejected": gr,
             "divergence_rate": round(dv/n,4) if n else 0,
             "done": done, "skip": skip,
             "timestamp": datetime.now(timezone.utc).isoformat()}
        with open(METRICS, "a") as f:
            f.write(json.dumps(m) + "\n")

    print(f"[{datetime.now(timezone.utc).isoformat()}] FINISHED: done={done} skip={skip}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        import traceback; traceback.print_exc(flush=True)
        sys.exit(1)
