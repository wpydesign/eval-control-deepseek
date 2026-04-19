#!/usr/bin/env python3
"""run_shadow_range.py — Process a specific index range of shadow prompts. Resume-safe via hash check."""
import os, json, time, sys
sys.path.insert(0, os.path.dirname(__file__))
from survival import SurvivalEngine, SurvivalConfig

DATASET = os.path.join(os.path.dirname(__file__), "dataset_shadow_200.json")
LOG = os.path.join(os.path.dirname(__file__), "shadow_200_log.jsonl")

def main():
    import argparse; parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--log", type=str, default="")
    args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: no key", flush=True); return

    log_path = args.log or LOG
    dataset = json.load(open(DATASET))
    done = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            for l in f:
                if l.strip(): done.add(json.loads(l)["query_id"])

    pending = [(i, x) for i, x in enumerate(dataset) if x["hash"] not in done and args.start <= i < args.end]
    if not pending:
        print(f"Range [{args.start},{args.end}): all done", flush=True); return

    cfg = SurvivalConfig(deepseek_api_key=key, n_perturbations=2, n_contexts=3,
                         lambda1=0.5, lambda2=0.5, tau_h=0.60, tau_l=0.20,
                         request_delay=0.3, survival_log_path="", drift_history_path="")
    engine = SurvivalEngine(cfg)

    for j, (oi, item) in enumerate(pending):
        t0 = time.time()
        try:
            r = engine.evaluate_shadow(item["prompt"], query_id=item["hash"], shadow_log_path=log_path)
            dv = " DV" if r["divergence"] else ""
            print(f"[{oi+1}/200] v1={r['v1']['S']:.3f}({r['v1']['decision']}) v4={r['v4']['S']:.3f}({r['v4']['decision']}){dv} {time.time()-t0:.0f}s | [{item['label']}] {item['prompt'][:45]}", flush=True)
        except Exception as ex:
            print(f"[{oi+1}/200] FAIL: {ex}", flush=True)
            time.sleep(3)
        time.sleep(0.1)

    done2 = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            for l in f:
                if l.strip(): done2.add(json.loads(l)["query_id"])
    print(f"Range [{args.start},{args.end}) done. Total unique: {len(done2)}", flush=True)

if __name__ == "__main__":
    main()
