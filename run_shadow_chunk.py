#!/usr/bin/env python3
"""run_shadow_chunk.py — Process N pending shadow prompts then exit. Resume-safe."""
import os, json, time, sys
sys.path.insert(0, os.path.dirname(__file__))
from survival import SurvivalEngine, SurvivalConfig

DATASET = os.path.join(os.path.dirname(__file__), "dataset_shadow_200.json")
LOG = os.path.join(os.path.dirname(__file__), "shadow_200_log.jsonl")

def main():
    import argparse; parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: DEEPSEEK_API_KEY", flush=True); return

    dataset = json.load(open(DATASET))
    done = set()
    if os.path.exists(LOG):
        with open(LOG) as f:
            for l in f:
                if l.strip(): done.add(json.loads(l)["query_id"])
    pending = [x for i, x in enumerate(dataset) if x["hash"] not in done]
    batch = pending[:args.n]
    if not batch:
        print(f"All done! {len(done)}/200", flush=True); return

    cfg = SurvivalConfig(deepseek_api_key=key, n_perturbations=2, n_contexts=3,
                         lambda1=0.5, lambda2=0.5, tau_h=0.60, tau_l=0.20,
                         request_delay=0.3, survival_log_path="", drift_history_path="")
    engine = SurvivalEngine(cfg)
    for j, item in enumerate(batch):
        t0 = time.time()
        try:
            r = engine.evaluate_shadow(item["prompt"], query_id=item["hash"], shadow_log_path=LOG)
            dv = " DV" if r["divergence"] else ""
            print(f"[{len(done)+j+1}/200] v1={r['v1']['S']:.3f}({r['v1']['decision']}) v4={r['v4']['S']:.3f}({r['v4']['decision']}){dv} {time.time()-t0:.0f}s | [{item['label']}] {item['prompt'][:50]}", flush=True)
        except Exception as ex:
            print(f"[{len(done)+j+1}/200] FAIL: {ex}", flush=True)
            time.sleep(3)
        time.sleep(0.1)
    done2 = set()
    if os.path.exists(LOG):
        with open(LOG) as f:
            for l in f:
                if l.strip(): done2.add(json.loads(l)["query_id"])
    print(f"Progress: {len(done2)}/200", flush=True)

if __name__ == "__main__":
    main()
