#!/usr/bin/env python3
"""
Evaluate a single batch of prompts through survival engine.
Called repeatedly by run_batches.sh for robust execution.
Usage:
    python3 eval_batch.py --start 0 --end 10
    python3 eval_batch.py --start 10 --end 20
    ...
"""
import os, json, time, sys, hashlib, argparse
from survival import SurvivalEngine, SurvivalConfig

PROMPTS_PATH = "prompts_300_balanced.json"
DATASET_PATH = "calibration_dataset_300.jsonl"


def label_output(r):
    k, dL, dG = r.kappa, r.delta_L, r.delta_G
    if k > 0.70 and dL < 0.05 and dG < 0.25:
        return "good"
    elif k < 0.35 or dG > 0.60 or dL > 0.15:
        return "bad"
    else:
        return "borderline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--perturb", type=int, default=2)
    parser.add_argument("--contexts", type=int, default=3)
    args = parser.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY")
        sys.exit(1)

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)

    # Load existing
    done = {}
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec

    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=args.perturb,
        n_contexts=args.contexts,
        lambda1=0.5, lambda2=0.5, tau_h=0.55, tau_l=0.20,
        request_delay=0.3,
        survival_log_path="eval_300_log.jsonl",
        drift_history_path="eval_300_drift.jsonl",
    )
    engine = SurvivalEngine(cfg)

    evaluated = 0
    for i in range(args.start, min(args.end, len(prompts))):
        p = prompts[i]
        pid = hashlib.sha256(p["prompt"].encode()).hexdigest()[:12]
        if pid in done:
            print(f"  [{i+1}] SKIP (already done)", flush=True)
            continue

        print(f"  [{i+1}/{len(prompts)}] {p['class']:>10} ...", end=" ", flush=True)
        try:
            result = engine.evaluate(p["prompt"])
            measured = label_output(result)
            record = {
                "id": pid, "prompt": p["prompt"],
                "intended_class": p["class"], "measured_label": measured,
                "kappa": result.kappa, "delta_L": result.delta_L,
                "delta_G": result.delta_G, "S": result.S, "A": result.A,
                "decision": result.decision,
                "baseline_response": result.baseline_response[:300],
                "timestamp": result.timestamp,
            }
            done[pid] = record
            print(f"S={result.S:.3f} k={result.kappa:.3f} dG={result.delta_G:.3f}", flush=True)
            evaluated += 1
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
        time.sleep(0.2)

    # Save
    dataset = list(done.values())
    with open(DATASET_PATH, "w") as f:
        for rec in dataset:
            f.write(json.dumps(rec) + "\n")

    print(f"\nBatch done: {evaluated} new, {len(done)} total", flush=True)


if __name__ == "__main__":
    main()
