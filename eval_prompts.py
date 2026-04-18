#!/usr/bin/env python3
"""
Step 2: Evaluate prompts from prompts_300.jsonl through survival engine.
Resume-safe. Saves after each sample.
"""
import os, json, time, sys, hashlib
from survival import SurvivalEngine, SurvivalConfig

DATASET_PATH = "calibration_dataset_300.jsonl"
PROMPTS_PATH = "prompts_300.jsonl"

def label_output(r):
    k, dL, dG = r.kappa, r.delta_L, r.delta_G
    if k > 0.70 and dL < 0.05 and dG < 0.25: return "good"
    elif k < 0.35 or dG > 0.60 or dL > 0.15: return "bad"
    else: return "borderline"

def main():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY"); sys.exit(1)

    # Load prompts
    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Load existing results
    done = {}
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec
    print(f"Already evaluated: {len(done)}", flush=True)

    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=3, n_contexts=3,
        lambda1=0.5, lambda2=0.5, tau_h=0.55, tau_l=0.20,
        survival_log_path="eval_300_log.jsonl",
        drift_history_path="eval_300_drift.jsonl",
    )
    engine = SurvivalEngine(cfg)

    total = len(prompts)
    for i, p in enumerate(prompts):
        pid = hashlib.sha256(p["prompt"].encode()).hexdigest()[:12]
        if pid in done:
            continue

        cls = p["class"]
        prompt = p["prompt"]
        print(f"[{i+1}/{total}] {cls:>10} {prompt[:55]}...", end=" ", flush=True)

        try:
            result = engine.evaluate(prompt)
            measured = label_output(result)
            record = {
                "id": pid, "prompt": prompt,
                "intended_class": cls, "measured_label": measured,
                "kappa": result.kappa, "delta_L": result.delta_L,
                "delta_G": result.delta_G, "S": result.S, "A": result.A,
                "decision": result.decision,
                "baseline_response": result.baseline_response[:300],
                "timestamp": result.timestamp,
            }
            done[pid] = record
            print(f"S={result.S:.3f} k={result.kappa:.3f} dG={result.delta_G:.3f} {measured}", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)

        # Save after each sample
        dataset = list(done.values())
        with open(DATASET_PATH, "w") as f:
            for rec in dataset:
                f.write(json.dumps(rec) + "\n")

        time.sleep(0.2)

    # Summary
    from collections import Counter
    labels = Counter(r["measured_label"] for r in dataset)
    intended = Counter(r["intended_class"] for r in dataset)
    print(f"\n{'='*60}", flush=True)
    print(f"  DONE: {len(dataset)} samples", flush=True)
    print(f"  Intended: {dict(intended)}", flush=True)
    print(f"  Measured: {dict(labels)}", flush=True)
    for label in ["good", "borderline", "bad"]:
        vals = [r["S"] for r in dataset if r["intended_class"] == label]
        if vals:
            print(f"  {label:<12} mean={sum(vals)/len(vals):.3f} min={min(vals):.3f} max={max(vals):.3f} n={len(vals)}", flush=True)

if __name__ == "__main__":
    main()
