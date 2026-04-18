#!/usr/bin/env python3
"""
Evaluate 300 prompts through survival engine.
Batched: 10 at a time with checkpoint after each batch.
Resume-safe: skips already-evaluated prompts.
Uses reduced params: 2 perturbations, 3 contexts (stability > precision).
"""
import os, json, time, sys, hashlib
from survival import SurvivalEngine, SurvivalConfig

PROMPTS_PATH = "prompts_300_balanced.json"
DATASET_PATH = "calibration_dataset_300.jsonl"
BATCH_SIZE = 10
SLEEP_BETWEEN = 3  # seconds between batches


def label_output(r):
    k, dL, dG = r.kappa, r.delta_L, r.delta_G
    if k > 0.70 and dL < 0.05 and dG < 0.25:
        return "good"
    elif k < 0.35 or dG > 0.60 or dL > 0.15:
        return "bad"
    else:
        return "borderline"


def main():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY")
        sys.exit(1)

    # Load prompts
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Load existing results (resume support)
    done = {}
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec
    print(f"Already done: {len(done)}/300", flush=True)

    if len(done) >= len(prompts):
        print("Already complete!")
        return

    # Config: reduced params for stability
    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=2,   # reduced for speed
        n_contexts=3,
        lambda1=0.5, lambda2=0.5, tau_h=0.55, tau_l=0.20,
        request_delay=0.3,
        survival_log_path="eval_300_log.jsonl",
        drift_history_path="eval_300_drift.jsonl",
    )
    engine = SurvivalEngine(cfg)

    total = len(prompts)
    batch_num = 0
    batch_done = 0

    for i, p in enumerate(prompts):
        pid = hashlib.sha256(p["prompt"].encode()).hexdigest()[:12]
        if pid in done:
            continue

        cls = p["class"]
        prompt = p["prompt"]

        # Track batches
        if batch_done == 0:
            batch_num += 1
            print(f"\n--- Batch {batch_num} (samples {len(done)+1}-{min(len(done)+BATCH_SIZE, total)}) ---", flush=True)

        print(f"  [{len(done)+1}/{total}] {cls:>10} S=...", end=" ", flush=True)

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

        time.sleep(0.2)
        batch_done += 1

        # Save + sleep after each batch
        if batch_done >= BATCH_SIZE:
            dataset = list(done.values())
            with open(DATASET_PATH, "w") as f:
                for rec in dataset:
                    f.write(json.dumps(rec) + "\n")
            print(f"  [checkpoint: {len(done)}/300 saved]", flush=True)
            time.sleep(SLEEP_BETWEEN)
            batch_done = 0

    # Final save
    dataset = list(done.values())
    with open(DATASET_PATH, "w") as f:
        for rec in dataset:
            f.write(json.dumps(rec) + "\n")

    # Summary
    from collections import Counter
    labels = Counter(r["measured_label"] for r in dataset)
    intended = Counter(r["intended_class"] for r in dataset)
    print(f"\n{'='*60}", flush=True)
    print(f"  DONE: {len(dataset)} samples", flush=True)
    print(f"  Intended:  {dict(intended)}", flush=True)
    print(f"  Measured:  {dict(labels)}", flush=True)

    # Worst-case good vs best-case bad (CRITICAL metric per GPT)
    good_S = sorted([r["S"] for r in dataset if r["intended_class"] == "good"])
    bad_S = sorted([r["S"] for r in dataset if r["intended_class"] == "bad"], reverse=True)
    if good_S and bad_S:
        print(f"\n  CRITICAL: worst good vs best bad", flush=True)
        print(f"  worst good  S = {good_S[0]:.3f} (5th percentile: {good_S[max(0,len(good_S)//20)]:.3f})", flush=True)
        print(f"  best bad   S = {bad_S[0]:.3f} (5th percentile: {bad_S[max(0,len(bad_S)//20)]:.3f})", flush=True)
        overlap = sum(1 for g in good_S for b in bad_S if abs(g - b) < 0.05)
        print(f"  near-overlap (|diff|<0.05): {overlap} pairs", flush=True)

    for label in ["good", "borderline", "bad"]:
        vals = [r["S"] for r in dataset if r["intended_class"] == label]
        if vals:
            print(f"  {label:<12} mean={sum(vals)/len(vals):.3f} "
                  f"min={min(vals):.3f} max={max(vals):.3f} n={len(vals)}", flush=True)


if __name__ == "__main__":
    main()
