#!/usr/bin/env python3
"""
Self-resuming 300-sample evaluation.
Uses perturbations=1, contexts=2 for maximum speed (3 API calls/sample).
Saves after EVERY sample so no data is lost on kill.
Run repeatedly — it skips already-done prompts.
"""
import os, json, time, sys, hashlib, signal
from survival import SurvivalEngine, SurvivalConfig

PROMPTS_PATH = "prompts_300_balanced.json"
DATASET_PATH = "calibration_dataset_300.jsonl"


def label_output(r):
    k, dL, dG = r.kappa, r.delta_L, r.delta_G
    if k > 0.70 and dL < 0.05 and dG < 0.25: return "good"
    elif k < 0.35 or dG > 0.60 or dL > 0.15: return "bad"
    else: return "borderline"


def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY"); sys.exit(1)

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)

    done = {}
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec

    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=1,   # MINIMUM for speed
        n_contexts=2,
        lambda1=0.5, lambda2=0.5, tau_h=0.55, tau_l=0.20,
        request_delay=0.2,
        survival_log_path="eval_300_log.jsonl",
        drift_history_path="eval_300_drift.jsonl",
    )
    engine = SurvivalEngine(cfg)

    evaluated = 0
    for i in range(start_idx, len(prompts)):
        if evaluated >= max_samples:
            break
        p = prompts[i]
        pid = hashlib.sha256(p["prompt"].encode()).hexdigest()[:12]
        if pid in done:
            continue

        print(f"[{i+1}] {p['class']:>10} ...", end=" ", flush=True)
        try:
            result = engine.evaluate(p["prompt"])
            record = {
                "id": pid, "prompt": p["prompt"],
                "intended_class": p["class"], "measured_label": label_output(result),
                "kappa": result.kappa, "delta_L": result.delta_L,
                "delta_G": result.delta_G, "S": result.S, "A": result.A,
                "decision": result.decision,
                "baseline_response": result.baseline_response[:300],
                "timestamp": result.timestamp,
            }
            done[pid] = record
            evaluated += 1
            print(f"S={result.S:.3f} ({evaluated} this run, {len(done)} total)", flush=True)
        except Exception as e:
            print(f"FAIL: {e}", flush=True)

        # Save after every sample
        with open(DATASET_PATH, "w") as f:
            for rec in done.values():
                f.write(json.dumps(rec) + "\n")

    print(f"Run complete: +{evaluated} new, {len(done)} total", flush=True)


if __name__ == "__main__":
    main()
