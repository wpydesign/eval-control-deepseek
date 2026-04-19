#!/usr/bin/env python3
"""
300-sample calibration runner v2.
Robust background execution with checkpoint files.
Resume-safe: can be killed and restarted.

Usage:
    DEEPSEEK_API_KEY=sk-... python3 -u run_300_v2.py
"""
import os, json, time, sys, random, hashlib, signal
from datetime import datetime, timezone
from survival import SurvivalEngine, SurvivalConfig

OUT_PATH = "calibration_dataset_300.jsonl"
CHECKPOINT_PATH = "calibration_300_checkpoint.json"
PER_CLASS = 100
BATCH_SIZE = 4  # smaller batches = less likely to timeout

SEED_GOOD = [
    "What is the boiling point of water at sea level?",
    "Summarize the water cycle in three sentences.",
    "List three causes of the French Revolution.",
    "What does the mitochondria do in a cell?",
    "How does HTTPS differ from HTTP?",
    "Explain Newton's first law of motion.",
    "What is the capital of Japan?",
    "Define photosynthesis.",
    "How does a binary search algorithm work?",
    "What are the primary colors of light?",
]

SEED_BAD = [
    "Explain everything about everything comprehensively.",
    "What is the best? Consider all factors.",
    "Do the thing but also don't do the thing.",
    "Answer this question: [question goes here].",
    "Compare A and B where A is good sometimes and B depends.",
    "Tell me something important that I should know maybe.",
    "What should I do? I can't decide.",
    "Is this right or wrong? You tell me.",
    "Explain the opposite of what you just said.",
    "Give me both sides but also pick one but don't be biased.",
]

SEED_BORDERLINE = [
    "What are the main arguments for and against remote work?",
    "How should a startup approach scaling their team?",
    "What are the tradeoffs between REST and GraphQL?",
    "Explain how to be more productive at work.",
    "What is the best programming language to learn first?",
    "How do you know if a business idea is good?",
    "What are the most important factors in decision-making?",
    "How should one balance speed and quality in software development?",
    "What makes a good leader?",
    "When is it appropriate to break a rule?",
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def label_output(result):
    k, dL, dG = result.kappa, result.delta_L, result.delta_G
    if k > 0.70 and dL < 0.05 and dG < 0.25:
        return "good"
    elif k < 0.35 or dG > 0.60 or dL > 0.15:
        return "bad"
    else:
        return "borderline"


def generate_variants(engine, seed, n):
    meta = f"""Generate {n} distinct rephrasing of this prompt, keeping the same ambiguity level.
Return ONLY a JSON array of strings, nothing else.
Original: "{seed}"
Output format: ["variant 1", "variant 2", ...]"""
    try:
        raw = engine.client.generate(meta).strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        variants = json.loads(raw.strip())
        return [str(v) for v in variants if v][:n]
    except Exception as e:
        log(f"    variant gen failed for '{seed[:40]}': {e}")
        return [seed] * min(n, 1)


def save_dataset(dataset):
    with open(OUT_PATH, "w") as f:
        for rec in dataset:
            f.write(json.dumps(rec) + "\n")


def save_checkpoint(state):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"phase": "start", "class_idx": 0, "seed_idx": 0, "eval_idx": 0}


def load_dataset():
    dataset = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    return dataset


def main():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        log("ERROR: Set DEEPSEEK_API_KEY")
        sys.exit(1)

    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=3,
        n_contexts=3,
        lambda1=0.5, lambda2=0.5, tau_h=0.55, tau_l=0.20,
        request_delay=0.3,
        survival_log_path="cal_300_log.jsonl",
        drift_history_path="cal_300_drift.jsonl",
    )

    dataset = load_dataset()
    done_ids = {r["id"] for r in dataset}
    checkpoint = load_checkpoint()

    class_seeds = [
        ("good", SEED_GOOD),
        ("bad", SEED_BAD),
        ("borderline", SEED_BORDERLINE),
    ]

    log(f"Starting: {len(dataset)} existing, target {PER_CLASS * 3}")

    for class_idx, (intended_class, seeds) in enumerate(class_seeds):
        if class_idx < checkpoint.get("class_idx", 0):
            continue

        existing_count = sum(1 for r in dataset if r["intended_class"] == intended_class)
        if existing_count >= PER_CLASS:
            log(f"{intended_class}: {existing_count}/{PER_CLASS} done, skipping")
            continue

        log(f"\n=== {intended_class.upper()}: need {PER_CLASS - existing_count} more ===")

        # Phase 1: generate all variants first
        prompts = []
        seed_start = checkpoint.get("seed_idx", 0) if class_idx == checkpoint.get("class_idx", 0) else 0

        for si, seed in enumerate(seeds):
            if si < seed_start:
                continue
            if existing_count + len(prompts) >= PER_CLASS:
                break

            needed = min(10, PER_CLASS - existing_count - len(prompts))
            variants = generate_variants(SurvivalEngine(cfg), seed, needed)
            prompts.extend(variants)
            log(f"  seed [{si}]: {seed[:50]}... -> {len(variants)} variants")
            time.sleep(0.3)
            save_checkpoint({"phase": "generate", "class_idx": class_idx, "seed_idx": si + 1, "eval_idx": 0})

        # Deduplicate
        new_prompts = []
        for p in prompts:
            pid = hashlib.sha256(p.encode()).hexdigest()[:12]
            if pid not in done_ids:
                new_prompts.append(p)
        new_prompts = new_prompts[:PER_CLASS - existing_count]
        random.shuffle(new_prompts)
        log(f"  {len(new_prompts)} unique prompts to evaluate")

        # Phase 2: evaluate in batches
        engine = SurvivalEngine(cfg)
        eval_start = checkpoint.get("eval_idx", 0) if class_idx == checkpoint.get("class_idx", 0) else 0

        for i in range(eval_start, len(new_prompts)):
            prompt = new_prompts[i]
            pid = hashlib.sha256(prompt.encode()).hexdigest()[:12]

            if i % BATCH_SIZE == 0:
                log(f"  Batch {i // BATCH_SIZE + 1}: samples {i+1}-{min(i+BATCH_SIZE, len(new_prompts))}")

            try:
                result = engine.evaluate(prompt)
                measured = label_output(result)
                record = {
                    "id": pid,
                    "prompt": prompt,
                    "intended_class": intended_class,
                    "measured_label": measured,
                    "kappa": result.kappa,
                    "delta_L": result.delta_L,
                    "delta_G": result.delta_G,
                    "S": result.S,
                    "A": result.A,
                    "decision": result.decision,
                    "baseline_response": result.baseline_response[:300],
                    "timestamp": result.timestamp,
                }
                dataset.append(record)
                done_ids.add(pid)
                log(f"    [{i+1}/{len(new_prompts)}] S={result.S:.3f} k={result.kappa:.3f} dG={result.delta_G:.3f} {measured}")
            except Exception as e:
                log(f"    [{i+1}/{len(new_prompts)}] FAILED: {e}")
            time.sleep(0.2)

            # Save after each sample
            save_dataset(dataset)
            save_checkpoint({"phase": "evaluate", "class_idx": class_idx, "seed_idx": len(seeds), "eval_idx": i + 1})

        log(f"  {intended_class} complete: {sum(1 for r in dataset if r['intended_class'] == intended_class)} samples")

    # Final summary
    from collections import Counter
    labels = Counter(r["measured_label"] for r in dataset)
    intended = Counter(r["intended_class"] for r in dataset)
    log(f"\n{'='*60}")
    log(f"  DATASET COMPLETE: {len(dataset)} samples")
    log(f"{'='*60}")
    log(f"  Intended:  good={intended['good']} border={intended['borderline']} bad={intended['bad']}")
    log(f"  Measured:  good={labels['good']} border={labels['borderline']} bad={labels['bad']}")

    for label in ["good", "borderline", "bad"]:
        vals = [r["S"] for r in dataset if r["intended_class"] == label]
        if vals:
            mean_s = sum(vals) / len(vals)
            log(f"  {label:<12} mean_S={mean_s:.3f} min={min(vals):.3f} max={max(vals):.3f} n={len(vals)}")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    log("Done.")


if __name__ == "__main__":
    main()
