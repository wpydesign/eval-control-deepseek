#!/usr/bin/env python3
"""
Batch runner for 300-sample calibration.
Runs in batches of 6 to avoid timeouts.
Saves intermediate results after each batch.
"""
import os, json, time, sys, random, hashlib
from datetime import datetime, timezone
from survival import SurvivalEngine, SurvivalConfig

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


def label_output(result) -> str:
    k = result.kappa
    dL = result.delta_L
    dG = result.delta_G
    if k > 0.70 and dL < 0.05 and dG < 0.25:
        return "good"
    elif k < 0.35 or dG > 0.60 or dL > 0.15:
        return "bad"
    else:
        return "borderline"


def generate_prompt_variants(engine, seed, n):
    meta_prompt = f"""Generate {n} distinct rephrasing of this prompt, keeping the same ambiguity level.
Return ONLY a JSON array of strings, nothing else.
Original: "{seed}"
Output format: ["variant 1", "variant 2", ...]"""
    raw = engine.client.generate(meta_prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    try:
        variants = json.loads(raw.strip())
        return [str(v) for v in variants if v][:n]
    except Exception:
        return [seed] * min(n, 1)


def main():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY")
        sys.exit(1)

    out_path = "calibration_dataset_300.jsonl"
    per_class = 100
    batch_size = 6  # evaluate 6 prompts per batch to avoid timeout

    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=4,
        n_contexts=3,
        lambda1=0.5, lambda2=0.5, tau_h=0.55, tau_l=0.20,
        survival_log_path="calibration_300_run_log.jsonl",
        drift_history_path="calibration_300_drift.jsonl",
    )
    engine = SurvivalEngine(cfg)

    # Load existing results if any (resume support)
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing[rec["id"]] = rec
        print(f"Loaded {len(existing)} existing results")

    dataset = list(existing.values())
    done_ids = set(existing.keys())

    class_seeds = [
        ("good", SEED_GOOD),
        ("bad", SEED_BAD),
        ("borderline", SEED_BORDERLINE),
    ]

    for intended_class, seeds in class_seeds:
        # Count how many we already have for this class
        existing_count = sum(1 for r in dataset if r["intended_class"] == intended_class)
        if existing_count >= per_class:
            print(f"\n── {intended_class}: already have {existing_count}/{per_class}, skipping ──")
            continue

        print(f"\n── Generating {intended_class} prompts (have {existing_count}, need {per_class}) ──")
        prompts = []

        for seed in seeds:
            if existing_count + len(prompts) >= per_class:
                break
            needed = min(3, per_class - existing_count - len(prompts))
            variants = generate_prompt_variants(engine, seed, needed)
            prompts.extend(variants)
            print(f"  seed: {seed[:50]}... -> {len(variants)} variants")
            time.sleep(0.5)

        # Deduplicate against existing
        new_prompts = []
        for p in prompts:
            pid = hashlib.sha256(p.encode()).hexdigest()[:12]
            if pid not in done_ids:
                new_prompts.append(p)
        new_prompts = new_prompts[:per_class - existing_count]
        random.shuffle(new_prompts)

        print(f"  {len(new_prompts)} new prompts to evaluate")

        # Process in batches of batch_size
        for batch_start in range(0, len(new_prompts), batch_size):
            batch = new_prompts[batch_start:batch_start + batch_size]
            print(f"\n  Batch {batch_start//batch_size + 1}/{(len(new_prompts)-1)//batch_size + 1} "
                  f"({len(batch)} prompts)")

            for i, prompt in enumerate(batch):
                pid = hashlib.sha256(prompt.encode()).hexdigest()[:12]
                if pid in done_ids:
                    continue

                print(f"    [{i+1}/{len(batch)}] {prompt[:60]}...", end=" ", flush=True)
                try:
                    result = engine.evaluate(prompt)
                    measured_label = label_output(result)
                    record = {
                        "id": pid,
                        "prompt": prompt,
                        "intended_class": intended_class,
                        "measured_label": measured_label,
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
                    print(f"S={result.S:.3f} k={result.kappa:.3f} dG={result.delta_G:.3f} "
                          f"label={measured_label}")
                except Exception as e:
                    print(f"FAILED: {e}")
                time.sleep(0.3)

            # Save after each batch
            with open(out_path, "w") as f:
                for rec in dataset:
                    f.write(json.dumps(rec) + "\n")
            print(f"  [saved {len(dataset)} total to {out_path}]")

    # Final summary
    from collections import Counter
    labels = Counter(r["measured_label"] for r in dataset)
    intended = Counter(r["intended_class"] for r in dataset)
    print(f"\n{'='*60}")
    print(f"  DATASET COMPLETE: {len(dataset)} samples")
    print(f"{'='*60}")
    print(f"  Intended:  good={intended['good']} borderline={intended['borderline']} bad={intended['bad']}")
    print(f"  Measured:  good={labels['good']} borderline={labels['borderline']} bad={labels['bad']}")
    print(f"  Saved to: {out_path}")

    mismatches = [r for r in dataset if r["intended_class"] != r["measured_label"]]
    print(f"\n  Cross-class mismatches: {len(mismatches)}/{len(dataset)}")

    # Per-class stats
    for label in ["good", "borderline", "bad"]:
        vals = [r["S"] for r in dataset if r["intended_class"] == label]
        if vals:
            mean = sum(vals) / len(vals)
            mn, mx = min(vals), max(vals)
            print(f"  {label:<12} mean_S={mean:.3f} range=[{mn:.3f}, {mx:.3f}] n={len(vals)}")


if __name__ == "__main__":
    main()
