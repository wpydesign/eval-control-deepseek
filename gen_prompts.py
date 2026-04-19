#!/usr/bin/env python3
"""
Step 1: Generate all prompt variants (no evaluation, just API calls to DeepSeek for rephrasing).
Output: prompts_300.jsonl with all prompts to evaluate.
"""
import os, json, time, sys
from survival import DeepSeekClient, SurvivalConfig

PER_CLASS = 100

SEEDS = {
    "good": [
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
    ],
    "bad": [
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
    ],
    "borderline": [
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
    ],
}

def gen_variants(client, seed, n):
    meta = f'Generate {n} distinct rephrasing of this prompt. Return ONLY a JSON array of strings. Original: "{seed}"'
    try:
        raw = client.generate(meta).strip()
        if raw.startswith("```"): raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"): raw = "\n".join(raw.split("\n")[:-1])
        v = json.loads(raw)
        return [str(x) for x in v if x][:n]
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)
        return [seed]

def main():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY"); sys.exit(1)

    cfg = SurvivalConfig(deepseek_api_key=key)
    client = DeepSeekClient(cfg)

    all_prompts = []
    for cls, seeds in SEEDS.items():
        print(f"\n=== {cls.upper()} ===", flush=True)
        for seed in seeds:
            needed = min(10, PER_CLASS - sum(1 for p in all_prompts if p["class"] == cls))
            if needed <= 0:
                break
            variants = gen_variants(client, seed, needed)
            for v in variants:
                all_prompts.append({"class": cls, "prompt": v, "seed": seed})
            print(f"  [{seed[:40]}...] -> {len(variants)}", flush=True)
            time.sleep(0.3)

    # Trim to per_class
    for cls in ["good", "bad", "borderline"]:
        cls_prompts = [p for p in all_prompts if p["class"] == cls]
        if len(cls_prompts) > PER_CLASS:
            # remove extras
            extra = len(cls_prompts) - PER_CLASS
            removed = 0
            all_prompts = [p for p in all_prompts if not (p["class"] == cls and removed < extra) or (removed := removed + 1) - 1 == 0]
            # Simpler approach:
            pass

    # Write
    import random
    random.shuffle(all_prompts)
    with open("prompts_300.jsonl", "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")

    from collections import Counter
    counts = Counter(p["class"] for p in all_prompts)
    print(f"\nTotal prompts generated: {len(all_prompts)}")
    print(f"  {dict(counts)}")

if __name__ == "__main__":
    main()
