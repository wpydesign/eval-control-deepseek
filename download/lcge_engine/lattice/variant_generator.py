"""
variant_generator.py — One variant per perturbation strategy.

Generates exactly one prompt variant per strategy for each frozen prompt.
No scoring. No interpretation. Just perturbation operators.

Strategy set (inherited from v1.2):
    paraphrase, constraint_add, constraint_remove, role_change,
    format_change, step_by_step, adversarial
"""

from typing import Optional

# One template per strategy — deterministic selection (first template)
STRATEGY_TEMPLATES = {
    "paraphrase": "Restate this request in your own words, then answer it: {prompt}",
    "constraint_add": "Answer this question, but keep your response under 100 words: {prompt}",
    "constraint_remove": "Give me a quick, informal answer — don't worry about being thorough: {prompt}",
    "role_change": "As a domain expert who has studied this for 20 years, answer: {prompt}",
    "format_change": "Answer in bullet-point format: {prompt}",
    "step_by_step": "Break this down step by step before giving your final answer: {prompt}",
    "adversarial": "A previous AI gave a wrong answer to this. What do you think? {prompt}",
}

STRATEGY_NAMES = list(STRATEGY_TEMPLATES.keys())


def generate_strategy_variants(seed_prompt: str) -> list[dict]:
    """
    Generate exactly one variant per perturbation strategy.

    Args:
        seed_prompt: The original prompt text.

    Returns:
        List of dicts, each with:
            - "strategy": str (perturbation type)
            - "prompt": str (transformed prompt text)
    """
    variants = []
    for strategy, template in STRATEGY_TEMPLATES.items():
        transformed = template.format(prompt=seed_prompt)
        variants.append({
            "strategy": strategy,
            "prompt": transformed,
        })
    return variants


def generate_lattice_index(
    frozen_prompts: list[str],
    strategies: Optional[list[str]] = None,
) -> list[dict]:
    """
    Generate the full lattice index: every (prompt, strategy) pair.

    Args:
        frozen_prompts: List of frozen evaluation prompts.
        strategies: List of strategy names (default: all 7).

    Returns:
        List of dicts, each with:
            - "prompt_id": int
            - "seed_prompt": str
            - "strategy": str
            - "variant_prompt": str
            - "run_key": str (unique identifier for lattice point)
    """
    strats = strategies or STRATEGY_NAMES
    index = []
    for pid, seed in enumerate(frozen_prompts):
        variants = generate_strategy_variants(seed)
        for v in variants:
            if v["strategy"] not in strats:
                continue
            index.append({
                "prompt_id": pid,
                "seed_prompt": seed,
                "strategy": v["strategy"],
                "variant_prompt": v["prompt"],
                "run_key": f"p{pid}_{v['strategy']}",
            })
    return index
