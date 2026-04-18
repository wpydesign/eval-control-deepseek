"""
tune.py
=======
Step 2. Pure-math grid search over lambda1, lambda2, tau_h, tau_l.
Reads calibration_dataset.jsonl (already has kappa, delta_L, delta_G).
No API calls needed.

Objective: maximize separation between good/bad/borderline classes.
Penalizes: good_rejected, bad_accepted. Rewards: bad_blocked, good_passed, borderline_reviewed.

Usage:
    python tune.py --dataset calibration_dataset.jsonl --out calibrated_config.json
"""

import json, argparse, itertools
from collections import defaultdict


def compute_S(kappa, delta_L, delta_G, lambda1, lambda2, eps_u=1e-6):
    denom = kappa + lambda1 * delta_L + lambda2 * delta_G + eps_u
    if denom == 0:
        return 0.0
    return kappa / denom


def decide_gate(S, tau_h, tau_l):
    if S > tau_h:
        return "accept"
    elif S > tau_l:
        return "review"
    else:
        return "reject"


def score_params(dataset, lambda1, lambda2, tau_h, tau_l):
    """
    Score a parameter combination against labeled dataset.

    Rewards:
        +0.45 * bad_blocked      (bad prompts correctly rejected or reviewed)
        +0.35 * good_passed      (good prompts correctly accepted or reviewed)
        +0.20 * border_reviewed  (borderline correctly sent to review)
    Penalties:
        -0.50 * good_rejected    (good prompts incorrectly rejected)
        -0.30 * bad_accepted     (bad prompts incorrectly accepted)
    """
    counts = defaultdict(int)
    total = len(dataset)

    for rec in dataset:
        S = compute_S(rec["kappa"], rec["delta_L"], rec["delta_G"],
                       lambda1, lambda2)
        rec["S_computed"] = round(S, 4)  # attach for analysis
        decision = decide_gate(S, tau_h, tau_l)
        label = rec["measured_label"]
        counts[f"{label}_{decision}"] += 1

    good_total = counts.get("good_accept", 0) + counts.get("good_review", 0) + counts.get("good_reject", 0)
    bad_total = counts.get("bad_accept", 0) + counts.get("bad_review", 0) + counts.get("bad_reject", 0)
    border_total = counts.get("borderline_accept", 0) + counts.get("borderline_review", 0) + counts.get("borderline_reject", 0)

    if good_total == 0 and bad_total == 0 and border_total == 0:
        return -999, counts

    good_pass_rate = (counts.get("good_accept", 0) + counts.get("good_review", 0)) / max(good_total, 1)
    bad_block_rate = (counts.get("bad_reject", 0) + counts.get("bad_review", 0)) / max(bad_total, 1)
    border_review_rate = counts.get("borderline_review", 0) / max(border_total, 1)

    # Core scoring function
    score = (
        0.45 * bad_block_rate +
        0.35 * good_pass_rate +
        0.20 * border_review_rate -
        0.50 * (counts.get("good_reject", 0) / max(good_total, 1)) -
        0.30 * (counts.get("bad_accept", 0) / max(bad_total, 1))
    )

    # Bonus for class separation: S should differ between classes
    good_S = [rec["S_computed"] for rec in dataset if rec["measured_label"] == "good"]
    bad_S = [rec["S_computed"] for rec in dataset if rec["measured_label"] == "bad"]
    if good_S and bad_S:
        mean_good = sum(good_S) / len(good_S)
        mean_bad = sum(bad_S) / len(bad_S)
        separation = mean_good - mean_bad
        score += 0.25 * separation  # reward wider gap

    return round(score, 4), counts


def run(dataset_path, out_path):
    print(f"Loading dataset from {dataset_path}...")
    dataset = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    print(f"  Loaded {len(dataset)} samples")

    from collections import Counter
    labels = Counter(r["measured_label"] for r in dataset)
    print(f"  Labels: {dict(labels)}")

    # Grid definition
    lambda1_vals = [0.5, 1.0, 1.5, 2.0, 3.0]
    lambda2_vals = [0.5, 1.0, 1.5, 2.0, 3.0]
    tau_h_vals = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    tau_l_vals = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

    total_combos = len(lambda1_vals) * len(lambda2_vals) * len(tau_h_vals) * len(tau_l_vals)
    print(f"\nGrid search: {total_combos} combinations...")
    print(f"  lambda1: {lambda1_vals}")
    print(f"  lambda2: {lambda2_vals}")
    print(f"  tau_h:   {tau_h_vals}")
    print(f"  tau_l:   {tau_l_vals}")

    best_score = -999
    best_params = None
    best_counts = None
    results = []

    combo = 0
    for l1, l2, th, tl in itertools.product(lambda1_vals, lambda2_vals, tau_h_vals, tau_l_vals):
        # tau_h must be > tau_l
        if th <= tl:
            continue

        combo += 1
        if combo % 200 == 0:
            print(f"  [{combo}/{total_combos}] testing...", flush=True)

        score, counts = score_params(dataset, l1, l2, th, tl)
        results.append({
            "lambda1": l1, "lambda2": l2, "tau_h": th, "tau_l": tl,
            "score": score, **counts
        })

        if score > best_score:
            best_score = score
            best_params = {"lambda1": l1, "lambda2": l2, "tau_h": th, "tau_l": tl}
            best_counts = counts

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Output calibrated config
    calibrated = {
        "best_params": best_params,
        "best_score": best_score,
        "confusion_at_best": dict(best_counts),
        "top_10": results[:10],
        "grid_info": {
            "total_combinations": total_combos,
            "valid_combinations": combo,
            "lambda1_range": lambda1_vals,
            "lambda2_range": lambda2_vals,
            "tau_h_range": tau_h_vals,
            "tau_l_range": tau_l_vals,
        }
    }

    with open(out_path, "w") as f:
        json.dump(calibrated, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print(f"  BEST PARAMETER COMBINATION")
    print(f"{'='*60}")
    print(f"  lambda1 = {best_params['lambda1']}")
    print(f"  lambda2 = {best_params['lambda2']}")
    print(f"  tau_h   = {best_params['tau_h']}")
    print(f"  tau_l   = {best_params['tau_l']}")
    print(f"  score   = {best_score}")
    print(f"\n  Confusion matrix at best params:")
    for k, v in sorted(best_counts.items()):
        print(f"    {k:<25} = {v}")

    # Print class separation
    print(f"\n  Class separation with best params:")
    for rec in dataset:
        S = compute_S(rec["kappa"], rec["delta_L"], rec["delta_G"],
                       best_params["lambda1"], best_params["lambda2"])
        rec["S_calibrated"] = round(S, 4)

    for label in ["good", "borderline", "bad"]:
        vals = [rec["S_calibrated"] for rec in dataset if rec["measured_label"] == label]
        if vals:
            mean = sum(vals) / len(vals)
            mn, mx = min(vals), max(vals)
            print(f"    {label:<12} mean={mean:.3f} range=[{mn:.3f}, {mx:.3f}] n={len(vals)}")

    print(f"\n  Full config saved to {out_path}")

    # Also save updated dataset with calibrated S values
    updated_path = out_path.replace(".json", "_calibrated.jsonl")
    with open(updated_path, "w") as f:
        for rec in dataset:
            f.write(json.dumps(rec) + "\n")
    print(f"  Updated dataset saved to {updated_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="calibration_dataset.jsonl")
    p.add_argument("--out", default="calibrated_config.json")
    args = p.parse_args()
    run(args.dataset, args.out)
