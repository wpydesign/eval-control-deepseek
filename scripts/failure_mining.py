#!/usr/bin/env python3
"""
failure_mining.py — Blind-spot proxy scorer for overconfidence discovery [v2.3.0]

This module is structurally different from uncertainty sampling. While uncertainty
sampling targets the decision boundary (where the model is unsure), blind-spot
mining targets OVERCONFIDENCE REGIONS — cases where the model is certain but
structurally unstable.

The blind-spot proxy is a MODEL-INTERNAL signal (no labels required at inference):
    blind_spot_proxy = kappa_v4 × |S_v4 - S_v1| / max(S_v4, eps)

This captures: "high inter-rater agreement (model thinks stable) combined with
high version disagreement (v4 and v1 disagree about the answer)."

Key empirical properties (validated on 140 labeled samples):
    - AUC for is_wrong: 0.569 (meaningful signal, not noise)
    - Spearman correlation with risk_score: +0.226 (LOW — orthogonal to uncertainty)
    - This means it finds DIFFERENT failures than uncertainty sampling

The combination uncertainty + blind_spot covers more failure modes than either alone.

Also computes:
    - kappa_gap_dL: kappa × gap_norm × delta_L (enriched variant with label variance)
    - Structural characterization of known blind-spot failures from labeled data

Usage:
    python scripts/failure_mining.py              # score all unlabeled, output queue
    python scripts/failure_mining.py --top 20      # show top 20 only
    python scripts/failure_mining.py --validate    # validate proxy against labeled data
"""

import json
import os
import sys
import numpy as np
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
MINING_QUEUE_PATH = os.path.join(BASE, "logs", "failure_mining_queue.jsonl")


def load_all_samples():
    """Load all samples from failure_dataset.jsonl."""
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: {DATASET_PATH} not found. Run build_failure_dataset.py first.")
        sys.exit(1)
    samples = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def compute_blind_spot_proxy(s):
    """Compute the blind-spot proxy score for a single sample.

    Proxy = kappa × gap_norm
    Where:
        kappa = inter-rater agreement (high = raters agree, model thinks stable)
        gap_norm = |S_v4 - S_v1| / max(S_v4, eps) (version disagreement pressure)

    High proxy = "model says stable, but versions disagree" = blind spot.

    Returns:
        dict with proxy_score, gap_norm, kappa, and components
    """
    s_v4 = s["S_v4"]
    s_v1 = s["S_v1"]
    kappa = s["kappa_v4"]
    delta_L = s["delta_L_v4"]
    delta_G = s["delta_G_v4"]

    gap = abs(s_v4 - s_v1)
    gap_norm = gap / max(s_v4, 0.01)

    # Primary proxy: kappa × gap_norm
    proxy = kappa * gap_norm

    # Enriched variant: kappa × gap_norm × delta_L
    proxy_dL = kappa * gap_norm * delta_L

    # Structural instability indicator
    stability_paradox = kappa * gap_norm / max(s_v4, 0.01)

    return {
        "proxy_score": round(proxy, 6),
        "proxy_dL": round(proxy_dL, 6),
        "stability_paradox": round(stability_paradox, 6),
        "gap_norm": round(gap_norm, 4),
        "gap": round(gap, 4),
        "kappa": round(kappa, 4),
        "delta_L": round(delta_L, 4),
        "delta_G": round(delta_G, 4),
    }


def validate_proxy(predictor, labeled):
    """Validate proxy signal quality against labeled data.

    Checks:
    1. AUC of proxy for predicting is_wrong
    2. Orthogonality with risk_score (Spearman correlation)
    3. Blind-spot ranking: how well proxy ranks confident failures (risk<0.3, is_wrong=1)
    4. Coverage analysis: does proxy find failures that risk_score misses?
    """
    from scipy.stats import spearmanr, rankdata
    from sklearn.metrics import roc_auc_score

    proxy_scores = []
    risk_scores = []
    labels = []

    for s in labeled:
        v4 = {"S": s["S_v4"], "kappa": s["kappa_v4"],
              "delta_G": s["delta_G_v4"], "delta_L": s["delta_L_v4"]}
        v1 = {"S": s["S_v1"]}
        r = predictor.predict(v4, v1)
        bp = compute_blind_spot_proxy(s)

        proxy_scores.append(bp["proxy_score"])
        risk_scores.append(r["risk_score"])
        labels.append(s["is_wrong"])

    proxy_arr = np.array(proxy_scores)
    risk_arr = np.array(risk_scores)
    y = np.array(labels)

    # AUC
    try:
        auc_proxy = roc_auc_score(y, proxy_arr)
    except Exception:
        auc_proxy = 0.5

    try:
        auc_risk = roc_auc_score(y, risk_arr)
    except Exception:
        auc_risk = 0.5

    # Orthogonality
    corr, pval = spearmanr(proxy_arr, risk_arr)

    # Blind-spot ranking
    blind_mask = (y == 1) & (risk_arr < 0.3)
    n_blind = int(blind_mask.sum())
    proxy_ranks = rankdata(proxy_arr)
    risk_ranks = rankdata(risk_arr)

    if n_blind > 0:
        blind_proxy_rank = proxy_ranks[blind_mask].mean() / len(y) * 100
        blind_risk_rank = risk_ranks[blind_mask].mean() / len(y) * 100
    else:
        blind_proxy_rank = 50.0
        blind_risk_rank = 50.0

    # Complementary coverage: how many wrong samples are in top-50 by proxy
    # but NOT in top-50 by risk?
    top50_proxy = set(np.argsort(proxy_arr)[-50:])
    top50_risk = set(np.argsort(risk_arr)[-50:])
    proxy_only_wrong = sum(1 for i in top50_proxy - top50_risk if y[i] == 1)
    risk_only_wrong = sum(1 for i in top50_risk - top50_proxy if y[i] == 1)

    print(f"\n  {'='*50}")
    print(f"  BLIND-SPOT PROXY VALIDATION")
    print(f"  {'='*50}")
    print(f"  Proxy AUC:           {auc_proxy:.4f}")
    print(f"  Risk AUC:            {auc_risk:.4f}")
    print(f"  Spearman(proxy,risk): {corr:+.4f} (p={pval:.4f})")
    print(f"  Orthogonal:          {'YES' if abs(corr) < 0.3 else 'no'} (|rho|<0.3)")
    print(f"  Confident failures:  {n_blind} (is_wrong=1, risk<0.3)")
    print(f"    Proxy blind_rank:  {blind_proxy_rank:.0f}% (lower = proxy ranks them higher)")
    print(f"    Risk blind_rank:   {blind_risk_rank:.0f}%")
    print(f"  Complementary coverage (top-50):")
    print(f"    Proxy-only wrong:  {proxy_only_wrong} (proxy finds, risk misses)")
    print(f"    Risk-only wrong:   {risk_only_wrong} (risk finds, proxy misses)")

    # Per-component analysis of blind-spot cases
    if n_blind > 0:
        blind_idx = np.where(blind_mask)[0]
        non_blind_idx = np.where(~blind_mask)[0]
        print(f"\n  Blind-spot case characteristics (is_wrong=1, risk<0.3):")
        for feature, name in [(0, "S_v4"), (None, None)]:
            pass
        # Show feature comparison
        blind_features = []
        non_blind_features = []
        for s in labeled:
            bp = compute_blind_spot_proxy(s)
            if (s["is_wrong"] == 1):
                blind_features.append(bp)
            else:
                non_blind_features.append(bp)

        print(f"\n  {'Feature':>20s} {'Blind_spot_mean':>16s} {'Correct_mean':>14s} {'Diff':>8s}")
        print(f"  {'-'*60}")
        for key in ["proxy_score", "gap_norm", "kappa", "delta_L", "delta_G"]:
            blind_mean = np.mean([b[key] for b in blind_features])
            correct_mean = np.mean([b[key] for b in non_blind_features])
            diff = blind_mean - correct_mean
            print(f"  {key:>20s} {blind_mean:>16.4f} {correct_mean:>14.4f} {diff:>+8.4f}")

    return {
        "auc_proxy": auc_proxy,
        "auc_risk": auc_risk,
        "spearman_rho": corr,
        "spearman_p": pval,
        "n_blind_spots": n_blind,
        "blind_proxy_rank_pct": round(blind_proxy_rank, 1),
        "proxy_only_wrong_top50": proxy_only_wrong,
        "risk_only_wrong_top50": risk_only_wrong,
    }


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    from predict_failure import FailurePredictor

    do_validate = "--validate" in sys.argv
    n_top = 20
    if "--top" in sys.argv:
        idx = sys.argv.index("--top")
        if idx + 1 < len(sys.argv):
            n_top = int(sys.argv[idx + 1])

    print("Loading predictor...")
    predictor = FailurePredictor()
    if not predictor.is_loaded:
        print("ERROR: No trained model. Run train_failure_predictor.py first.")
        sys.exit(1)

    print("Loading samples...")
    samples = load_all_samples()
    labeled = [s for s in samples if s.get("is_wrong") is not None]
    unlabeled = [s for s in samples if s.get("is_wrong") is None]
    print(f"  {len(labeled)} labeled, {len(unlabeled)} unlabeled")

    # Validate proxy against labeled data
    if do_validate:
        validation = validate_proxy(predictor, labeled)
        print()

    # Score all unlabeled samples
    print("Computing blind-spot proxy scores...")
    scored = []
    for s in unlabeled:
        v4 = {"S": s['S_v4'], "kappa": s['kappa_v4'],
              "delta_G": s['delta_G_v4'], "delta_L": s['delta_L_v4']}
        v1 = {"S": s['S_v1']}
        risk = predictor.predict(v4, v1)
        bp = compute_blind_spot_proxy(s)

        scored.append({
            "query_id": s["query_id"],
            "prompt": s["prompt"],
            "proxy_score": bp["proxy_score"],
            "proxy_dL": bp["proxy_dL"],
            "stability_paradox": bp["stability_paradox"],
            "gap_norm": bp["gap_norm"],
            "kappa": bp["kappa"],
            "delta_L": bp["delta_L"],
            "risk_score": risk["risk_score"],
            "risk_action": risk["action"],
            "S_v4": s["S_v4"],
            "S_v1": s["S_v1"],
            "source_class": s.get("source_class", "unknown"),
            "failure_mode": s.get("failure_mode", "none"),
        })

    # Sort by proxy_score (higher = more likely blind spot)
    scored.sort(key=lambda x: x["proxy_score"], reverse=True)

    # Categorize
    high_proxy = [s for s in scored if s["proxy_score"] > np.percentile(
        [s["proxy_score"] for s in scored], 90)]

    print(f"\n{'='*60}")
    print(f"  BLIND-SPOT MINING QUEUE")
    print(f"{'='*60}")
    print(f"  Unlabeled samples:       {len(scored)}")
    print(f"  Top 10% proxy score:     {len(high_proxy)} samples")
    all_proxies = [s["proxy_score"] for s in scored]
    print(f"  Proxy range:            [{min(all_proxies):.4f}, {max(all_proxies):.4f}]")
    print(f"  Proxy mean:             {np.mean(all_proxies):.4f}")
    print(f"  Proxy P90:              {np.percentile(all_proxies, 90):.4f}")
    print(f"  Proxy formula:          kappa × |S_v4 - S_v1| / max(S_v4, 0.01)")

    # Top priority samples
    n_show = min(n_top, len(scored))
    print(f"\n  TOP {n_show} BLIND-SPOT CANDIDATES:")
    print(f"  {'#':>3s} {'proxy':>7s} {'risk':>6s} {'action':>12s} {'kappa':>6s} {'gap_n':>6s} {'prompt':>40s}")
    print(f"  {'-'*85}")
    for i, s in enumerate(scored[:n_show]):
        prompt_preview = s["prompt"][:40].replace("\n", " ")
        safe = "safe" if s["risk_action"] == "none" else "flag"
        print(f"  {i+1:>3d} {s['proxy_score']:>7.4f} {s['risk_score']:>6.3f} "
              f"{s['risk_action']:>12s} {s['kappa']:>6.3f} {s['gap_norm']:>6.3f} "
              f"{prompt_preview:>40s}")

    # Count how many are "safe" by risk (risk < review_threshold) — these are
    # the TRUE blind spots the model would miss
    rt = predictor._review_threshold
    safe_by_risk = [s for s in scored if s["risk_score"] < rt]
    safe_top_proxy = [s for s in scored[:50] if s["risk_score"] < rt]
    print(f"\n  Risk-safe samples (risk<{rt}): {len(safe_by_risk)}")
    print(f"  In top-50 by proxy: {len(safe_top_proxy)} (proxy finds, risk misses)")

    # Write full queue to file
    os.makedirs(os.path.dirname(MINING_QUEUE_PATH), exist_ok=True)
    with open(MINING_QUEUE_PATH, "w") as f:
        for i, s in enumerate(scored):
            entry = {
                "query_id": s["query_id"],
                "prompt": s["prompt"],
                "priority": i + 1,
                "proxy_score": s["proxy_score"],
                "proxy_dL": s["proxy_dL"],
                "stability_paradox": s["stability_paradox"],
                "gap_norm": s["gap_norm"],
                "kappa": s["kappa"],
                "delta_L": s["delta_L"],
                "risk_score": s["risk_score"],
                "risk_action": s["risk_action"],
                "source_class": s["source_class"],
                "failure_mode": s["failure_mode"],
                "is_risk_safe": s["risk_score"] < predictor._review_threshold,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n  Full queue ({len(scored)} samples) written to {MINING_QUEUE_PATH}")
    print(f"  Strategy: label high-proxy samples that risk_score marks as safe")
    print(f"  These are the model's blind spots — confident but structurally unstable")


if __name__ == "__main__":
    main()
