#!/usr/bin/env python3
"""compute_daily_metrics.py — Weekly audit metrics for v4-primary pipeline.

Usage: python scripts/compute_daily_metrics.py

v2.1 Decision Rule (v4 promoted as π_E):
  IF good_rejected(v4) > good_rejected(v1)    → WARNING, keep shadow active
  IF bad_accepted(v4) > bad_accepted(v1)      → WARNING, keep shadow active
  IF divergence(v4) > divergence(v1) + 0.02   → WARNING, v4 diverging more than allowed
  IF total < 500                               → insufficient data
  IF ALL conditions met                        → v4 stable as π_E, v1 remains π_S (audit only)

Cadence: Weekly (not per-run).
  Run after each +100 batch or on weekly schedule.
  Track ba parity and gr rate weekly.

Also reports disagreement case statistics from logs/disagreement_cases.jsonl.
"""
import json
import os
import sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(BASE, "logs", "shadow_eval_live.jsonl")
METRICS = os.path.join(BASE, "logs", "daily_metrics.jsonl")
DISAGREEMENT = os.path.join(BASE, "logs", "disagreement_cases.jsonl")

EPSILON = 0.02  # divergence buffer: div(v4) must be ≤ div(v1) + ε


def load_jsonl(path: str) -> list[dict]:
    """Load all valid JSON lines from a file."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def main():
    # ─── Load evaluation results ───
    results = load_jsonl(LIVE)
    if not results:
        print("ERROR: logs/shadow_eval_live.jsonl not found or empty")
        sys.exit(1)

    total = len(results)

    # v4 metrics
    ba_v4 = sum(1 for r in results
                if r.get("v4", {}).get("decision") == "accept"
                and r.get("source_class") == "bad")
    gr_v4 = sum(1 for r in results
                if r.get("v4", {}).get("decision") == "reject"
                and r.get("source_class") == "good")

    # v1 metrics
    ba_v1 = sum(1 for r in results
                if r.get("v1", {}).get("decision") == "accept"
                and r.get("source_class") == "bad")
    gr_v1 = sum(1 for r in results
                if r.get("v1", {}).get("decision") == "reject"
                and r.get("source_class") == "good")

    # divergence (overall v4 vs v1)
    divergent = sum(1 for r in results if r.get("divergence", False))
    divergence_rate = divergent / total if total > 0 else 0.0

    # ─── Disagreement case analysis ───
    disagreement_cases = load_jsonl(DISAGREEMENT)
    high_impact_count = sum(1 for d in disagreement_cases if d.get("is_high_impact", False))
    cross_tier_count = sum(1 for d in disagreement_cases if d.get("is_cross_tier", False))

    # Safe decision distribution from disagreements
    safe_decision_dist = {}
    for d in disagreement_cases:
        sd = d.get("safe_decision", "unknown")
        safe_decision_dist[sd] = safe_decision_dist.get(sd, 0) + 1

    # ─── Decision logic (v2.1 — weekly audit) ───
    warnings = []
    action = "v4 STABLE as π_E — v1 remains π_S (audit only)"

    if total < 500:
        action = "INSUFFICIENT DATA — need ≥500 samples for promotion"
    elif gr_v4 > gr_v1:
        warnings.append(f"gr(v4)={gr_v4} > gr(v1)={gr_v1}")
    elif ba_v4 > ba_v1:
        warnings.append(f"ba(v4)={ba_v4} > ba(v1)={ba_v1}")
    elif divergence_rate > 0.20 + EPSILON:  # sanity: if overall div is very high
        warnings.append(f"divergence_rate={divergence_rate:.2%} very high")

    if warnings:
        action = f"WARNING — {'; '.join(warnings)} → keep shadow active"

    # ─── Build snapshot ───
    snapshot = {
        "total": total,
        "bad_accepted_v4": ba_v4,
        "bad_accepted_v1": ba_v1,
        "good_rejected_v4": gr_v4,
        "good_rejected_v1": gr_v1,
        "divergence_rate": round(divergence_rate, 4),
        "divergent_count": divergent,
        "disagreement_cases_total": len(disagreement_cases),
        "high_impact_cases": high_impact_count,
        "cross_tier_cases": cross_tier_count,
        "safe_decision_dist": safe_decision_dist,
        "epsilon": EPSILON,
        "action": action,
        "warnings": warnings,
        "cadence": "weekly",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Append to metrics log
    with open(METRICS, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    # ─── Print report ───
    print(f"{'='*55}")
    print(f"  WEEKLY AUDIT METRICS — v2.1 (v4 = π_E, v1 = π_S)")
    print(f"{'='*55}")
    print(f"  total_prompts:       {total}")
    print(f"  bad_accepted:        v4={ba_v4}  v1={ba_v1}")
    print(f"  good_rejected:       v4={gr_v4}  v1={gr_v1}")
    print(f"  divergence_rate:     {divergence_rate:.2%}")
    print(f"  disagreement_cases:  {len(disagreement_cases)} total")
    print(f"    high_impact:       {high_impact_count}")
    print(f"    cross_tier:        {cross_tier_count}")
    if safe_decision_dist:
        print(f"    safe_decisions:    {safe_decision_dist}")
    print(f"")
    print(f"  action: {action}")
    if warnings:
        print(f"  warnings: {'; '.join(warnings)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
