#!/usr/bin/env python3
"""failure_mode_analysis.py — Cluster disagreement cases by failure mode type.

Reads: logs/disagreement_cases.jsonl + logs/shadow_eval_live.jsonl
Writes: logs/failure_mode_analysis.json

Cluster categories (derived from prompt content + S-score patterns):
  1. vague_ambiguous     — underspecified prompts ("help", "fix this", "explain more")
  2. scope_overreach     — impossibly broad ("teach me everything", "complete guide")
  3. borderline_accept   — v4 accepts near threshold, v1 catches (high-impact)
  4. underspecified_tech — vague tech requests ("make it faster", "set up CI/CD")
  5. impossible_request  — physically/logically impossible ("infinite scale", "0 latency")
  6. trick_question      — gotcha/riddle prompts ("where do you bury survivors?")
  7. opinion_debate      — subjective/contested ("is TS worth it?", "framework choice")
  8. domain_knowledge    — specific factual questions near threshold boundary
  9. debug_underspecified — debugging without context ("it's broken", "500 error")
  10. confused_user      — non-technical user, unclear intent ("the thing won't let me in")

Usage: python scripts/failure_mode_analysis.py
"""
import json
import os
import re
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISAGREEMENT_PATH = os.path.join(BASE, "logs", "disagreement_cases.jsonl")
LIVE_LOG_PATH = os.path.join(BASE, "logs", "shadow_eval_live.jsonl")
OUTPUT_PATH = os.path.join(BASE, "logs", "failure_mode_analysis.json")


def load_jsonl(path):
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


def classify_prompt(prompt: str, s_v4: float, dec_v4: str, dec_v1: str) -> str:
    """Classify a disagreement into a failure mode cluster."""
    p = prompt.lower().strip()
    s = s_v4

    # ── Category classifiers (ordered by specificity) ──

    # 1. Trick/gotcha questions
    trick_patterns = [
        r"bury.*survivor", r"plane.*crash.*border", r"word.*word.*word",
        r"compile.*breakfast", r"colorless green ideas",
    ]
    for pat in trick_patterns:
        if re.search(pat, p):
            return "trick_question"

    # 2. Impossible/absurd requests
    impossible_patterns = [
        r"infinite", r"zero latency", r"solve everything", r"complete guide to life",
        r"teach me everything", r"impossible", r"perpetual",
    ]
    for pat in impossible_patterns:
        if re.search(pat, p):
            return "impossible_request"

    # 3. Scope overreach (impossibly broad)
    scope_patterns = [
        r"teach me everything", r"complete guide", r"explain everything",
        r"tell me everything", r"write a book", r"what do i need to know about",
        r"all about", r"tell me what i should know",
    ]
    for pat in scope_patterns:
        if re.search(pat, p):
            return "scope_overreach"

    # 4. Vague debugging (no context)
    debug_patterns = [
        r"fix my code", r"it'?s broken", r"doesn'?t work", r"why doesn'?t it",
        r"help.*it'?s broken", r"500 error", r"error.*what('?s)? the problem",
        r"undefined is not a function", r"race condition", r"make (it|this) (fast|bigger)",
        r"show me how it works",
    ]
    for pat in debug_patterns:
        if re.search(pat, p):
            return "debug_underspecified"

    # 5. Vague/ambiguous requests (very short, no specifics)
    vague_patterns = [
        r"^(can you )?(help|explain more|just tell me|fix this|improve this|optimize this|debug|show me)",
        r"^(what should i|tell me what|what do i|how do i( make| get| handle))",
        r"^(write (a |me )?(function|code|test))",
        r"^(implement|set up|design|build) (a |me |my )?",
    ]
    is_short = len(prompt.split()) <= 6
    is_vague_start = any(re.search(pat, p) for pat in vague_patterns)

    if is_vague_start and is_short:
        return "vague_ambiguous"
    if is_vague_start:
        return "underspecified_tech"

    # 6. Opinion/debate prompts
    opinion_patterns = [
        r"(worth|should i|better|which is|your take|what do you think|opinion)",
        r"(some (say|developers|people)|debate|controversial|argue)",
        r"(typescript.*worth|framework.*should|sql.*programming language)",
    ]
    for pat in opinion_patterns:
        if re.search(pat, p):
            return "opinion_debate"

    # 7. Confused user (non-technical, imprecise language)
    confused_patterns = [
        r"the thing", r"won'?t let me", r"my phone", r"my computer",
        r"my daughter says", r"my son says", r"i need the one",
        r"doing something weird", r"letters.*bigger", r"accept cookie",
        r"first (smartphone|computer|time)", r"retiring",
        r"password manager", r"2-factor|two.factor|2fa",
        r"why does (everything|technology|website).*",
        r"(bank|email).*link", r"signal",
    ]
    for pat in confused_patterns:
        if re.search(pat, p):
            return "confused_user"

    # 8. Domain knowledge near threshold (specific factual questions)
    # These are borderline cases where v4 and v1 diverge on well-formed questions
    domain_patterns = [
        r"what (is|are|causes|year|does|'?s)",
        r"how (do|does|many|much|to|what|why)",
        r"difference between",
        r"explain.*step.by.step",
        r"(sum|area|calculate|percentage|\d+%|\d+ \*|\d+ \+)",
        r"(stack|queue|binary search|data structure|algorithm)",
        r"(http|https|sql|bitcoin|docker|ci/cd|machine learning|ai|javascript|python)",
        r"(iphone|laptop|backup|icloud|authentication|async)",
        r"(tides|speed of light|climate|medicine|law|history)",
    ]
    has_domain_signal = any(re.search(pat, p) for pat in domain_patterns)
    is_near_threshold = (0.55 <= s <= 0.80)  # in the review zone
    if has_domain_signal and is_near_threshold:
        return "domain_knowledge"

    # 9. Borderline accept (v4 accepts with low confidence, v1 catches)
    if dec_v4 == "accept" and s < 0.80:
        return "borderline_accept"

    # 10. Fallback: underspecified tech
    return "underspecified_tech"


def cluster_analysis(cases):
    """Cluster all disagreement cases and compute per-cluster statistics."""
    clusters = defaultdict(list)

    for c in cases:
        prompt = c.get("prompt", "")
        s_v4 = c.get("v4", {}).get("S", 0)
        dec_v4 = c.get("v4", {}).get("decision", "")
        dec_v1 = c.get("v1", {}).get("decision", "")
        mode = classify_prompt(prompt, s_v4, dec_v4, dec_v1)
        c["_cluster"] = mode
        clusters[mode].append(c)

    return dict(clusters)


def cluster_stats(clusters):
    """Compute per-cluster statistics."""
    stats = {}
    for mode, cases in sorted(clusters.items(), key=lambda x: -len(x[1])):
        total = len(cases)
        high_impact = sum(1 for c in cases if c.get("is_high_impact", False))
        cross_tier = sum(1 for c in cases if c.get("is_cross_tier", False))

        s_v4_values = [c.get("v4", {}).get("S", 0) for c in cases]
        s_v1_values = [c.get("v1", {}).get("S", 0) for c in cases]
        s_delta_values = [c.get("S_delta", 0) for c in cases]

        # Direction: v4_more_lenient means v4 gives higher S than v1
        v4_more_lenient = sum(1 for c in cases if c.get("S_delta", 0) > 0)
        v1_more_lenient = sum(1 for c in cases if c.get("S_delta", 0) < 0)

        # Decision distribution
        dec_pairs = Counter(
            f"{c.get('v4', {}).get('decision', '?')}_vs_{c.get('v1', {}).get('decision', '?')}"
            for c in cases
        )

        # Safe decision distribution
        safe_dec = Counter(c.get("safe_decision", "?") for c in cases)

        stats[mode] = {
            "count": total,
            "high_impact_count": high_impact,
            "high_impact_pct": round(high_impact / total * 100, 1) if total > 0 else 0,
            "cross_tier_count": cross_tier,
            "v4_more_lenient": v4_more_lenient,
            "v1_more_lenient": v1_more_lenient,
            "S_v4_mean": round(sum(s_v4_values) / len(s_v4_values), 4) if s_v4_values else 0,
            "S_v4_std": round(_std(s_v4_values), 4) if len(s_v4_values) > 1 else 0,
            "S_v1_mean": round(sum(s_v1_values) / len(s_v1_values), 4) if s_v1_values else 0,
            "S_delta_mean": round(sum(s_delta_values) / len(s_delta_values), 4) if s_delta_values else 0,
            "decision_pairs": dict(dec_pairs.most_common()),
            "safe_decisions": dict(safe_dec.most_common()),
            "sample_prompts": [c.get("prompt", "")[:80] for c in cases[:3]],
        }

    return stats


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def temporal_trends(cases):
    """Analyze disagreement trends over time (by date)."""
    by_date = defaultdict(lambda: {"total": 0, "high_impact": 0, "S_delta_sum": 0})
    for c in cases:
        ts = c.get("timestamp", "")
        if not ts:
            continue
        date_str = ts[:10]  # YYYY-MM-DD
        by_date[date_str]["total"] += 1
        by_date[date_str]["high_impact"] += 1 if c.get("is_high_impact") else 0
        by_date[date_str]["S_delta_sum"] += c.get("S_delta", 0)

    trends = []
    for date in sorted(by_date.keys()):
        d = by_date[date]
        trends.append({
            "date": date,
            "total_disagreements": d["total"],
            "high_impact": d["high_impact"],
            "avg_S_delta": round(d["S_delta_sum"] / d["total"], 4) if d["total"] > 0 else 0,
        })

    return trends


def high_impact_deep_dive(cases):
    """Detailed analysis of high-impact cases."""
    hi = [c for c in cases if c.get("is_high_impact", False)]

    # Classify high-impact by direction
    v4_accepts_v1_reviews = []
    v4_rejects_v1_reviews = []
    v4_rejects_v1_other = []

    for c in hi:
        dec_v4 = c.get("v4", {}).get("decision", "")
        dec_v1 = c.get("v1", {}).get("decision", "")
        s_v4 = c.get("v4", {}).get("S", 0)

        if dec_v4 == "accept" and dec_v1 == "review":
            v4_accepts_v1_reviews.append(c)
        elif dec_v4 == "reject":
            v4_rejects_v1_reviews.append(c)

    # Risk assessment
    # v4 accepting near threshold while v1 catches = false positive risk
    false_accept_risk = len(v4_accepts_v1_reviews)
    # v4 rejecting while v1 reviews = potential over-rejection
    over_rejection_risk = len(v4_rejects_v1_reviews)

    # S-score distribution for high-impact cases
    hi_s_scores = [c.get("v4", {}).get("S", 0) for c in hi]

    return {
        "total_high_impact": len(hi),
        "false_accept_risk": {
            "count": false_accept_risk,
            "description": "v4=accept near threshold, v1=review (v1 more conservative)",
            "cases": [
                {
                    "prompt": c.get("prompt", "")[:100],
                    "S_v4": c.get("v4", {}).get("S", 0),
                    "S_v1": c.get("v1", {}).get("S", 0),
                    "safe_decision": c.get("safe_decision", ""),
                }
                for c in v4_accepts_v1_reviews[:5]
            ],
        },
        "over_rejection_risk": {
            "count": over_rejection_risk,
            "description": "v4=reject, v1=review (v4 more conservative)",
            "cases": [
                {
                    "prompt": c.get("prompt", "")[:100],
                    "S_v4": c.get("v4", {}).get("S", 0),
                    "S_v1": c.get("v1", {}).get("S", 0),
                    "safe_decision": c.get("safe_decision", ""),
                }
                for c in v4_rejects_v1_reviews[:5]
            ],
        },
        "S_v4_distribution": {
            "mean": round(sum(hi_s_scores) / len(hi_s_scores), 4) if hi_s_scores else 0,
            "min": round(min(hi_s_scores), 4) if hi_s_scores else 0,
            "max": round(max(hi_s_scores), 4) if hi_s_scores else 0,
        },
    }


def v4_vs_ps_gap(cases):
    """Measure the systematic gap between v4 (π_E) and v1 (π_S)."""
    if not cases:
        return {}

    s_deltas = [c.get("S_delta", 0) for c in cases]
    # Positive delta = v4 scores higher (more lenient)
    # Negative delta = v1 scores higher (v1 more lenient)

    v4_higher = sum(1 for d in s_deltas if d > 0)
    v1_higher = sum(1 for d in s_deltas if d < 0)
    tied = sum(1 for d in s_deltas if d == 0)

    # Magnitude of gap
    abs_deltas = [abs(d) for d in s_deltas]
    mean_gap = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0

    return {
        "total_disagreements": len(cases),
        "v4_more_lenient_count": v4_higher,
        "v1_more_lenient_count": v1_higher,
        "tied_count": tied,
        "v4_more_lenient_pct": round(v4_higher / len(cases) * 100, 1) if cases else 0,
        "mean_abs_S_delta": round(mean_gap, 4),
        "median_abs_S_delta": round(sorted(abs_deltas)[len(abs_deltas)//2], 4) if abs_deltas else 0,
        "max_S_delta": round(max(s_deltas), 4) if s_deltas else 0,
        "min_S_delta": round(min(s_deltas), 4) if s_deltas else 0,
        "interpretation": (
            "v4 systematically MORE lenient than v1" if v4_higher > v1_higher * 2
            else "v4 systematically MORE conservative than v1" if v1_higher > v4_higher * 2
            else "v4 and v1 show bidirectional disagreement (no systematic bias)"
        ),
    }


def main():
    # Load data
    cases = load_jsonl(DISAGREEMENT_PATH)
    if not cases:
        print("No disagreement cases found. Run evaluation pipeline first.")
        return

    print(f"Loaded {len(cases)} disagreement cases")

    # Deduplicate by query_id (keep first occurrence)
    seen = set()
    unique = []
    for c in cases:
        qid = c.get("query_id", "")
        if qid and qid not in seen:
            seen.add(qid)
            unique.append(c)
    dedup_count = len(cases) - len(unique)
    if dedup_count > 0:
        print(f"  Deduplicated: {dedup_count} duplicates removed, {len(unique)} unique")
        cases = unique

    # 1. Cluster analysis
    clusters = cluster_analysis(cases)
    stats = cluster_stats(clusters)

    # 2. High-impact deep dive
    hi_analysis = high_impact_deep_dive(cases)

    # 3. v4-vs-π_S gap
    gap = v4_vs_ps_gap(cases)

    # 4. Temporal trends
    trends = temporal_trends(cases)

    # Build output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "unique_prompts": len(set(c.get("prompt", "") for c in cases)),
        "high_impact_total": sum(1 for c in cases if c.get("is_high_impact")),
        "cross_tier_total": sum(1 for c in cases if c.get("is_cross_tier")),
        "summary": {
            "dominant_mode": max(stats, key=lambda k: stats[k]["count"]),
            "most_dangerous_mode": max(
                (k for k in stats if stats[k]["high_impact_count"] > 0),
                key=lambda k: stats[k]["high_impact_pct"],
                default="none",
            ),
            "v4_systematic_bias": gap.get("interpretation", "unknown"),
        },
        "cluster_breakdown": stats,
        "high_impact_analysis": hi_analysis,
        "v4_vs_ps_gap": gap,
        "temporal_trends": trends,
    }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print report
    print(f"\n{'='*60}")
    print(f"  FAILURE MODE ANALYSIS — v2.1 (v4=π_E, v1=π_S)")
    print(f"{'='*60}")
    print(f"  Total cases:        {len(cases)}")
    print(f"  Unique prompts:     {output['unique_prompts']}")
    print(f"  High-impact:        {output['high_impact_total']}")
    print(f"  Cross-tier:         {output['cross_tier_total']}")
    print(f"")

    print(f"  CLUSTER BREAKDOWN:")
    print(f"  {'Mode':<25} {'Count':>5} {'HI%':>6} {'v4↑':>4} {'v1↑':>4} {'S_v4':>6} {'ΔS':>6}")
    print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*4} {'-'*4} {'-'*6} {'-'*6}")
    for mode, s in sorted(stats.items(), key=lambda x: -x[1]["count"]):
        print(f"  {mode:<25} {s['count']:>5} {s['high_impact_pct']:>5}% "
              f"{s['v4_more_lenient']:>4} {s['v1_more_lenient']:>4} "
              f"{s['S_v4_mean']:>6.3f} {s['S_delta_mean']:>+6.3f}")

    print(f"\n  HIGH-IMPACT ANALYSIS:")
    print(f"    False-accept risk (v4=accept, v1=review):  {hi_analysis['false_accept_risk']['count']}")
    print(f"    Over-rejection risk (v4=reject, v1=review): {hi_analysis['over_rejection_risk']['count']}")
    print(f"    S_v4 range: [{hi_analysis['S_v4_distribution']['min']:.3f}, "
          f"{hi_analysis['S_v4_distribution']['max']:.3f}]")

    print(f"\n  v4 vs π_S GAP:")
    print(f"    v4 more lenient: {gap.get('v4_more_lenient_pct', 0)}% of disagreements")
    print(f"    Mean |ΔS|:      {gap.get('mean_abs_S_delta', 0):.4f}")
    print(f"    Interpretation: {gap.get('interpretation', 'unknown')}")

    if trends:
        print(f"\n  TEMPORAL TRENDS ({len(trends)} data points):")
        for t in trends[-5:]:  # last 5 dates
            print(f"    {t['date']}: {t['total_disagreements']} disagreements, "
                  f"{t['high_impact']} high-impact, avg ΔS={t['avg_S_delta']:+.4f}")

    print(f"\n  Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
