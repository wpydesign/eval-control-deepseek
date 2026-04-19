#!/usr/bin/env python3
"""weekly_report.py — Weekly stability and signal extraction report.

Reads:
  - logs/disagreement_cases.jsonl (failure mode data)
  - logs/shadow_eval_live.jsonl (full evaluation log)
  - logs/daily_metrics.jsonl (historical metrics)
  - logs/failure_mode_analysis.json (cluster analysis, if exists)

Writes:
  - logs/weekly_report_YYYY-MM-DD.json (timestamped snapshot)

Tracks week-over-week:
  1. Disagreement rate trend (per 100 evaluations)
  2. High-impact escalation trend
  3. v4-vs-π_S gap drift (mean |ΔS|)
  4. Failure mode distribution shifts
  5. Safety rule effectiveness

Usage: python scripts/weekly_report.py
"""
import json
import os
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISAGREEMENT_PATH = os.path.join(BASE, "logs", "disagreement_cases.jsonl")
LIVE_LOG_PATH = os.path.join(BASE, "logs", "shadow_eval_live.jsonl")
METRICS_PATH = os.path.join(BASE, "logs", "daily_metrics.jsonl")
ANALYSIS_PATH = os.path.join(BASE, "logs", "failure_mode_analysis.json")
REPORT_DIR = os.path.join(BASE, "logs")


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


def disagreement_rate_trend(live_cases, disagreement_cases):
    """Compute disagreement rate per time window (chunks of 100 evaluations)."""
    # Build a lookup set of disagreeing query_ids
    disagree_qids = set()
    for d in disagreement_cases:
        qid = d.get("query_id", "")
        if qid:
            disagree_qids.add(qid)

    # Chunk evaluations by order, 100 per chunk
    chunk_size = 100
    rates = []
    for i in range(0, len(live_cases), chunk_size):
        chunk = live_cases[i:i + chunk_size]
        chunk_total = len(chunk)
        chunk_disagree = sum(1 for c in chunk if c.get("query_id", "") in disagree_qids)
        rate = chunk_disagree / chunk_total if chunk_total > 0 else 0.0

        # Time range for this chunk
        timestamps = [c.get("timestamp", "") for c in chunk if c.get("timestamp")]
        date_from = timestamps[0][:10] if timestamps else "?"
        date_to = timestamps[-1][:10] if timestamps else "?"

        rates.append({
            "chunk": i // chunk_size + 1,
            "evaluations": chunk_total,
            "disagreements": chunk_disagree,
            "rate": round(rate, 4),
            "date_from": date_from,
            "date_to": date_to,
        })

    return rates


def high_impact_trend(disagreement_cases):
    """Track high-impact escalation rate over time."""
    # Group by date
    by_date = defaultdict(lambda: {"total": 0, "high_impact": 0})
    for c in disagreement_cases:
        ts = c.get("timestamp", "")
        if ts:
            by_date[ts[:10]]["total"] += 1
            by_date[ts[:10]]["high_impact"] += 1 if c.get("is_high_impact") else 0

    trend = []
    for date in sorted(by_date.keys()):
        d = by_date[date]
        hi_rate = d["high_impact"] / d["total"] if d["total"] > 0 else 0.0
        trend.append({
            "date": date,
            "total_disagreements": d["total"],
            "high_impact": d["high_impact"],
            "hi_rate": round(hi_rate, 4),
        })

    return trend


def gap_drift_trend(disagreement_cases):
    """Track the v4-vs-π_S gap (mean |S_delta|) over time."""
    by_date = defaultdict(list)
    for c in disagreement_cases:
        ts = c.get("timestamp", "")
        if ts:
            by_date[ts[:10]].append(c.get("S_delta", 0))

    trend = []
    for date in sorted(by_date.keys()):
        deltas = by_date[date]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0
        mean_abs = sum(abs(d) for d in deltas) / len(deltas) if deltas else 0
        v4_higher = sum(1 for d in deltas if d > 0)
        trend.append({
            "date": date,
            "count": len(deltas),
            "mean_S_delta": round(mean_delta, 4),
            "mean_abs_S_delta": round(mean_abs, 4),
            "v4_more_lenient_pct": round(v4_higher / len(deltas) * 100, 1) if deltas else 0,
        })

    return trend


def safety_rule_effectiveness(disagreement_cases):
    """Measure how the safety rule (shadow review on high-impact) is performing."""
    total = len(disagreement_cases)
    if total == 0:
        return {}

    hi = [c for c in disagreement_cases if c.get("is_high_impact")]
    hi_count = len(hi)

    # How many high-impact cases were escalated to review (not auto-accepted)?
    escalated = sum(1 for c in hi if c.get("safe_decision") == "review")
    still_accepted = sum(1 for c in hi if c.get("safe_decision") == "accept")

    # Of low-impact cases, how many were unnecessarily escalated?
    lo = [c for c in disagreement_cases if not c.get("is_high_impact")]
    lo_escalated = sum(1 for c in lo if c.get("safe_decision") == "review")
    lo_passed = sum(1 for c in lo if c.get("safe_decision") == "accept")

    return {
        "total_disagreements": total,
        "high_impact_total": hi_count,
        "high_impact_rate": round(hi_count / total * 100, 1),
        "safety_rule": {
            "escalated_to_review": escalated,
            "still_accepted": still_accepted,
            "escalation_rate": round(escalated / hi_count * 100, 1) if hi_count > 0 else 0,
        },
        "low_impact": {
            "total": len(lo),
            "escalated": lo_escalated,
            "passed_through": lo_passed,
            "unnecessary_escalation_rate": round(lo_escalated / len(lo) * 100, 1) if lo else 0,
        },
        "net_safety": "EFFECTIVE" if escalated > still_accepted else "NEEDS ATTENTION",
    }


def metrics_history_trend(metrics_records):
    """Extract week-over-week metrics from daily_metrics.jsonl."""
    if not metrics_records:
        return []

    return [
        {
            "timestamp": m.get("timestamp", ""),
            "total": m.get("total", 0),
            "bad_accepted_v4": m.get("bad_accepted_v4", 0),
            "bad_accepted_v1": m.get("bad_accepted_v1", 0),
            "good_rejected_v4": m.get("good_rejected_v4", 0),
            "good_rejected_v1": m.get("good_rejected_v1", 0),
            "divergence_rate": m.get("divergence_rate", 0),
            "high_impact_cases": m.get("high_impact_cases", 0),
            "action": m.get("action", ""),
        }
        for m in metrics_records
    ]


def main():
    # Load all data sources
    live_cases = load_jsonl(LIVE_LOG_PATH)
    disagreement_cases = load_jsonl(DISAGREEMENT_PATH)
    metrics_records = load_jsonl(METRICS_PATH)

    # Load failure mode analysis if available
    fma = None
    if os.path.exists(ANALYSIS_PATH):
        with open(ANALYSIS_PATH) as f:
            fma = json.load(f)

    # Deduplicate disagreement cases
    seen = set()
    unique_disagreements = []
    for c in disagreement_cases:
        qid = c.get("query_id", "")
        if qid and qid not in seen:
            seen.add(qid)
            unique_disagreements.append(c)
    disagreement_cases = unique_disagreements

    # Compute trends
    disc_rate = disagreement_rate_trend(live_cases, disagreement_cases)
    hi_trend = high_impact_trend(disagreement_cases)
    gap_trend = gap_drift_trend(disagreement_cases)
    safety = safety_rule_effectiveness(disagreement_cases)
    metrics_hist = metrics_history_trend(metrics_records)

    # Week-over-week comparison
    latest_disc_rate = disc_rate[-1]["rate"] if disc_rate else 0
    prev_disc_rate = disc_rate[-2]["rate"] if len(disc_rate) >= 2 else None
    latest_hi_pct = hi_trend[-1]["hi_rate"] if hi_trend else 0
    prev_hi_pct = hi_trend[-2]["hi_rate"] if len(hi_trend) >= 2 else None
    latest_gap = gap_trend[-1]["mean_abs_S_delta"] if gap_trend else 0
    prev_gap = gap_trend[-2]["mean_abs_S_delta"] if len(gap_trend) >= 2 else None

    # Drift detection
    drifts = []
    if prev_disc_rate is not None and latest_disc_rate - prev_disc_rate > 0.05:
        drifts.append(f"disagreement_rate UP +{(latest_disc_rate - prev_disc_rate):.1%}")
    if prev_hi_pct is not None and latest_hi_pct - prev_hi_pct > 0.05:
        drifts.append(f"high_impact_rate UP +{(latest_hi_pct - prev_hi_pct):.1%}")
    if prev_gap is not None and latest_gap - prev_gap > 0.05:
        drifts.append(f"v4-πS_gap WIDENING +{latest_gap - prev_gap:.4f}")

    # Build report
    report = {
        "report_date": datetime.now(timezone.utc).isoformat(),
        "report_type": "weekly_stability_report",
        "version": "v2.1",
        "pipeline_status": "v4 = π_E (primary), v1 = π_S (audit)",

        "overview": {
            "total_evaluations": len(live_cases),
            "total_disagreements": len(disagreement_cases),
            "overall_disagreement_rate": round(
                len(disagreement_cases) / len(live_cases), 4
            ) if live_cases else 0,
            "high_impact_total": sum(1 for c in disagreement_cases if c.get("is_high_impact")),
            "drift_alerts": drifts,
            "stability_verdict": "STABLE" if not drifts else "DRIFT DETECTED",
        },

        "disagreement_rate_trend": disc_rate,
        "high_impact_trend": hi_trend,
        "v4_vs_ps_gap_trend": gap_trend,
        "safety_rule_effectiveness": safety,
        "metrics_history": metrics_hist,

        "week_over_week": {
            "disagreement_rate": {
                "latest": latest_disc_rate,
                "previous": prev_disc_rate,
                "change": round(latest_disc_rate - prev_disc_rate, 4) if prev_disc_rate is not None else None,
            },
            "high_impact_rate": {
                "latest": latest_hi_pct,
                "previous": prev_hi_pct,
                "change": round(latest_hi_pct - prev_hi_pct, 4) if prev_hi_pct is not None else None,
            },
            "mean_gap": {
                "latest": latest_gap,
                "previous": prev_gap,
                "change": round(latest_gap - prev_gap, 4) if prev_gap is not None else None,
            },
        },

        "failure_mode_summary": {
            "dominant_mode": fma.get("summary", {}).get("dominant_mode", "N/A") if fma else "run failure_mode_analysis.py first",
            "v4_bias": fma.get("summary", {}).get("v4_systematic_bias", "N/A") if fma else "N/A",
        } if fma else None,
    }

    # Save timestamped report
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = os.path.join(REPORT_DIR, f"weekly_report_{date_str}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print report ──
    print(f"{'='*60}")
    print(f"  WEEKLY STABILITY REPORT — v2.1")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    ov = report["overview"]
    print(f"\n  OVERVIEW")
    print(f"    Total evaluations:        {ov['total_evaluations']}")
    print(f"    Total disagreements:      {ov['total_disagreements']}")
    print(f"    Disagreement rate:        {ov['overall_disagreement_rate']:.2%}")
    print(f"    High-impact cases:        {ov['high_impact_total']}")
    print(f"    Stability verdict:        {ov['stability_verdict']}")
    if drifts:
        print(f"    DRIFT ALERTS:            {'; '.join(drifts)}")

    wow = report["week_over_week"]
    print(f"\n  WEEK-OVER-WEEK")
    for metric, data in wow.items():
        lat = data["latest"]
        prev = data["previous"]
        chg = data["change"]
        arrow = "↑" if chg and chg > 0 else "↓" if chg and chg < 0 else "→"
        prev_str = f"{prev:.4f}" if prev is not None else "N/A"
        chg_str = f"{chg:+.4f}" if chg is not None else "N/A"
        print(f"    {metric:<25} {lat:.4f}  (prev: {prev_str}, Δ: {arrow} {chg_str})")

    if safety:
        print(f"\n  SAFETY RULE EFFECTIVENESS")
        sr = safety["safety_rule"]
        lo = safety["low_impact"]
        print(f"    High-impact escalated:   {sr['escalated_to_review']}/{safety['high_impact_total']} ({sr['escalation_rate']:.0f}%)")
        print(f"    Low-impact unnecessary:  {lo['escalated']}/{lo['total']} ({lo['unnecessary_escalation_rate']:.0f}%)")
        print(f"    Net safety:              {safety['net_safety']}")

    if gap_trend:
        latest = gap_trend[-1]
        print(f"\n  v4 vs π_S GAP (latest window)")
        print(f"    Mean |ΔS|:              {latest['mean_abs_S_delta']:.4f}")
        print(f"    Mean ΔS:               {latest['mean_S_delta']:+.4f}")
        print(f"    v4 more lenient:       {latest['v4_more_lenient_pct']}%")

    if fma:
        print(f"\n  FAILURE MODES")
        print(f"    Dominant mode:          {fma['summary']['dominant_mode']}")
        print(f"    v4 systematic bias:     {fma['summary']['v4_systematic_bias']}")

    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
