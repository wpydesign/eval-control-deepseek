#!/usr/bin/env python3
"""
manifold_kpi.py — Manifold-level KPI tracker [v2.6.1]

Replaces global metrics (AUC, ECE) with manifold-specific KPIs.
Global metrics are now misleading — they average over three different
failure geometries with fundamentally different characteristics.

Primary KPIs (v2.6.1):
  1. router_drift_rate      — monitors manifold GEOMETRY stability
  2. ref_decay              — monitors REFERENCE TRUTH (this decides everything)
  3. contradiction_wrong_rate — PRIMARY domain KPI: are we catching failures?
  4. overconfidence_capture  — STABILITY CHECK: should be ~100%
  5. boundary_accept_rate    — STABILITY CHECK: should be high (~80%+)

Secondary metrics:
  - contradiction_precision — cost of escalation
  - ref_accuracy / live_accuracy — router vs router accuracy
  - manifold_distribution   — are we allocating labels correctly?

Design principle:
  - If ref_decay drops below -0.10 → π_ref is stale → controlled refresh
  - If ref_decay > 0 → π_live drifting wrong → keep current anchor
  - If router_drift_rate rising → manifolds moving (DANGER)
  - If contradiction_wrong_rate improves → system improves
  - Global AUC is NOT a decision driver anymore

Usage:
  python scripts/manifold_kpi.py                # compute current KPIs
  python scripts/manifold_kpi.py --compare prev  # compare with previous snapshot
  python scripts/manifold_kpi.py --save          # save snapshot for later comparison
"""

import json
import os
import sys
import numpy as np
from datetime import datetime, timezone
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_LABELS_PATH = os.path.join(BASE, "logs", "batch_label_results.jsonl")
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
KPI_SNAPSHOT_DIR = os.path.join(BASE, "logs", "manifold_kpi_snapshots")
DRIFT_LOG_PATH = os.path.join(BASE, "logs", "router_drift_log.jsonl")
DRIFT_STATE_PATH = os.path.join(BASE, "logs", "drift_state.json")


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


def compute_manifold_kpis():
    """Compute manifold-level KPIs from all labeled data.

    Merges batch_label_results (with failure_type) and failure_dataset
    to get the most complete picture.
    """
    # Load batch labels (have failure_type taxonomy)
    batch_labels = load_jsonl(BATCH_LABELS_PATH)

    # Load existing dataset (has is_wrong but may lack failure_type)
    existing = load_jsonl(DATASET_PATH)
    labeled_existing = [r for r in existing if r.get("is_wrong") is not None]

    # Build failure_type index from batch labels
    batch_ft = {r["query_id"]: r["failure_type"] for r in batch_labels}
    batch_disc = {r["query_id"]: r.get("disagreement_flag", False) for r in batch_labels}

    # Assign manifold to batch labels
    for r in batch_labels:
        r["is_wrong"] = r["final_label"]
        r["manifold"] = r["failure_type"]
        if r["source_channel"] == "blind_spot":
            r["manifold"] = "overconfidence"

    # Assign manifold to existing dataset
    for r in labeled_existing:
        ft = batch_ft.get(r.get("query_id", ""), "boundary")
        if ft == "overconfidence":
            r["manifold"] = "overconfidence"
        elif ft == "contradiction":
            r["manifold"] = "contradiction"
        else:
            r["manifold"] = "boundary"

    # Combine
    all_labeled = batch_labels + labeled_existing

    # Compute per-manifold KPIs
    kpis = {}
    total_labeled = len(all_labeled)

    # --- OVERCONFIDENCE MANIFOLD ---
    oc = [r for r in all_labeled if r.get("manifold") == "overconfidence"]
    oc_wrong = sum(1 for r in oc if r.get("is_wrong") == 1)
    oc_total = len(oc)
    oc_capture = oc_wrong / oc_total if oc_total > 0 else 0

    kpis["overconfidence"] = {
        "n": oc_total,
        "n_wrong": oc_wrong,
        "capture_rate": round(oc_capture, 4),
        "target": 1.0,
        "status": "PASS" if oc_capture >= 0.95 else "FAIL",
        "description": "Should be ~100% — if lower, detection is failing",
    }

    # --- CONTRADICTION MANIFOLD ---
    cd = [r for r in all_labeled if r.get("manifold") == "contradiction"]
    cd_wrong = sum(1 for r in cd if r.get("is_wrong") == 1)
    cd_correct = sum(1 for r in cd if r.get("is_wrong") == 0)
    cd_total = len(cd)

    # Recall: of all contradiction cases that are wrong, how many did we catch?
    # This depends on action policy — for now, measure label coverage
    cd_wrong_rate = cd_wrong / cd_total if cd_total > 0 else 0
    cd_precision = cd_wrong / max(cd_wrong + cd_correct, 1)

    kpis["contradiction"] = {
        "n": cd_total,
        "n_wrong": cd_wrong,
        "n_correct": cd_correct,
        "wrong_rate": round(cd_wrong_rate, 4),
        "precision": round(cd_precision, 4),
        "recall_target": 0.80,
        "status": "HIGH_VALUE" if cd_total > 0 else "NO_DATA",
        "description": "PRIMARY KPI — if this improves, system improves",
    }

    # --- BOUNDARY MANIFOLD ---
    bd = [r for r in all_labeled if r.get("manifold") == "boundary"]
    bd_wrong = sum(1 for r in bd if r.get("is_wrong") == 1)
    bd_correct = sum(1 for r in bd if r.get("is_wrong") == 0)
    bd_total = len(bd)
    bd_accept_rate = bd_correct / bd_total if bd_total > 0 else 0

    kpis["boundary"] = {
        "n": bd_total,
        "n_wrong": bd_wrong,
        "n_correct": bd_correct,
        "accept_rate": round(bd_accept_rate, 4),
        "wrong_rate": round(bd_wrong / bd_total if bd_total > 0 else 0, 4),
        "status": "STABLE" if bd_accept_rate >= 0.70 else "DEGRADED",
        "description": "STABILITY CHECK — mostly correct, do not optimize",
    }

    # --- GLOBAL SUMMARY (for reference, not as decision driver) ---
    total_wrong = sum(1 for r in all_labeled if r.get("is_wrong") == 1)
    total_correct = total_labeled - total_wrong

    kpis["global"] = {
        "n_labeled": total_labeled,
        "n_wrong": total_wrong,
        "n_correct": total_correct,
        "overall_wrong_rate": round(total_wrong / total_labeled if total_labeled > 0 else 0, 4),
        "note": "Global metrics are NOT decision drivers — see manifold-specific KPIs above",
    }

    # --- MANIFOLD DISTRIBUTION ---
    kpis["distribution"] = {
        "overconfidence_pct": round(oc_total / total_labeled if total_labeled > 0 else 0, 4),
        "contradiction_pct": round(cd_total / total_labeled if total_labeled > 0 else 0, 4),
        "boundary_pct": round(bd_total / total_labeled if total_labeled > 0 else 0, 4),
    }

    # --- ROUTER DRIFT RATE (v2.6.0: geometry KPI) ---
    drift = compute_router_drift()
    kpis["router_drift"] = drift

    # --- REFERENCE STALENESS (v2.6.1: truth KPI — this decides everything) ---
    ref_staleness = compute_ref_staleness()
    kpis["ref_staleness"] = ref_staleness

    kpis["computed_at"] = datetime.now(timezone.utc).isoformat()
    kpis["version"] = "v2.6.1"

    return kpis


def compute_ref_staleness():
    """Compute ref vs live accuracy and ref_decay from reference_router.

    Delegates to ReferenceRouter.compute_ref_accuracy() which:
      - Loads labeled samples with true manifold annotations
      - Routes each through both π_ref and π_live
      - Computes ref_accuracy, live_accuracy, ref_decay
    """
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from reference_router import ReferenceRouter
        ref = ReferenceRouter()
        return ref.compute_ref_accuracy()
    except Exception:
        return {
            "ref_accuracy": 0.0, "live_accuracy": 0.0, "ref_decay": 0.0,
            "n_evaluated": 0, "status": "NO_DATA",
            "interpretation": "Could not compute ref staleness",
        }


def compute_router_drift():
    """Compute router_drift_rate from drift log.

    Returns:
        dict with drift_rate, window_size, n_disagreements, status
    """
    if not os.path.exists(DRIFT_LOG_PATH):
        return {
            "drift_rate": 0.0,
            "window_size": 0,
            "n_disagreements": 0,
            "status": "NO_DATA",
            "interpretation": "No drift log found — run prediction with π_ref loaded",
        }

    entries = []
    with open(DRIFT_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        return {
            "drift_rate": 0.0,
            "window_size": 0,
            "n_disagreements": 0,
            "status": "NO_DATA",
            "interpretation": "Drift log is empty",
        }

    # Use last 100 entries (rolling window)
    window = entries[-100:]
    n = len(window)
    n_disc = sum(1 for e in window if e.get("manifold_disagreement", False))
    rate = n_disc / n

    # Import thresholds from reference_router
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from reference_router import DRIFT_WARNING, DRIFT_CRITICAL
    except ImportError:
        DRIFT_WARNING = 0.15
        DRIFT_CRITICAL = 0.25

    if rate >= DRIFT_CRITICAL:
        status = "CRITICAL"
        interpretation = "Manifold decomposition drifting severely — fallback to balanced sampling"
    elif rate >= DRIFT_WARNING:
        status = "WARNING"
        interpretation = "Manifold boundaries shifting — freeze acquisition weights"
    else:
        status = "STABLE"
        interpretation = "Decomposition stable — normal operation"

    # Disagreement pattern breakdown
    disc_entries = [e for e in window if e.get("manifold_disagreement", False)]
    patterns = {}
    for e in disc_entries:
        key = f"{e.get('m_ref','?')}->{e.get('m_live','?')}"
        patterns[key] = patterns.get(key, 0) + 1

    return {
        "drift_rate": round(rate, 4),
        "window_size": n,
        "n_disagreements": n_disc,
        "status": status,
        "interpretation": interpretation,
        "patterns": patterns,
    }


def print_kpis(kpis):
    """Print KPI dashboard."""
    print("=" * 65)
    print("  MANIFOLD KPI DASHBOARD [v2.6.1]")
    print("=" * 65)
    print(f"  Computed: {kpis['computed_at'][:19]}")

    # --- REFERENCE STALENESS (v2.6.1: this decides everything) ---
    staleness = kpis.get("ref_staleness", {})
    ref_decay = staleness.get("ref_decay", 0)
    staleness_status = staleness.get("status", "NO_DATA")
    staleness_icon = {
        "VALID": "OK",
        "REF_AGING": "~",
        "REF_STALE": "!!",
        "LIVE_DRIFTING_WRONG": "<<",
        "NO_DATA": "...",
        "NO_REF_ROUTER": "...",
        "NO_LABELED_DATA": "...",
    }.get(staleness_status, "?")
    print(f"\n  >> REFERENCE STALENESS (truth monitor): {staleness_icon}")
    print(f"     ref_accuracy:  {staleness.get('ref_accuracy', 0):.4f}")
    print(f"     live_accuracy: {staleness.get('live_accuracy', 0):.4f}")
    print(f"     ref_decay:     {ref_decay:+.4f}  (ref - live)")
    print(f"     n_evaluated:   {staleness.get('n_evaluated', 0)}")
    print(f"     status:        {staleness_status}")
    print(f"     meaning:       {staleness.get('interpretation', 'N/A')}")

    # --- ROUTER DRIFT (geometry monitor) ---
    drift = kpis.get("router_drift", {})
    drift_rate = drift.get("drift_rate", 0)
    drift_status = drift.get("status", "NO_DATA")
    drift_icon = {"STABLE": "OK", "WARNING": "!!", "CRITICAL": "!!!", "NO_DATA": "..."}.get(drift_status, "?")
    print(f"\n  >> ROUTER DRIFT (geometry monitor): {drift_icon}")
    print(f"     drift_rate:  {drift_rate:.4f}  ({drift.get('n_disagreements',0)}/{drift.get('window_size',0)})")
    print(f"     status:      {drift_status}")
    print(f"     meaning:     {drift.get('interpretation', 'N/A')}")
    patterns = drift.get("patterns", {})
    if patterns:
        print(f"     patterns:")
        for p, c in sorted(patterns.items(), key=lambda x: -x[1])[:5]:
            print(f"       {p:30s}: {c}")

    g = kpis["global"]
    print(f"\n  Global (reference only, NOT decision driver):")
    print(f"    Labeled: {g['n_labeled']}  Wrong: {g['n_wrong']}  "
          f"Correct: {g['n_correct']}  Wrong rate: {g['overall_wrong_rate']:.1%}")

    d = kpis["distribution"]
    print(f"\n  Manifold distribution:")
    print(f"    Overconfidence: {d['overconfidence_pct']:.1%}")
    print(f"    Contradiction:  {d['contradiction_pct']:.1%}")
    print(f"    Boundary:       {d['boundary_pct']:.1%}")

    print(f"\n  {'MANIFOLD':20s} {'KPI':20s} {'VALUE':>8s}  {'STATUS':>12s}")
    print(f"  {'-'*65}")

    oc = kpis["overconfidence"]
    print(f"  {'OVERCONFIDENCE':20s} {'capture_rate':20s} {oc['capture_rate']:>8.1%}  {oc['status']:>12s}")
    print(f"    ({oc['description']})")

    cd = kpis["contradiction"]
    print(f"\n  {'CONTRADICTION':20s} {'wrong_rate':20s} {cd['wrong_rate']:>8.1%}  {cd['status']:>12s}")
    print(f"    ({cd['description']})")
    if cd["n"] > 0:
        print(f"    n={cd['n']} wrong={cd['n_wrong']} correct={cd['n_correct']}")

    bd = kpis["boundary"]
    print(f"\n  {'BOUNDARY':20s} {'accept_rate':20s} {bd['accept_rate']:>8.1%}  {bd['status']:>12s}")
    print(f"    ({bd['description']})")
    if bd["n"] > 0:
        print(f"    n={bd['n']} wrong={bd['n_wrong']} correct={bd['n_correct']}")

    print(f"\n  {'-'*65}")
    print(f"  PRIMARY KPI: contradiction wrong_rate (higher = more failures found)")
    print(f"  STABILITY:   overconfidence capture >= 95%, boundary accept >= 70%")

    # Verdict
    issues = []
    if staleness_status == "REF_STALE":
        issues.append(f"REFERENCE STALE — ref_decay={ref_decay:+.4f} < -0.10, consider controlled refresh")
    elif staleness_status == "LIVE_DRIFTING_WRONG":
        issues.append(f"LIVE DRIFTING WRONG — ref_decay={ref_decay:+.4f} > 0, keep current anchor")
    if drift_status == "CRITICAL":
        issues.append("ROUTER DRIFT CRITICAL -- manifolds moving, fallback to 33/33/33")
    elif drift_status == "WARNING":
        issues.append("ROUTER DRIFT WARNING -- freeze acquisition weights")
    if oc["capture_rate"] < 0.95:
        issues.append("OVERCONFIDENCE CAPTURE LOW -- detection failing")
    if cd["n"] == 0:
        issues.append("NO CONTRADICTION DATA -- cannot measure primary KPI")
    if bd["accept_rate"] < 0.70:
        issues.append("BOUNDARY ACCEPT DEGRADED -- check for regression")

    if issues:
        print(f"\n  ISSUES:")
        for issue in issues:
            print(f"    !! {issue}")
    else:
        print(f"\n  STATUS: ALL CHECKS PASS")

    return issues


def save_snapshot(kpis):
    """Save KPI snapshot for temporal comparison."""
    os.makedirs(KPI_SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(KPI_SNAPSHOT_DIR, f"kpi_{ts}.json")
    with open(path, "w") as f:
        json.dump(kpis, f, indent=2, ensure_ascii=False)
    print(f"  Snapshot saved to {path}")
    return path


def compare_snapshots(kpis, prev_path):
    """Compare current KPIs with a previous snapshot."""
    if not os.path.exists(prev_path):
        print(f"  Previous snapshot not found: {prev_path}")
        return

    with open(prev_path) as f:
        prev = json.load(f)

    print(f"\n  COMPARISON: now vs {prev.get('computed_at', '?')[:19]}")
    print(f"  {'MANIFOLD':20s} {'KPI':20s} {'PREV':>8s} {'NOW':>8s} {'DELTA':>8s}")
    print(f"  {'-'*65}")

    comparisons = [
        ("overconfidence", "capture_rate"),
        ("contradiction", "wrong_rate"),
        ("boundary", "accept_rate"),
    ]

    for manifold, kpi_name in comparisons:
        curr_val = kpis.get(manifold, {}).get(kpi_name, 0)
        prev_val = prev.get(manifold, {}).get(kpi_name, 0)
        delta = curr_val - prev_val
        direction = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
        print(f"  {manifold:20s} {kpi_name:20s} {prev_val:>8.4f} {curr_val:>8.4f} {delta:>+8.4f} {direction}")


def main():
    compare_prev = None
    if "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        if idx + 1 < len(sys.argv):
            compare_prev = sys.argv[idx + 1]
        else:
            # Find most recent snapshot
            os.makedirs(KPI_SNAPSHOT_DIR, exist_ok=True)
            snapshots = sorted(os.listdir(KPI_SNAPSHOT_DIR)) if os.path.exists(KPI_SNAPSHOT_DIR) else []
            if snapshots:
                compare_prev = os.path.join(KPI_SNAPSHOT_DIR, snapshots[-1])

    do_save = "--save" in sys.argv

    kpis = compute_manifold_kpis()
    issues = print_kpis(kpis)

    if compare_prev:
        compare_snapshots(kpis, compare_prev)

    if do_save:
        save_snapshot(kpis)

    # Return non-zero if issues found
    if issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
