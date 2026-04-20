#!/usr/bin/env python3
"""
audit_uncertainty.py — Segment and validate the 471 uncertainty-zone samples [v2.3.1]

This is a DATA QUALITY tool, not an architecture change.

Before any retrain, we need to know which of the 471 uncertainty samples are:
  1. Genuinely ambiguous — both models agree, both moderately confident → human-level hard
  2. Model blind spots — v4 confident, v1 disagrees → systematic disagreement pattern
  3. Low-signal noise — unstable geometry, no consistent signal → waste of label budget

Plus a label integrity check:
  - Flag samples where v4/v1 signals are internally inconsistent
  - Detect cases where kappa is very low (unstable scoring)
  - Flag extreme delta_G (ill-conditioned optimization)
  - Any sample failing integrity → excluded from training

Outputs:
  logs/uncertainty_audit.jsonl       — per-sample audit with segment + flags
  logs/uncertainty_segments.json     — summary stats per segment
  logs/integrity_exclusions.jsonl    — samples that should NOT be trained on

Usage:
    python scripts/audit_uncertainty.py
    python scripts/audit_uncertainty.py --show 20    # show details for top 20
"""

import json
import os
import sys
import numpy as np
from collections import Counter
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
ACTIVE_LEARNING_PATH = os.path.join(BASE, "logs", "active_learning_queue.jsonl")
LIVE_LOG_PATH = os.path.join(BASE, "logs", "shadow_eval_live.jsonl")
AUDIT_PATH = os.path.join(BASE, "logs", "uncertainty_audit.jsonl")
SEGMENTS_PATH = os.path.join(BASE, "logs", "uncertainty_segments.json")
EXCLUSIONS_PATH = os.path.join(BASE, "logs", "integrity_exclusions.jsonl")

UNCERTAINTY_BAND = 0.15

# Integrity thresholds
KAPPA_FLOOR = 0.15       # below this, scoring is unreliable
DELTA_G_CEIL = 0.95      # above this, optimization geometry is ill-conditioned
S_V4_FLOOR = 0.20        # below this, signal is too weak to be informative


def load_uncertainty_zone():
    """Load all samples in the uncertainty zone (uncertainty_score < 0.15)."""
    al_queue = []
    with open(ACTIVE_LEARNING_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                al_queue.append(json.loads(line))

    return {s["query_id"]: s for s in al_queue
            if s.get("uncertainty_score", 1.0) < UNCERTAINTY_BAND}


def load_live_signals():
    """Load v4/v1 signals from live eval log."""
    records = {}
    with open(LIVE_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                qid = entry.get("query_id", "")
                if qid:
                    records[qid] = entry
    return records


def segment_sample(qid, al, rec):
    """Segment a single uncertainty sample into one of 3 categories.

    Uses continuous scoring with soft thresholds:
    - ambiguous_score: high when both models agree and are moderately confident
    - blind_spot_score: high when v4 confident but v1 disagrees
    - noise_score: high when signals are weak or unstable
    """
    v4 = rec.get("v4", {})
    v1 = rec.get("v1", {})
    s_v4 = v4.get("S", 0)
    s_v1 = v1.get("S", 0)
    kappa = v4.get("kappa", 0)
    delta_G = v4.get("delta_G", 0)
    delta_L = v4.get("delta_L", 0)
    divergence = rec.get("divergence", False)
    gap = abs(s_v4 - s_v1)
    gap_norm = gap / max(s_v4, 0.01)

    # Continuous scores (0-1 range, higher = stronger signal for that category)

    # AMBIGUOUS: both models agree + moderate confidence
    # High when: S_v4 and S_v1 are both in [0.3, 0.7], gap is small, kappa is decent
    s4_in_range = 1.0 - min(abs(s_v4 - 0.5), 0.5) * 2  # peak at 0.5
    s1_in_range = 1.0 - min(abs(s_v1 - 0.5), 0.5) * 2
    agreement = 1.0 - min(gap_norm, 1.0)
    stability = min(kappa, 1.0)
    ambiguous_score = 0.3 * s4_in_range + 0.3 * s1_in_range + 0.2 * agreement + 0.2 * stability

    # BLIND SPOT: v4 confident, v1 disagrees
    # High when: S_v4 > 0.4, gap is large, divergence true
    v4_confident = min(max(s_v4 - 0.3, 0) / 0.4, 1.0)  # ramps from 0.3 to 0.7
    disagreement = min(gap_norm, 1.0)
    div_bonus = 0.3 if divergence else 0.0
    blind_spot_score = 0.4 * v4_confident + 0.3 * disagreement + div_bonus

    # NOISE: weak signal or unstable geometry
    # High when: S_v4 low, kappa low, delta_G extreme
    weak_signal = 1.0 - min(max(s_v4, 0), 0.5) * 2  # high when S_v4 < 0.25
    unstable_kappa = 1.0 - min(max(kappa, 0), 0.3) / 0.3  # high when kappa < 0.15
    extreme_dG = max(delta_G - 0.7, 0) / 0.3  # high when delta_G > 0.7
    noise_score = 0.35 * weak_signal + 0.35 * unstable_kappa + 0.3 * extreme_dG

    # Assign segment: highest score wins, with noise as tiebreaker
    scores = {
        "ambiguous": ambiguous_score,
        "blind_spot": blind_spot_score,
        "noise": noise_score,
    }
    segment = max(scores, key=scores.get)

    return segment, scores


def check_integrity(qid, al, rec):
    """Check if a sample passes label integrity requirements.

    Flags:
    - low_kappa: scoring unreliable (kappa < KAPPA_FLOOR)
    - extreme_delta_G: ill-conditioned optimization (delta_G > DELTA_G_CEIL)
    - weak_signal: S_v4 too low to be informative
    - decision_flip: v4 and v1 make opposite decisions (accept vs reject)
    - score_mismatch: S_v4 high but decision is reject (or vice versa)

    Returns:
        dict with integrity flags and pass/fail status
    """
    v4 = rec.get("v4", {})
    v1 = rec.get("v1", {})
    s_v4 = v4.get("S", 0)
    kappa = v4.get("kappa", 0)
    delta_G = v4.get("delta_G", 0)
    dec_v4 = v4.get("decision", "")
    dec_v1 = v1.get("decision", "")

    flags = {}

    if kappa < KAPPA_FLOOR:
        flags["low_kappa"] = round(kappa, 4)

    if delta_G > DELTA_G_CEIL:
        flags["extreme_delta_G"] = round(delta_G, 4)

    if s_v4 < S_V4_FLOOR:
        flags["weak_signal"] = round(s_v4, 4)

    # Decision flip: one accepts, other rejects
    if {dec_v4, dec_v1} == {"accept", "reject"}:
        flags["decision_flip"] = f"v4={dec_v4},v1={dec_v1}"

    # Score mismatch: high S but reject, or low S but accept
    if s_v4 > 0.5 and dec_v4 == "reject":
        flags["score_mismatch"] = f"S_v4={s_v4:.3f} but decision=reject"
    elif s_v4 < 0.3 and dec_v4 == "accept":
        flags["score_mismatch"] = f"S_v4={s_v4:.3f} but decision=accept"

    # Integrity passes if no critical flags
    critical_flags = {"low_kappa", "extreme_delta_G", "weak_signal"}
    has_critical = bool(set(flags.keys()) & critical_flags)

    return {
        "passes_integrity": not has_critical,
        "n_flags": len(flags),
        "flags": flags,
        "has_critical": has_critical,
    }


def main():
    show_n = 0
    if "--show" in sys.argv:
        idx = sys.argv.index("--show")
        if idx + 1 < len(sys.argv):
            show_n = int(sys.argv[idx + 1])

    print("Loading data...")
    uz = load_uncertainty_zone()
    live = load_live_signals()
    print(f"  Uncertainty zone: {len(uz)} samples")
    print(f"  Live signals:     {len(live)} records")

    # Match uncertainty samples to live signals
    matched = []
    unmatched = []
    for qid, al in uz.items():
        rec = live.get(qid)
        if rec:
            matched.append((qid, al, rec))
        else:
            unmatched.append(qid)

    print(f"  Matched:          {len(matched)}")
    print(f"  Unmatched:        {len(unmatched)}")

    # Segment and check integrity
    print("\nSegmenting and checking integrity...")
    audit_entries = []
    segments = {"ambiguous": [], "blind_spot": [], "noise": []}
    exclusions = []

    for qid, al, rec in matched:
        segment, scores = segment_sample(qid, al, rec)
        integrity = check_integrity(qid, al, rec)

        v4 = rec.get("v4", {})
        v1 = rec.get("v1", {})

        entry = {
            "query_id": qid,
            "prompt": al["prompt"],
            "segment": segment,
            "segment_scores": {k: round(v, 4) for k, v in scores.items()},
            "passes_integrity": integrity["passes_integrity"],
            "integrity_flags": integrity["flags"],
            "has_critical_flags": integrity["has_critical"],
            "risk_score": al["risk_score"],
            "uncertainty_score": al["uncertainty_score"],
            "S_v4": v4.get("S", 0),
            "S_v1": v1.get("S", 0),
            "kappa": v4.get("kappa", 0),
            "delta_G": v4.get("delta_G", 0),
            "decision_v4": v4.get("decision", ""),
            "decision_v1": v1.get("decision", ""),
            "divergence": rec.get("divergence", False),
            "source_class": al.get("source_class", "unknown"),
            "failure_mode": al.get("failure_mode", "none"),
        }

        audit_entries.append(entry)
        segments[segment].append(entry)

        if integrity["has_critical"]:
            exclusions.append(entry)

    # Sort by uncertainty_score within each segment
    for seg in segments:
        segments[seg].sort(key=lambda x: x["uncertainty_score"])

    # Report
    total = len(audit_entries)
    n_excluded = len(exclusions)
    n_clean = total - n_excluded

    print(f"\n{'='*65}")
    print(f"  UNCERTAINTY ZONE AUDIT — {total} SAMPLES")
    print(f"{'='*65}")

    print(f"\n  SEGMENTATION:")
    print(f"  {'Segment':>15s} {'Count':>6s} {'%':>5s} {'Avg risk':>10s} {'Avg unc':>8s} {'Diverge':>8s}")
    print(f"  {'-'*55}")
    for seg_name, items in segments.items():
        if items:
            avg_risk = np.mean([i["risk_score"] for i in items])
            avg_unc = np.mean([i["uncertainty_score"] for i in items])
            n_div = sum(1 for i in items if i["divergence"])
            pct = len(items) / total * 100
            print(f"  {seg_name:>15s} {len(items):>6d} {pct:>4.0f}% {avg_risk:>10.3f} {avg_unc:>8.4f} {n_div:>5d}/{len(items)}")

    print(f"\n  INTEGRITY CHECK:")
    print(f"  Pass:      {n_clean} ({n_clean/total:.0%})")
    print(f"  Excluded:  {n_excluded} ({n_excluded/total:.0%})")

    # Flag breakdown
    flag_counts = Counter()
    for e in audit_entries:
        for flag in e["integrity_flags"]:
            flag_counts[flag] += 1
    if flag_counts:
        print(f"\n  Flag breakdown:")
        for flag, count in flag_counts.most_common():
            print(f"    {flag:>25s}: {count}")

    # Top priority: ambiguous samples that pass integrity (best label candidates)
    clean_ambiguous = [e for e in segments["ambiguous"] if e["passes_integrity"]]
    clean_blind_spot = [e for e in segments["blind_spot"] if e["passes_integrity"]]
    clean_noise = [e for e in segments["noise"] if e["passes_integrity"]]

    print(f"\n  LABELING PRIORITY (integrity-checked):")
    print(f"    Ambiguous (highest value):  {len(clean_ambiguous)}")
    print(f"    Blind spot (overconfidence): {len(clean_blind_spot)}")
    print(f"    Noise (skip):                {len(clean_noise)}")

    # Show details
    if show_n > 0:
        print(f"\n  TOP {show_n} AMBIGUOUS SAMPLES (best label candidates):")
        print(f"  {'#':>3s} {'unc':>6s} {'risk':>6s} {'S_v4':>6s} {'S_v1':>6s} {'kappa':>6s} {'dG':>6s} {'flags':>20s} {'prompt':>35s}")
        print(f"  {'-'*100}")
        for i, e in enumerate(clean_ambiguous[:show_n]):
            flags_str = ",".join(e["integrity_flags"].keys()) if e["integrity_flags"] else "ok"
            prompt_preview = e["prompt"][:35].replace("\n", " ")
            print(f"  {i+1:>3d} {e['uncertainty_score']:>6.4f} {e['risk_score']:>6.3f} "
                  f"{e['S_v4']:>6.3f} {e['S_v1']:>6.3f} {e['kappa']:>6.3f} "
                  f"{e['delta_G']:>6.3f} {flags_str:>20s} {prompt_preview:>35s}")

        if clean_blind_spot:
            print(f"\n  BLIND-SPOT SAMPLES (divergence cases):")
            for i, e in enumerate(clean_blind_spot[:min(show_n, len(clean_blind_spot))]):
                prompt_preview = e["prompt"][:50].replace("\n", " ")
                print(f"  {i+1:>3d} risk={e['risk_score']:.3f} S_v4={e['S_v4']:.3f} "
                      f"S_v1={e['S_v1']:.3f} gap={abs(e['S_v4']-e['S_v1']):.3f} "
                      f"v4={e['decision_v4']} v1={e['decision_v1']}")
                print(f"      {prompt_preview}")

    # Write audit log
    os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
    with open(AUDIT_PATH, "w") as f:
        for e in audit_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Write segment summary
    seg_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": total,
        "excluded": n_excluded,
        "clean": n_clean,
        "segments": {
            name: {
                "count": len(items),
                "pct": round(len(items) / total * 100, 1),
                "clean_count": len([i for i in items if i["passes_integrity"]]),
            }
            for name, items in segments.items()
        },
        "flag_counts": dict(flag_counts),
        "labeling_priority": {
            "ambiguous": len(clean_ambiguous),
            "blind_spot": len(clean_blind_spot),
            "noise": len(clean_noise),
        },
    }
    with open(SEGMENTS_PATH, "w") as f:
        json.dump(seg_summary, f, indent=2, ensure_ascii=False)

    # Write exclusions
    os.makedirs(os.path.dirname(EXCLUSIONS_PATH), exist_ok=True)
    with open(EXCLUSIONS_PATH, "w") as f:
        for e in exclusions:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n  Audit log:      {AUDIT_PATH}")
    print(f"  Segment summary: {SEGMENTS_PATH}")
    print(f"  Exclusions:     {EXCLUSIONS_PATH}")
    print(f"\n  Architecture: FROZEN. This is a data quality tool only.")


if __name__ == "__main__":
    main()
