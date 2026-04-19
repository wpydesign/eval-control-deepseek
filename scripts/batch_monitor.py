#!/usr/bin/env python3
"""
batch_monitor.py — Streaming rolling-window monitor [v2.1.2]

Replaces weekly reporting with event-driven monitoring.
Runs after every batch append. No reports, just signals.

Windows:  50 (fast spike) | 200 (drift)

Alert rules (immediate, no delay):
  risk_spike        >= 2   in last 50   (factuality_risk_flag bursts)
  false_accept      >= 3   in last 200  (v4 accepting questionable outputs)
  dk_HI_rate        > 0.5 in last 200  (domain_knowledge regression)
  confidence_gap_mean > 0.15 in last 200 (v4 getting too lenient)

v2.1.2: alert → action hooks (routing layer only, no model change):
  RISK_SPIKE   → force shadow review for domain_knowledge (next 50 samples)
  FALSE_ACCEPT → tighten tau_h to 0.80 for domain_knowledge (next 50 samples)
  DK_DRIFT     → log critical, no auto-change
  GAP_DRIFT    → log leniency drift, no auto-change
  All actions: temporary, domain_knowledge-scoped, auto-expire after 50 samples.

Usage:
  python scripts/batch_monitor.py              # scan full log, print alerts
  python scripts/batch_monitor.py --watch      # tail-follow mode (for live use)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from collections import deque

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISAGREEMENT_PATH = os.path.join(BASE, "logs", "disagreement_cases.jsonl")
ALERT_LOG_PATH = os.path.join(BASE, "logs", "monitor_alerts.jsonl")

WINDOW_50 = 50
WINDOW_200 = 200
ACTION_EXPIRY = 50  # samples before auto-expire

# ── thresholds ──
RISK_SPIKE_THRESHOLD = 2
FALSE_ACCEPT_THRESHOLD = 3
DK_HI_RATE_THRESHOLD = 0.5
CONFIDENCE_GAP_THRESHOLD = 0.15
TIGHTENED_TAU_H = 0.80  # elevated accept threshold for domain_knowledge


# ═══════════════════════════════════════════════════════════════
# MONITOR STATE — alert-triggered routing actions
# ═══════════════════════════════════════════════════════════════

class MonitorState:
    """Holds active routing interventions triggered by alerts.

    All actions are:
      - temporary (auto-expire after ACTION_EXPIRY samples)
      - scoped to domain_knowledge failure mode only
      - routing-layer only (no model/scoring/gating changes)

    Usage in batch runner:
      monitor = MonitorState()
      # before each eval:
      action = monitor.get_action(failure_mode)
      # after each eval:
      monitor.tick()
      # after each batch:
      monitor.update(alerts)
    """

    def __init__(self):
        self._active = {}  # action_name -> remaining_ticks

    @property
    def active_actions(self):
        return dict(self._active)

    def get_action(self, failure_mode: str) -> str:
        """Return the routing action for this failure mode.

        Returns:
          "forced_review"       — RISK_SPIKE active, dk → override to shadow review
          "tightened_threshold" — FALSE_ACCEPT active, dk → require S >= 0.80
          "none"                — no active intervention
        """
        if failure_mode != "domain_knowledge":
            return "none"
        if self._active.get("forced_review", 0) > 0:
            return "forced_review"
        if self._active.get("tightened_threshold", 0) > 0:
            return "tightened_threshold"
        return "none"

    def tick(self):
        """Decrement all counters by 1. Call after each sample evaluation."""
        expired = [k for k, v in self._active.items() if v <= 1]
        for k in expired:
            del self._active[k]
        for k in self._active:
            self._active[k] -= 1

    def update(self, alerts: list[str]):
        """Activate new interventions from alert strings.

        When both RISK_SPIKE and FALSE_ACCEPT fire simultaneously,
        tightened_threshold is staggered to activate after forced_review expires.
        """
        has_spike = any("RISK_SPIKE" in a for a in alerts)
        has_fa = any("FALSE_ACCEPT" in a for a in alerts)

        if has_spike and "forced_review" not in self._active:
            self._active["forced_review"] = ACTION_EXPIRY

        if has_fa and "tightened_threshold" not in self._active:
            if "forced_review" in self._active:
                # stagger: start counting after forced_review expires
                self._active["tightened_threshold"] = self._active["forced_review"] + ACTION_EXPIRY
            else:
                self._active["tightened_threshold"] = ACTION_EXPIRY


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


def scan(cases):
    """Scan disagreement cases over rolling windows. Return list of alert strings."""
    alerts = []

    if not cases:
        return alerts

    w200 = cases[-WINDOW_200:]
    w50 = cases[-WINDOW_50:]

    # ── window_200 metrics ──
    dk_cases = [c for c in w200 if c.get("failure_mode") == "domain_knowledge"]
    dk_hi = [c for c in dk_cases if c.get("is_high_impact")]
    dk_hi_rate = len(dk_hi) / len(dk_cases) if dk_cases else 0.0

    false_accepts = [c for c in w200 if c.get("factuality_risk_flag")]

    gaps = [c.get("confidence_gap", c.get("S_delta", 0)) for c in w200]
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0

    # ── window_50 metrics ──
    risk_spike_count = sum(1 for c in w50 if c.get("factuality_risk_flag"))

    # ── alert checks ──
    if risk_spike_count >= RISK_SPIKE_THRESHOLD:
        alerts.append(f"[ALERT] type=RISK_SPIKE count={risk_spike_count} window=50")

    if len(false_accepts) >= FALSE_ACCEPT_THRESHOLD:
        alerts.append(f"[ALERT] type=FALSE_ACCEPT count={len(false_accepts)} window=200")

    if dk_hi_rate > DK_HI_RATE_THRESHOLD:
        alerts.append(f"[ALERT] type=DK_DRIFT rate={dk_hi_rate:.2f} window=200")

    if gap_mean > CONFIDENCE_GAP_THRESHOLD:
        alerts.append(f"[ALERT] type=GAP_DRIFT gap={gap_mean:.4f} window=200")

    return alerts


def log_alerts(alerts, path=ALERT_LOG_PATH):
    """Append alerts to JSONL log."""
    if not alerts:
        return
    ts = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            for a in alerts:
                entry = {"timestamp": ts, "alert": a}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def print_status(cases):
    """Print one-line status (always, even with no alerts)."""
    w200 = cases[-WINDOW_200:]
    w50 = cases[-WINDOW_50:]

    dk = [c for c in w200 if c.get("failure_mode") == "domain_knowledge"]
    dk_hi = [c for c in dk if c.get("is_high_impact")]
    dk_rate = len(dk_hi) / len(dk) if dk else 0.0

    fa = sum(1 for c in w200 if c.get("factuality_risk_flag"))
    spike = sum(1 for c in w50 if c.get("factuality_risk_flag"))
    gaps = [c.get("confidence_gap", c.get("S_delta", 0)) for c in w200]
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] monitor  n={len(cases):>4}  "
          f"dk_HI={dk_rate:.0%}  fa={fa}  spike={spike}  gap={gap_mean:+.4f}")


def main():
    watch = "--watch" in sys.argv

    cases = load_jsonl(DISAGREEMENT_PATH)
    print(f"Loaded {len(cases)} disagreement cases")

    if not watch:
        # One-shot scan
        alerts = scan(cases)
        if alerts:
            for a in alerts:
                print(a)
            log_alerts(alerts)
        else:
            print_status(cases)
            print("OK — no alerts")
        return

    # Watch mode: poll for new entries
    print("Watching for new entries (Ctrl+C to stop)...")
    print_status(cases)

    last_n = len(cases)
    try:
        while True:
            time.sleep(5)
            cases = load_jsonl(DISAGREEMENT_PATH)
            if len(cases) > last_n:
                new_count = len(cases) - last_n
                last_n = len(cases)
                alerts = scan(cases)
                print_status(cases)
                for a in alerts:
                    print(a)
                log_alerts(alerts)
                if not alerts:
                    print(f"  +{new_count} new — OK")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
