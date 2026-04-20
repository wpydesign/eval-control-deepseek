#!/usr/bin/env python3
"""
reference_router.py — Reference router freeze + drift tracking [v2.6.0]

Fixes the manifold stability gap identified in v2.5.1.

Problem:
  π_live learns on contradiction-biased data (65% contradiction allocation).
  This creates a self-reinforcing bias loop:
    1. Model defines contradiction
    2. Acquisition selects more "contradiction-like" data
    3. Model sharpens only that region
    4. Manifold boundaries shift silently
  → router drift: manifolds are moving, not fixed

Solution:
  1. Freeze π_ref from v2.5.1 router → NEVER changes
  2. Dual-route every sample: m_live vs m_ref
  3. Track router_drift_rate = P(m_live ≠ m_ref) over rolling windows
  4. Hard guardrails:
       drift > 0.15 → freeze acquisition weights, log CRITICAL
       drift > 0.25 → fallback to balanced sampling (33/33/33), no debate
  5. Contradiction integrity: if contradiction channel selects sample,
     require m_ref == "contradiction" (stops leakage, redefinition)

The decomposition becomes a FIXED coordinate system, not a moving target.

Outputs:
    model/reference_router.pkl          - frozen π_ref (immutable)
    logs/router_drift_log.jsonl         - per-sample dual-route log
    logs/drift_guardrail_log.jsonl      - guardrail trigger events

Usage:
    python scripts/reference_router.py --freeze       # snapshot current router as π_ref
    python scripts/reference_router.py --status        # drift status report
    python scripts/reference_router.py --check-drift   # run drift assessment
"""

import json
import os
import sys
import pickle
import numpy as np
from datetime import datetime, timezone
from collections import deque

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFOLD_MODEL_PATH = os.path.join(BASE, "model", "manifold_models.pkl")
REFERENCE_ROUTER_PATH = os.path.join(BASE, "model", "reference_router.pkl")
DRIFT_LOG_PATH = os.path.join(BASE, "logs", "router_drift_log.jsonl")
GUARDRAIL_LOG_PATH = os.path.join(BASE, "logs", "drift_guardrail_log.jsonl")
DRIFT_STATE_PATH = os.path.join(BASE, "logs", "drift_state.json")

# --- Guardrail thresholds ---
DRIFT_WARNING = 0.15    # freeze acquisition weights
DRIFT_CRITICAL = 0.25   # fallback to balanced sampling
ROLLING_WINDOW = 100    # window for drift rate computation
MIN_DRIFT_SAMPLES = 20  # need this many before computing drift

# Balanced fallback weights (activated at CRITICAL)
BALANCED_WEIGHTS = {
    "contradiction": 0.333,
    "blind_spot": 0.333,
    "boundary": 0.334,
}

CLASS_NAMES = {0: "overconfidence", 1: "contradiction", 2: "boundary"}


class ReferenceRouter:
    """Manages the frozen reference router π_ref and drift tracking.

    π_ref is the v2.5.1 router snapshot. It NEVER changes.
    π_live continues learning via the normal retrain cycle.

    By comparing π_live routing decisions against π_ref, we detect
    when the manifold decomposition is drifting — meaning the live
    model is silently redefining what counts as "contradiction",
    "overconfidence", or "boundary".
    """

    def __init__(self):
        self._ref_router = None
        self._ref_frozen_at = None
        self._ref_loaded = False
        self._drift_buffer = deque(maxlen=ROLLING_WINDOW)
        self._load_reference()

    @property
    def is_loaded(self):
        return self._ref_loaded

    @property
    def frozen_at(self):
        return self._ref_frozen_at

    @property
    def buffer_size(self):
        return len(self._drift_buffer)

    def _load_reference(self):
        """Load frozen reference router from disk."""
        if not os.path.exists(REFERENCE_ROUTER_PATH):
            return False
        try:
            with open(REFERENCE_ROUTER_PATH, "rb") as f:
                package = pickle.load(f)
            self._ref_router = package.get("router")
            self._ref_frozen_at = package.get("frozen_at", "unknown")
            self._ref_loaded = True
            return True
        except Exception as e:
            print(f"  [REF_ROUTER] Failed to load: {e}")
            return False

    def freeze(self, router_model=None):
        """Snapshot the current router as π_ref.

        Args:
            router_model: sklearn multinomial LR router model.
                         If None, loads from manifold_models.pkl.
        """
        # Load from manifold models if no explicit model
        if router_model is None:
            if not os.path.exists(MANIFOLD_MODEL_PATH):
                print(f"  [REF_ROUTER] Cannot freeze — no manifold models at {MANIFOLD_MODEL_PATH}")
                return False
            with open(MANIFOLD_MODEL_PATH, "rb") as f:
                package = pickle.load(f)
            router_model = package.get("router_model")
            if router_model is None:
                print(f"  [REF_ROUTER] Cannot freeze — no router_model in package")
                return False

        # Freeze
        now = datetime.now(timezone.utc).isoformat()
        package = {
            "router": router_model,
            "frozen_at": now,
            "version": "v2.6.0",
            "description": "Reference router π_ref — frozen from v2.5.1 manifold decomposition. NEVER changes.",
        }

        os.makedirs(os.path.dirname(REFERENCE_ROUTER_PATH), exist_ok=True)
        with open(REFERENCE_ROUTER_PATH, "wb") as f:
            pickle.dump(package, f)

        self._ref_router = router_model
        self._ref_frozen_at = now
        self._ref_loaded = True

        print(f"  [REF_ROUTER] FROZEN π_ref at {now[:19]}")
        print(f"  [REF_ROUTER] Saved to {REFERENCE_ROUTER_PATH}")
        print(f"  [REF_ROUTER] This router will NEVER change — it is the fixed coordinate system")
        return True

    def dual_route(self, features, live_router=None, disagreement_flag=False):
        """Route sample through both live and reference routers.

        Args:
            features: [S_v4, S_v1, confidence_gap, kappa_v4]
            live_router: the current (evolving) router model
            disagreement_flag: explicit v4/v1 disagreement

        Returns:
            dict with m_live, m_ref, disagreement, routing info
        """
        X = np.array([features], dtype=np.float64)
        confidence_gap = features[2]
        man_result = {}

        # --- Reference route (π_ref) ---
        m_ref = None
        if self._ref_loaded and self._ref_router is not None:
            try:
                ref_probs = self._ref_router.predict_proba(X)[0]
                ref_class = int(self._ref_router.predict(X)[0])
                m_ref = CLASS_NAMES.get(ref_class, "boundary")
                man_result["m_ref"] = m_ref
                man_result["m_ref_confidence"] = float(max(ref_probs))
                man_result["m_ref_probabilities"] = {
                    CLASS_NAMES[i]: float(ref_probs[i]) for i in range(len(ref_probs))
                }
            except Exception:
                m_ref = "boundary"
                man_result["m_ref"] = m_ref
                man_result["m_ref_confidence"] = 0.0
                man_result["m_ref_error"] = "reference router failed"
        else:
            m_ref = "unknown"
            man_result["m_ref"] = m_ref
            man_result["m_ref_confidence"] = 0.0

        # --- Live route (π_live) ---
        m_live = None
        if live_router is not None:
            try:
                live_probs = live_router.predict_proba(X)[0]
                live_class = int(live_router.predict(X)[0])
                m_live = CLASS_NAMES.get(live_class, "boundary")
                man_result["m_live"] = m_live
                man_result["m_live_confidence"] = float(max(live_probs))
                man_result["m_live_probabilities"] = {
                    CLASS_NAMES[i]: float(live_probs[i]) for i in range(len(live_probs))
                }
            except Exception:
                m_live = "boundary"
                man_result["m_live"] = m_live
        else:
            # Rule-based routing for live (same as ManifoldPredictor._route heuristic)
            if disagreement_flag and confidence_gap > 0.05:
                m_live = "contradiction"
            elif confidence_gap > 0.25 and features[3] < 0.35:
                m_live = "overconfidence"
            elif confidence_gap > 0.10:
                m_live = "contradiction"
            else:
                m_live = "boundary"
            man_result["m_live"] = m_live
            man_result["m_live_confidence"] = 0.0
            man_result["m_live_method"] = "heuristic_fallback"

        # --- Drift signal ---
        disagreement = (m_live != m_ref)
        man_result["manifold_disagreement"] = disagreement

        # Record to buffer
        self._drift_buffer.append({
            "m_live": m_live,
            "m_ref": m_ref,
            "disagreement": disagreement,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return man_result

    def compute_drift_rate(self):
        """Compute router_drift_rate = P(m_live ≠ m_ref) over rolling window.

        Returns:
            dict with drift_rate, window_size, n_disagreements
        """
        if len(self._drift_buffer) < MIN_DRIFT_SAMPLES:
            return {
                "drift_rate": 0.0,
                "window_size": len(self._drift_buffer),
                "n_disagreements": 0,
                "status": "INSUFFICIENT_DATA",
                "interpretation": f"Need {MIN_DRIFT_SAMPLES} samples, have {len(self._drift_buffer)}",
            }

        n = len(self._drift_buffer)
        n_disc = sum(1 for entry in self._drift_buffer if entry["disagreement"])
        rate = n_disc / n

        if rate > DRIFT_CRITICAL:
            status = "CRITICAL"
            interpretation = "Manifold decomposition drifting severely — fallback to balanced sampling"
        elif rate > DRIFT_WARNING:
            status = "WARNING"
            interpretation = "Manifold boundaries shifting — freeze acquisition weights"
        else:
            status = "STABLE"
            interpretation = "Decomposition stable — normal operation"

        return {
            "drift_rate": round(rate, 4),
            "window_size": n,
            "n_disagreements": n_disc,
            "status": status,
            "interpretation": interpretation,
        }

    def check_guardrails(self, drift_rate):
        """Check drift guardrails and return action.

        Args:
            drift_rate: current P(m_live ≠ m_ref)

        Returns:
            dict with action, reason, new_weights (if applicable)
        """
        if drift_rate >= DRIFT_CRITICAL:
            action = {
                "level": "CRITICAL",
                "action": "FALLBACK_BALANCED",
                "new_weights": dict(BALANCED_WEIGHTS),
                "reason": f"router_drift_rate={drift_rate:.4f} >= {DRIFT_CRITICAL} — falling back to 33/33/33",
                "recommendation": "Investigate retrain cycle for contradiction overfit. Do NOT continue normal operation.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._log_guardrail(action)
            return action
        elif drift_rate >= DRIFT_WARNING:
            action = {
                "level": "WARNING",
                "action": "FREEZE_WEIGHTS",
                "new_weights": None,  # keep current weights
                "reason": f"router_drift_rate={drift_rate:.4f} >= {DRIFT_WARNING} — freezing acquisition weights",
                "recommendation": "Do NOT adapt weights this cycle. Investigate manifold boundary shift.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._log_guardrail(action)
            return action
        else:
            return {
                "level": "OK",
                "action": "NONE",
                "new_weights": None,
                "reason": f"router_drift_rate={drift_rate:.4f} < {DRIFT_WARNING} — stable",
            }

    def validate_contradiction_integrity(self, features, live_router=None,
                                          disagreement_flag=False):
        """Check if a sample is truly contradiction according to π_ref.

        Protection against self-reinforcing bias:
          If π_live routes to "contradiction" but π_ref disagrees,
          that sample should NOT be selected by the contradiction channel.

        Returns:
            dict with valid (bool), m_ref, m_live, reason
        """
        dual = self.dual_route(features, live_router, disagreement_flag)
        m_live = dual.get("m_live", "unknown")
        m_ref = dual.get("m_ref", "unknown")

        if m_ref == "contradiction":
            return {
                "valid": True,
                "m_ref": m_ref,
                "m_live": m_live,
                "reason": "π_ref confirms contradiction — safe to select",
            }
        elif m_ref == "unknown":
            # No reference router loaded — allow but warn
            return {
                "valid": True,
                "m_ref": m_ref,
                "m_live": m_live,
                "reason": "WARNING: no π_ref loaded — integrity check bypassed",
            }
        else:
            # π_ref disagrees — this is leakage from another manifold
            return {
                "valid": False,
                "m_ref": m_ref,
                "m_live": m_live,
                "reason": f"π_ref={m_ref}, π_live={m_live} — REJECTED: contradiction integrity violation",
            }

    def load_drift_state(self):
        """Load persistent drift state from disk (survives restarts)."""
        if os.path.exists(DRIFT_STATE_PATH):
            try:
                with open(DRIFT_STATE_PATH) as f:
                    state = json.load(f)
                # Restore buffer from log
                if os.path.exists(DRIFT_LOG_PATH):
                    with open(DRIFT_LOG_PATH) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                entry = json.loads(line)
                                self._drift_buffer.append({
                                    "m_live": entry.get("m_live"),
                                    "m_ref": entry.get("m_ref"),
                                    "disagreement": entry.get("disagreement", False),
                                    "timestamp": entry.get("timestamp", ""),
                                })
                return state
            except Exception:
                pass
        return {}

    def save_drift_state(self, drift_info):
        """Save drift state to disk."""
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v2.6.0",
            "frozen_at": self._ref_frozen_at,
            "drift": drift_info,
            "buffer_size": len(self._drift_buffer),
        }
        os.makedirs(os.path.dirname(DRIFT_STATE_PATH), exist_ok=True)
        with open(DRIFT_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _log_guardrail(self, action):
        """Log guardrail trigger to persistent log."""
        os.makedirs(os.path.dirname(GUARDRAIL_LOG_PATH), exist_ok=True)
        with open(GUARDRAIL_LOG_PATH, "a") as f:
            f.write(json.dumps(action, ensure_ascii=False) + "\n")

    def log_dual_route(self, query_id, dual_result, features):
        """Log a dual-route result for audit trail."""
        os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
        entry = {
            "query_id": query_id,
            "m_live": dual_result.get("m_live"),
            "m_ref": dual_result.get("m_ref"),
            "manifold_disagreement": dual_result.get("manifold_disagreement", False),
            "features": [round(float(x), 4) for x in features],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(DRIFT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def print_status(self):
        """Print reference router status and drift report."""
        print("=" * 65)
        print("  REFERENCE ROUTER π_ref STATUS [v2.6.0]")
        print("=" * 65)

        if not self._ref_loaded:
            print("  Status: NOT FROZEN — run with --freeze to create π_ref")
            print("  Without π_ref, drift tracking is disabled")
            print("  This is the FIXED coordinate system for manifold decomposition")
            return

        print(f"  Frozen:      {self._ref_frozen_at[:19] if self._ref_frozen_at else 'unknown'}")
        print(f"  Drift buffer: {len(self._drift_buffer)}/{ROLLING_WINDOW} samples")

        drift = self.compute_drift_rate()
        status_icon = {
            "STABLE": "OK",
            "WARNING": "!!",
            "CRITICAL": "!!!",
            "INSUFFICIENT_DATA": "...",
        }.get(drift["status"], "?")

        print(f"\n  Router drift: {status_icon}")
        print(f"    drift_rate:  {drift['drift_rate']:.4f}  ({drift['n_disagreements']}/{drift['window_size']})")
        print(f"    status:      {drift['status']}")
        print(f"    meaning:     {drift['interpretation']}")

        print(f"\n  Guardrails:")
        print(f"    WARNING:     drift > {DRIFT_WARNING} → freeze acquisition weights")
        print(f"    CRITICAL:    drift > {DRIFT_CRITICAL} → fallback to 33/33/33")

        # Show per-manifold disagreement breakdown
        if len(self._drift_buffer) >= MIN_DRIFT_SAMPLES:
            disc_entries = [e for e in self._drift_buffer if e["disagreement"]]
            if disc_entries:
                print(f"\n  Disagreement patterns (last {len(disc_entries)}):")
                patterns = {}
                for e in disc_entries:
                    key = f"{e['m_ref']} -> {e['m_live']}"
                    patterns[key] = patterns.get(key, 0) + 1
                for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
                    print(f"    {pattern:30s}: {count}")

        print(f"\n  π_ref = fixed coordinate system. π_live = evolving learner.")
        print(f"  If drift rises, the manifolds are moving. That is danger.")


def main():
    """CLI: manage reference router."""
    do_freeze = "--freeze" in sys.argv
    do_status = "--status" in sys.argv or len(sys.argv) == 1
    do_check = "--check-drift" in sys.argv

    ref = ReferenceRouter()
    ref.load_drift_state()

    if do_freeze:
        print("  Freezing reference router π_ref from current manifold models...")
        ref.freeze()
        print()

    if do_status:
        ref.print_status()

    if do_check:
        print("  Checking drift guardrails...")
        drift = ref.compute_drift_rate()
        action = ref.check_guardrails(drift["drift_rate"])
        ref.save_drift_state(drift)

        print(f"\n  Drift rate: {drift['drift_rate']:.4f} (status: {drift['status']})")
        print(f"  Guardrail:  {action['level']} → {action['action']}")
        if action.get("new_weights"):
            print(f"  New weights: {action['new_weights']}")
        print(f"  Reason: {action['reason']}")


if __name__ == "__main__":
    main()
