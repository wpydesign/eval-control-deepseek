#!/usr/bin/env python3
"""
reference_router.py — Reference router freeze + drift tracking [v2.6.1]

Fixes the manifold stability gap identified in v2.5.1.
Fixes the reference staleness gap identified in v2.6.0.

Problem 1 (v2.6.0 — solved):
  π_live learns on contradiction-biased data → self-reinforcing bias loop
  → router drift: manifolds are moving, not fixed

Problem 2 (v2.6.1 — solved here):
  π_ref was frozen at v2.5.1 but system keeps learning on top of it.
  Over time π_ref stops being "ground truth" and becomes "historical bias".
  → reference staleness: anchoring to an outdated map

Solution:
  1. Freeze π_ref from current router → stable anchor (with controlled refresh)
  2. Dual-route every sample: m_live vs m_ref
  3. Track router_drift_rate = P(m_live ≠ m_ref) over rolling windows
  4. Shadow evaluate: ref_accuracy vs live_accuracy against true labels
  5. Track ref_decay = ref_accuracy - live_accuracy
  6. Controlled refresh: if ref_decay < -0.10 AND samples > 200:
     - snapshot current π_live → new π_ref
     - reset drift history
     - enforce cooldown (REF_COOLDOWN = 3 cycles)
  7. Hard guardrails:
       drift > 0.15 → freeze acquisition weights
       drift > 0.25 → fallback to balanced sampling (33/33/33)
  8. Contradiction integrity: require m_ref == "contradiction"

Full control loop:
  π_live learns
  π_ref anchors
  drift monitors geometry
  ref_decay monitors truth
  guardrails stabilize
  refresh updates anchor (rarely, with cooldown)

Outputs:
    model/reference_router.pkl          - frozen π_ref (immutable until controlled refresh)
    logs/router_drift_log.jsonl         - per-sample dual-route log
    logs/drift_guardrail_log.jsonl      - guardrail trigger events
    logs/ref_refresh_log.jsonl          - reference refresh events

Usage:
    python scripts/reference_router.py --freeze       # snapshot current router as π_ref
    python scripts/reference_router.py --status        # drift + staleness status
    python scripts/reference_router.py --check-drift   # run drift assessment
    python scripts/reference_router.py --eval-accuracy  # compute ref vs live accuracy
    python scripts/reference_router.py --check-refresh  # check if refresh needed
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
REF_REFRESH_LOG_PATH = os.path.join(BASE, "logs", "ref_refresh_log.jsonl")
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
BATCH_LABELS_PATH = os.path.join(BASE, "logs", "batch_label_results.jsonl")

# --- Guardrail thresholds ---
DRIFT_WARNING = 0.15    # freeze acquisition weights
DRIFT_CRITICAL = 0.25   # fallback to balanced sampling
ROLLING_WINDOW = 100    # window for drift rate computation
MIN_DRIFT_SAMPLES = 20  # need this many before computing drift

# --- Reference staleness thresholds (v2.6.1) ---
REF_DECAY_THRESHOLD = -0.10   # if ref_accuracy - live_accuracy drops below this, refresh
REF_MIN_EVAL_SAMPLES = 200     # minimum labeled samples before considering refresh
REF_COOLDOWN = 3                # cycles between refreshes (prevents oscillation)

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
        self._refresh_history = {
            "last_refresh": None,
            "total_refreshes": 0,
            "cycles_since_refresh": 0,
        }
        self._load_reference()
        self._load_refresh_history()

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

    def _load_refresh_history(self):
        """Load refresh history from drift state."""
        if os.path.exists(DRIFT_STATE_PATH):
            try:
                with open(DRIFT_STATE_PATH) as f:
                    state = json.load(f)
                refresh = state.get("refresh_history", {})
                if refresh:
                    self._refresh_history = refresh
            except Exception:
                pass

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
                self._load_refresh_history()
                return state
            except Exception:
                pass
        return {}

    def save_drift_state(self, drift_info, ref_accuracy_info=None):
        """Save drift state to disk."""
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v2.6.1",
            "frozen_at": self._ref_frozen_at,
            "drift": drift_info,
            "refresh_history": self._refresh_history,
            "buffer_size": len(self._drift_buffer),
        }
        if ref_accuracy_info:
            state["ref_accuracy"] = ref_accuracy_info
        os.makedirs(os.path.dirname(DRIFT_STATE_PATH), exist_ok=True)
        with open(DRIFT_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    # --- v2.6.1: Reference staleness detection ---

    def compute_ref_accuracy(self):
        """Shadow evaluate π_ref vs π_live against true manifold labels.

        For every labeled sample, compare which router got the manifold right:
          ref_correct  = (m_ref  == true_manifold)
          live_correct = (m_live == true_manifold)

        Returns:
            dict with ref_accuracy, live_accuracy, ref_decay, n_evaluated
        """
        if not self._ref_loaded:
            return {
                "ref_accuracy": 0.0, "live_accuracy": 0.0, "ref_decay": 0.0,
                "n_evaluated": 0, "status": "NO_REF_ROUTER",
                "interpretation": "No π_ref loaded — cannot evaluate staleness",
            }

        # Load labeled samples with manifold annotations
        labeled = []

        # From batch_label_results.jsonl (has failure_type = manifold)
        if os.path.exists(BATCH_LABELS_PATH):
            with open(BATCH_LABELS_PATH) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        ft = r.get("failure_type", "boundary")
                        if r.get("source_channel") == "blind_spot":
                            ft = "overconfidence"
                        s_v4 = r.get("v4_scores", {}).get("S", r.get("score_v4", 0.5))
                        s_v1 = r.get("v1_scores", {}).get("S", r.get("score_v1", 0.5))
                        kappa = r.get("v4_scores", {}).get("kappa", r.get("kappa_v4", 0.3))
                        gap = abs(s_v4 - s_v1)
                        labeled.append({
                            "features": [s_v4, s_v1, gap, kappa],
                            "true_manifold": ft,
                        })

        # From failure_dataset.jsonl (manifold assigned via batch_ft)
        if os.path.exists(DATASET_PATH):
            batch_ft_map = {}
            if os.path.exists(BATCH_LABELS_PATH):
                with open(BATCH_LABELS_PATH) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            r = json.loads(line)
                            batch_ft_map[r.get("query_id", "")] = r.get("failure_type", "boundary")

            with open(DATASET_PATH) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        if r.get("is_wrong") is None:
                            continue
                        qid = r.get("query_id", "")
                        ft = batch_ft_map.get(qid, "boundary")
                        if ft == "overconfidence":
                            true_m = "overconfidence"
                        elif ft == "contradiction":
                            true_m = "contradiction"
                        else:
                            true_m = "boundary"
                        s_v4 = r.get("score_v4", r.get("S_v4", 0.5))
                        s_v1 = r.get("score_v1", r.get("S_v1", 0.5))
                        kappa = r.get("kappa_v4", 0.3)
                        gap = abs(s_v4 - s_v1)
                        labeled.append({
                            "features": [s_v4, s_v1, gap, kappa],
                            "true_manifold": true_m,
                        })

        if not labeled:
            return {
                "ref_accuracy": 0.0, "live_accuracy": 0.0, "ref_decay": 0.0,
                "n_evaluated": 0, "status": "NO_LABELED_DATA",
                "interpretation": "No labeled samples with manifold annotations found",
            }

        # Load π_live router
        live_router = None
        if os.path.exists(MANIFOLD_MODEL_PATH):
            try:
                with open(MANIFOLD_MODEL_PATH, "rb") as f:
                    pkg = pickle.load(f)
                live_router = pkg.get("router_model")
            except Exception:
                pass

        # Evaluate both routers
        n_ref_correct = 0
        n_live_correct = 0
        n_both_correct = 0
        n_eval = len(labeled)

        # Per-manifold accuracy breakdown
        ref_per_m = {"overconfidence": {"correct": 0, "total": 0},
                     "contradiction": {"correct": 0, "total": 0},
                     "boundary": {"correct": 0, "total": 0}}
        live_per_m = {"overconfidence": {"correct": 0, "total": 0},
                     "contradiction": {"correct": 0, "total": 0},
                     "boundary": {"correct": 0, "total": 0}}

        for sample in labeled:
            features = sample["features"]
            true_m = sample["true_manifold"]
            X = np.array([features], dtype=np.float64)

            # π_ref routing
            m_ref = "unknown"
            if self._ref_router is not None:
                try:
                    ref_class = int(self._ref_router.predict(X)[0])
                    m_ref = CLASS_NAMES.get(ref_class, "boundary")
                except Exception:
                    pass

            # π_live routing
            m_live = "boundary"  # default
            if live_router is not None:
                try:
                    live_class = int(live_router.predict(X)[0])
                    m_live = CLASS_NAMES.get(live_class, "boundary")
                except Exception:
                    pass

            ref_hit = (m_ref == true_m)
            live_hit = (m_live == true_m)

            if ref_hit:
                n_ref_correct += 1
            if live_hit:
                n_live_correct += 1
            if ref_hit and live_hit:
                n_both_correct += 1

            # Per-manifold
            for router_name, hit, per_m in [("ref", ref_hit, ref_per_m), ("live", live_hit, live_per_m)]:
                if true_m in per_m:
                    per_m[true_m]["total"] += 1
                    if hit:
                        per_m[true_m]["correct"] += 1

        ref_acc = n_ref_correct / n_eval
        live_acc = n_live_correct / n_eval
        ref_decay = ref_acc - live_acc

        # Interpretation
        if ref_decay > 0.05:
            status = "LIVE_DRIFTING_WRONG"
            interpretation = "π_live is drifting wrong — π_ref is still the better anchor"
        elif ref_decay < REF_DECAY_THRESHOLD:
            status = "REF_STALE"
            interpretation = f"π_ref is decaying (ref_decay={ref_decay:.3f} < {REF_DECAY_THRESHOLD}) — consider controlled refresh"
        elif ref_decay < -0.05:
            status = "REF_AGING"
            interpretation = "π_live is becoming more accurate — monitor for staleness"
        else:
            status = "VALID"
            interpretation = "Reference is still valid — no refresh needed"

        return {
            "ref_accuracy": round(ref_acc, 4),
            "live_accuracy": round(live_acc, 4),
            "ref_decay": round(ref_decay, 4),
            "n_evaluated": n_eval,
            "n_both_correct": n_both_correct,
            "status": status,
            "interpretation": interpretation,
            "ref_per_manifold": ref_per_m,
            "live_per_manifold": live_per_m,
        }

    def check_ref_refresh(self, ref_accuracy_info):
        """Check if a controlled reference refresh is warranted.

        Conditions (ALL must be true):
          1. ref_decay < REF_DECAY_THRESHOLD (-0.10)
          2. n_evaluated >= REF_MIN_EVAL_SAMPLES (200)
          3. cycles_since_refresh >= REF_COOLDOWN (3)

        Returns:
            dict with should_refresh (bool), reason, conditions
        """
        ref_decay = ref_accuracy_info.get("ref_decay", 0)
        n_eval = ref_accuracy_info.get("n_evaluated", 0)
        cycles = self._refresh_history.get("cycles_since_refresh", 999)
        last_refresh = self._refresh_history.get("last_refresh")

        conditions = {
            "ref_decay_below_threshold": ref_decay < REF_DECAY_THRESHOLD,
            "sufficient_samples": n_eval >= REF_MIN_EVAL_SAMPLES,
            "cooldown_passed": cycles >= REF_COOLDOWN,
        }

        all_met = all(conditions.values())

        if all_met:
            return {
                "should_refresh": True,
                "reason": (f"ref_decay={ref_decay:.4f} < {REF_DECAY_THRESHOLD}, "
                          f"n={n_eval} >= {REF_MIN_EVAL_SAMPLES}, "
                          f"cooldown={cycles} >= {REF_COOLDOWN}"),
                "conditions": conditions,
            }
        else:
            return {
                "should_refresh": False,
                "reason": "Conditions not met",
                "conditions": conditions,
            }

    def controlled_refresh(self, router_model=None):
        """Perform a controlled reference refresh.

        Snapshots the current π_live as the new π_ref.
        Resets drift history. Updates cooldown counter.
        This is a DISCRETE JUMP — no continuous updating.

        After refresh, all integrity checks automatically move with π_ref.
        No extra logic needed.

        Returns:
            bool — success
        """
        # Freeze new reference
        success = self.freeze(router_model)
        if not success:
            return False

        # Reset drift buffer and history
        self._drift_buffer.clear()

        # Clear drift log
        if os.path.exists(DRIFT_LOG_PATH):
            os.remove(DRIFT_LOG_PATH)

        # Update refresh history
        now = datetime.now(timezone.utc).isoformat()
        self._refresh_history = {
            "last_refresh": now,
            "total_refreshes": self._refresh_history.get("total_refreshes", 0) + 1,
            "cycles_since_refresh": 0,
            "previous_frozen_at": self._refresh_history.get("last_refresh"),
        }

        # Log refresh event
        os.makedirs(os.path.dirname(REF_REFRESH_LOG_PATH), exist_ok=True)
        event = {
            "timestamp": now,
            "action": "reference_refresh",
            "new_frozen_at": now,
            "total_refreshes": self._refresh_history["total_refreshes"],
            "cooldown_cycles": REF_COOLDOWN,
            "reason": "controlled refresh — ref_decay exceeded threshold",
        }
        with open(REF_REFRESH_LOG_PATH, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        print(f"  [REF_ROUTER] Controlled refresh complete")
        print(f"  [REF_ROUTER] New π_ref frozen at {now[:19]}")
        print(f"  [REF_ROUTER] Drift history RESET")
        print(f"  [REF_ROUTER] Cooldown: {REF_COOLDOWN} cycles until next refresh allowed")
        print(f"  [REF_ROUTER] Total refreshes: {self._refresh_history['total_refreshes']}")

        return True

    def increment_cycle(self):
        """Increment the cycle counter for cooldown tracking."""
        self._refresh_history["cycles_since_refresh"] = \
            self._refresh_history.get("cycles_since_refresh", 0) + 1

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
        print("  REFERENCE ROUTER π_ref STATUS [v2.6.1]")
        print("=" * 65)

        if not self._ref_loaded:
            print("  Status: NOT FROZEN — run with --freeze to create π_ref")
            print("  Without π_ref, drift tracking is disabled")
            print("  This is the FIXED coordinate system for manifold decomposition")
            return

        print(f"  Frozen:      {self._ref_frozen_at[:19] if self._ref_frozen_at else 'unknown'}")
        print(f"  Drift buffer: {len(self._drift_buffer)}/{ROLLING_WINDOW} samples")

        rh = self._refresh_history
        print(f"  Refreshes:   {rh.get('total_refreshes', 0)} total")
        print(f"  Cooldown:    {rh.get('cycles_since_refresh', 0)}/{REF_COOLDOWN} cycles since last refresh")
        if rh.get("last_refresh"):
            print(f"  Last refresh: {rh['last_refresh'][:19]}")

        # --- Router drift ---
        drift = self.compute_drift_rate()
        drift_icon = {
            "STABLE": "OK",
            "WARNING": "!!",
            "CRITICAL": "!!!",
            "INSUFFICIENT_DATA": "...",
        }.get(drift["status"], "?")

        print(f"\n  Router drift (geometry): {drift_icon}")
        print(f"    drift_rate:  {drift['drift_rate']:.4f}  ({drift['n_disagreements']}/{drift['window_size']})")
        print(f"    status:      {drift['status']}")
        print(f"    meaning:     {drift['interpretation']}")

        # --- Reference staleness (v2.6.1) ---
        ref_acc = self.compute_ref_accuracy()
        ref_decay = ref_acc.get("ref_decay", 0)
        staleness_icon = {
            "VALID": "OK",
            "REF_AGING": "~",
            "REF_STALE": "!!",
            "LIVE_DRIFTING_WRONG": "<<",
            "NO_LABELED_DATA": "...",
            "NO_REF_ROUTER": "...",
        }.get(ref_acc["status"], "?")

        print(f"\n  Reference staleness (truth): {staleness_icon}")
        print(f"    ref_accuracy:  {ref_acc['ref_accuracy']:.4f}")
        print(f"    live_accuracy: {ref_acc['live_accuracy']:.4f}")
        print(f"    ref_decay:     {ref_decay:+.4f}  (ref - live)")
        print(f"    n_evaluated:   {ref_acc['n_evaluated']}")
        print(f"    status:        {ref_acc['status']}")
        print(f"    meaning:       {ref_acc['interpretation']}")

        # Per-manifold accuracy breakdown
        if ref_acc.get("n_evaluated", 0) > 0:
            ref_pm = ref_acc.get("ref_per_manifold", {})
            live_pm = ref_acc.get("live_per_manifold", {})
            print(f"\n  Per-manifold routing accuracy:")
            print(f"    {'MANIFOLD':18s} {'REF':>6s} {'LIVE':>6s} {'DELTA':>7s}")
            for m in ["overconfidence", "contradiction", "boundary"]:
                r_total = ref_pm.get(m, {}).get("total", 0)
                l_total = live_pm.get(m, {}).get("total", 0)
                r_acc = ref_pm.get(m, {}).get("correct", 0) / max(r_total, 1)
                l_acc = live_pm.get(m, {}).get("correct", 0) / max(l_total, 1)
                delta = r_acc - l_acc
                print(f"    {m:18s} {r_acc:>6.1%} {l_acc:>6.1%} {delta:>+7.1%}")

        print(f"\n  Guardrails:")
        print(f"    WARNING:     drift > {DRIFT_WARNING} → freeze acquisition weights")
        print(f"    CRITICAL:    drift > {DRIFT_CRITICAL} → fallback to 33/33/33")
        print(f"    REFRESH:     ref_decay < {REF_DECAY_THRESHOLD} + n>{REF_MIN_EVAL_SAMPLES} + cooldown>{REF_COOLDOWN}")

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

        print(f"\n  π_ref = anchor. π_live = learner. drift = geometry. ref_decay = truth.")
        print(f"  Full loop: learn -> anchor -> monitor -> refresh (rarely, with cooldown).")


def main():
    """CLI: manage reference router."""
    do_freeze = "--freeze" in sys.argv
    do_status = "--status" in sys.argv or len(sys.argv) == 1
    do_check = "--check-drift" in sys.argv
    do_eval = "--eval-accuracy" in sys.argv
    do_check_refresh = "--check-refresh" in sys.argv
    do_refresh = "--refresh" in sys.argv

    ref = ReferenceRouter()
    ref.load_drift_state()

    if do_freeze:
        print("  Freezing reference router π_ref from current manifold models...")
        ref.freeze()
        print()

    if do_eval or do_check_refresh:
        print("  Evaluating ref vs live accuracy against labeled data...")
        ref_acc = ref.compute_ref_accuracy()
        print(f"  ref_accuracy:  {ref_acc['ref_accuracy']:.4f}")
        print(f"  live_accuracy: {ref_acc['live_accuracy']:.4f}")
        print(f"  ref_decay:     {ref_acc['ref_decay']:+.4f}")
        print(f"  n_evaluated:   {ref_acc['n_evaluated']}")
        print(f"  status:        {ref_acc['status']}")
        print(f"  meaning:       {ref_acc['interpretation']}")
        ref.save_drift_state(ref.compute_drift_rate(), ref_acc)

    if do_check_refresh:
        print("\n  Checking if controlled refresh is warranted...")
        check = ref.check_ref_refresh(ref_acc)
        print(f"  should_refresh: {check['should_refresh']}")
        print(f"  reason: {check['reason']}")
        for cond, met in check["conditions"].items():
            icon = "MET" if met else "NOT_MET"
            print(f"    {cond}: {icon}")
        if check["should_refresh"]:
            print(f"\n  Run with --refresh to perform controlled reference refresh")

    if do_refresh:
        print("  Performing controlled reference refresh...")
        ref_acc = ref.compute_ref_accuracy()
        check = ref.check_ref_refresh(ref_acc)
        if check["should_refresh"]:
            ref.controlled_refresh()
            drift = ref.compute_drift_rate()
            ref.save_drift_state(drift, ref_acc)
        else:
            print(f"  Refresh BLOCKED: {check['reason']}")
            for cond, met in check["conditions"].items():
                icon = "MET" if met else "BLOCKED"
                print(f"    {cond}: {icon}")

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
