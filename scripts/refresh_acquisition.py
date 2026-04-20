#!/usr/bin/env python3
"""
refresh_acquisition.py — Manifold-level acquisition refresh cycle [v2.6.0]

v2.6.0: Added drift check before retrain + guardrail integration.

Cycle:
    0. Check router drift guardrails (v2.6.0: prevent biased retraining)
    1. Retrain contradiction head ONLY (not unified model)
    2. Adapt manifold weights (contradiction-primary KPI)
    3. Refresh queues (uncertainty + blind-spot)
    4. Rebuild acquisition with manifold-aware allocation (+ guardrails)
    5. Report manifold KPIs (not global AUC)

Key change: retraining now targets the contradiction manifold specifically,
not the unified predictor. The contradiction head is the only learnable surface.

v2.6.0: If router_drift_rate > 0.25, retrain is SKIPPED (manifold drift danger).
       If router_drift_rate > 0.15, weight adaptation is SKIPPED (freeze).

Usage:
    python scripts/refresh_acquisition.py                # full manifold cycle
    python scripts/refresh_acquisition.py --skip-retrain  # only adapt + refresh
    python scripts/refresh_acquisition.py --budget 50     # custom budget
    python scripts/refresh_acquisition.py --freeze-ref     # freeze π_ref before cycle
"""

import json
import os
import sys

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(DIR, "scripts")
sys.path.insert(0, SCRIPTS)


def run_step(name, module, func_name, args=None):
    """Run a function from a module, return success."""
    try:
        mod = __import__(module, fromlist=[func_name])
        fn = getattr(mod, func_name)
        if args:
            fn(*args)
        else:
            fn()
        return True
    except Exception as e:
        print(f"  [{name}] FAILED: {e}")
        return False


def main():
    skip_retrain = "--skip-retrain" in sys.argv
    budget = 50
    if "--budget" in sys.argv:
        idx = sys.argv.index("--budget")
        if idx + 1 < len(sys.argv):
            budget = int(sys.argv[idx + 1])
    do_freeze_ref = "--freeze-ref" in sys.argv

    print("=" * 65)
    print("  ACQUISITION FLYWHEEL — FULL REFRESH CYCLE [v2.6.0]")
    print("=" * 65)

    # Step 0: Optionally freeze π_ref (v2.6.0: one-time setup)
    if do_freeze_ref:
        print("\n--- Step 0a: Freeze reference router π_ref ---")
        try:
            from reference_router import ReferenceRouter
            ref = ReferenceRouter()
            ref.freeze()
        except Exception as e:
            print(f"  Failed to freeze π_ref: {e}")

    # Step 0b: Check router drift guardrails (v2.6.0)
    print("\n--- Step 0b: Check router drift guardrails ---")
    drift_guardrail = "NONE"
    drift_rate = 0.0
    try:
        from reference_router import ReferenceRouter, DRIFT_WARNING, DRIFT_CRITICAL
        ref = ReferenceRouter()
        ref.load_drift_state()
        drift_info = ref.compute_drift_rate()
        drift_rate = drift_info.get("drift_rate", 0)
        guardrail = ref.check_guardrails(drift_rate)
        drift_guardrail = guardrail["action"]

        if drift_info["status"] != "INSUFFICIENT_DATA":
            print(f"  router_drift_rate: {drift_rate:.4f} (status: {drift_info['status']})")
            print(f"  Guardrail: {drift_guardrail}")
        else:
            print(f"  {drift_info['interpretation']}")
            drift_guardrail = "NONE"  # can't guardrail without data

        ref.save_drift_state(drift_info)
    except Exception as e:
        print(f"  Drift check failed: {e}")
        print(f"  Continuing without drift protection (run --freeze-ref first)")

    # Step 1: Retrain contradiction head (v2.6.0: skip if drift CRITICAL)
    if not skip_retrain:
        if drift_guardrail == "FALLBACK_BALANCED":
            print("\n--- Step 1: Retrain SKIPPED (drift CRITICAL) ---")
            print("  router_drift_rate too high — retraining would worsen manifold drift")
            print("  Investigate contradiction overfit before continuing")
        else:
            print("\n--- Step 1: Retrain contradiction head (manifold-level) ---")
            try:
                sys.path.insert(0, SCRIPTS)
                from manifold_classifier import main as classify_main
                old_argv = sys.argv
                sys.argv = ["manifold_classifier.py", "--retrain"]
                classify_main()
                sys.argv = old_argv
                print("  Contradiction head retrained")
            except Exception as e:
                print(f"  Retrain failed: {e}")
    else:
        print("\n--- Step 1: Retrain SKIPPED (user flag) ---")

    # Step 2: Adapt manifold weights (v2.6.0: skip if drift WARNING or CRITICAL)
    print("\n--- Step 2: Adapt manifold weights ---")
    if drift_guardrail in ("FALLBACK_BALANCED", "FREEZE_WEIGHTS"):
        print(f"  Weight adaptation SKIPPED (drift guardrail: {drift_guardrail})")
        print(f"  Using guardrail-adjusted weights instead")
    else:
        from acquisition_policy import update_weights_cli
        weights = update_weights_cli()

    # Step 3: Refresh uncertainty queue
    print("\n--- Step 3: Refresh uncertainty queue ---")
    ok_unc = run_step("uncertainty", "active_learning", "main")
    if ok_unc:
        print("  Uncertainty queue refreshed")

    # Step 4: Refresh blind-spot queue
    print("\n--- Step 4: Refresh blind-spot queue ---")
    ok_bs = run_step("blind_spot", "failure_mining", "main")
    if ok_bs:
        print("  Blind-spot queue refreshed")

    # Step 5: Rebuild acquisition policy (manifold-aware)
    print("\n--- Step 5: Rebuild manifold-aware acquisition ---")
    sys.argv = ["acquisition_policy.py", "--budget", str(budget), "--show", "15"]
    ok_acq = run_step("acquisition", "acquisition_policy", "main")
    sys.argv = [sys.argv[0]]

    # Step 6: Report manifold KPIs
    print("\n--- Step 6: Manifold KPI check ---")
    try:
        from manifold_kpi import compute_manifold_kpis, print_kpis
        kpis = compute_manifold_kpis()
        print_kpis(kpis)
    except Exception as e:
        print(f"  KPI check failed: {e}")

    # Summary
    print(f"\n{'='*65}")
    print("  MANIFOLD-LEVEL FLYWHEEL CYCLE COMPLETE [v2.6.0]")
    print(f"{'='*65}")
    print(f"  Drift check:        {drift_guardrail} (rate={drift_rate:.4f})")
    print(f"  Retrain:           {'skipped (drift CRITICAL)' if drift_guardrail == 'FALLBACK_BALANCED' and not skip_retrain else ('skipped' if skip_retrain else 'contradiction head only')}")
    print(f"  Weight adaptation:  {'skipped (drift guardrail)' if drift_guardrail in ('FREEZE_WEIGHTS', 'FALLBACK_BALANCED') else 'manifold-aware (cd-primary)'}")
    print(f"  Uncertainty queue:  {'refreshed' if ok_unc else 'failed'}")
    print(f"  Blind-spot queue:   {'refreshed' if ok_bs else 'failed'}")
    print(f"  Acquisition policy: {'rebuilt (manifold-aware + guardrails)' if ok_acq else 'failed'}")
    print(f"\n  Loop: label contradiction -> retrain cd head -> check drift -> repeat")
    if drift_guardrail != "NONE":
        print(f"\n  !! DRIFT GUARDRAIL ACTIVE: {drift_guardrail} — investigation recommended")


if __name__ == "__main__":
    main()
