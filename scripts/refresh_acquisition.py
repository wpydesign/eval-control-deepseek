#!/usr/bin/env python3
"""
refresh_acquisition.py — Full acquisition cycle: retrain → adapt → refresh [v2.3.0]

This is the single entry point that closes the data flywheel loop:

    1. Retrain failure predictor with new labels
    2. Adapt channel weights based on per-channel efficiency
    3. Refresh uncertainty queue + blind-spot queue
    4. Rebuild acquisition policy with new weights
    5. Report new AUC, weight changes, and next-priority samples

Usage:
    python scripts/refresh_acquisition.py                # full cycle
    python scripts/refresh_acquisition.py --skip-retrain  # only adapt + refresh
    python scripts/refresh_acquisition.py --budget 50     # custom budget
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

    print("=" * 65)
    print("  ACQUISITION FLYWHEEL — FULL REFRESH CYCLE")
    print("=" * 65)

    # Step 1: Retrain (if not skipped)
    if not skip_retrain:
        print("\n--- Step 1: Retrain failure predictor ---")
        from predict_failure import FailurePredictor
        predictor = FailurePredictor()
        if predictor.is_loaded:
            # Record pre-retrain AUC
            pre_auc = predictor.metadata.get("metrics", {}).get("auc", "N/A")
            print(f"  Pre-retrain AUC: {pre_auc}")

            print("  Retraining...")
            ok = predictor.retrain()
            if ok:
                post_auc = predictor.metadata.get("metrics", {}).get("auc", "N/A")
                print(f"  Post-retrain AUC: {post_auc}")
                delta = "IMPROVED" if (isinstance(pre_auc, (int, float)) and
                                       isinstance(post_auc, (int, float)) and
                                       post_auc > pre_auc) else "stable"
                print(f"  Status: {delta}")
            else:
                print("  Retrain failed — continuing with existing model")
        else:
            print("  No model loaded — skipping retrain")
    else:
        print("\n--- Step 1: Retrain SKIPPED ---")

    # Step 2: Adapt weights
    print("\n--- Step 2: Adapt channel weights ---")
    from acquisition_policy import update_weights_cli, load_channel_performance
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

    # Step 5: Rebuild acquisition policy
    print("\n--- Step 5: Rebuild acquisition policy ---")
    sys.argv = ["acquisition_policy.py", "--budget", str(budget), "--show", "15"]
    ok_acq = run_step("acquisition", "acquisition_policy", "main")
    sys.argv = [sys.argv[0]]  # reset

    # Summary
    print(f"\n{'='*65}")
    print("  FLYWHEEL CYCLE COMPLETE")
    print(f"{'='*65}")
    print(f"  Retrain:           {'skipped' if skip_retrain else 'executed'}")
    print(f"  Weight adaptation:  done")
    print(f"  Uncertainty queue:  {'refreshed' if ok_unc else 'failed'}")
    print(f"  Blind-spot queue:   {'refreshed' if ok_bs else 'failed'}")
    print(f"  Acquisition policy: {'rebuilt' if ok_acq else 'failed'}")
    print(f"\n  Loop: label from acquisition_queue → retrain → refresh → repeat")


if __name__ == "__main__":
    main()
