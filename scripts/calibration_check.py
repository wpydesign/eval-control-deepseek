#!/usr/bin/env python3
"""
calibration_check.py — Probability calibration analysis for failure predictor [v2.2.1]

Answers: "When the model says 80% risk, is it actually 80% wrong?"

Outputs:
  model/calibration_report.json   — ECE, per-bin stats, calibration quality
  model/reliability_diagram.png   — visual reliability diagram (predicted vs actual)

Usage:
  python scripts/calibration_check.py
"""

import json
import os
import sys
import numpy as np
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
MODEL_PATH = os.path.join(BASE, "model", "failure_predictor.pkl")
CALIBRATION_REPORT_PATH = os.path.join(BASE, "model", "calibration_report.json")
RELIABILITY_DIAGRAM_PATH = os.path.join(BASE, "model", "reliability_diagram.png")

N_BINS = 10


def load_labeled_data():
    """Load labeled samples from failure_dataset.jsonl."""
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: {DATASET_PATH} not found. Run build_failure_dataset.py first.")
        sys.exit(1)

    labeled = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_wrong") is not None:
                labeled.append(row)
    return labeled


def get_predictions(predictor, samples):
    """Get risk_score predictions for all labeled samples."""
    y_true = []
    y_prob = []
    for s in samples:
        v4 = {
            "S": s["S_v4"],
            "kappa": s["kappa_v4"],
            "delta_G": s["delta_G_v4"],
            "delta_L": s["delta_L_v4"],
        }
        v1 = {"S": s["S_v1"]}
        risk = predictor.predict(v4, v1)
        y_true.append(s["is_wrong"])
        y_prob.append(risk["risk_score"])

    return np.array(y_true), np.array(y_prob)


def compute_ece(y_true, y_prob, n_bins=N_BINS):
    """Compute Expected Calibration Error with per-bin stats."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []

    ece = 0.0
    total_n = len(y_true)

    for i in range(n_bins):
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n = int(mask.sum())
        if n == 0:
            bins.append({
                "bin_low": round(float(bin_edges[i]), 2),
                "bin_high": round(float(bin_edges[i + 1]), 2),
                "bin_center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2),
                "n": 0, "pred_mean": None, "actual_rate": None,
                "gap": None, "weight": 0.0,
            })
            continue

        pred_mean = float(y_prob[mask].mean())
        actual_rate = float(y_true[mask].mean())
        gap = pred_mean - actual_rate  # positive = overconfident, negative = underconfident
        weight = n / total_n
        ece += abs(gap) * weight

        bins.append({
            "bin_low": round(float(bin_edges[i]), 2),
            "bin_high": round(float(bin_edges[i + 1]), 2),
            "bin_center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2),
            "n": n,
            "pred_mean": round(pred_mean, 4),
            "actual_rate": round(actual_rate, 4),
            "gap": round(gap, 4),
            "weight": round(weight, 4),
        })

    return round(ece, 4), bins


def compute_brier(y_true, y_prob):
    """Compute Brier Score (mean squared error of probabilities)."""
    from sklearn.metrics import brier_score_loss
    return round(float(brier_score_loss(y_true, y_prob)), 4)


def plot_reliability_diagram(bins, ece, output_path):
    """Plot reliability diagram: predicted probability vs actual frequency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Filter out empty bins
    non_empty = [b for b in bins if b["n"] > 0]

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", zorder=1)

    # Actual calibration points
    centers = [b["bin_center"] for b in non_empty]
    actuals = [b["actual_rate"] for b in non_empty]
    sizes = [max(30, b["n"] * 5) for b in non_empty]  # scale by sample count

    ax.scatter(centers, actuals, s=sizes, alpha=0.6, color="#e74c3c",
               edgecolors="#c0392b", linewidth=1, zorder=3, label="Model bins")

    # Connect dots for visual trend
    sorted_idx = np.argsort(centers)
    ax.plot(np.array(centers)[sorted_idx], np.array(actuals)[sorted_idx],
            "-", color="#3498db", linewidth=2, alpha=0.7, zorder=2, label="Model trend")

    # Annotations for high-gap bins
    for b in non_empty:
        if b["n"] >= 5 and abs(b["gap"]) > 0.08:
            direction = "underconfident" if b["gap"] < 0 else "overconfident"
            ax.annotate(
                f"{b['actual_rate']:.0%}\n({direction})",
                (b["bin_center"], b["actual_rate"]),
                textcoords="offset points",
                xytext=(0, 12),
                fontsize=7,
                ha="center",
                color="#555",
            )

    ax.set_xlabel("Predicted P(is_wrong)", fontsize=12)
    ax.set_ylabel("Actual wrong rate", fontsize=12)
    ax.set_title(f"Reliability Diagram — ECE = {ece:.3f}", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Reliability diagram saved to {output_path}")


def main():
    from scripts.predict_failure import FailurePredictor

    print("Loading predictor...")
    predictor = FailurePredictor()
    if not predictor.is_loaded:
        print("ERROR: No trained model. Run train_failure_predictor.py first.")
        sys.exit(1)

    print("Loading labeled data...")
    samples = load_labeled_data()
    print(f"  {len(samples)} labeled samples")

    print("Computing predictions...")
    y_true, y_prob = get_predictions(predictor, samples)

    # ECE
    print("Computing calibration metrics...")
    ece, bins = compute_ece(y_true, y_prob)
    brier = compute_brier(y_true, y_prob)

    # Print report
    print(f"\n{'='*60}")
    print(f"  CALIBRATION REPORT")
    print(f"{'='*60}")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  Brier Score:                      {brier:.4f}")
    print(f"  Mean predicted P(wrong):          {y_prob.mean():.3f}")
    print(f"  Actual wrong rate:                {y_true.mean():.3f}")
    print(f"  Prediction-Reality gap:           {y_prob.mean() - y_true.mean():+.3f}")

    # Interpretation
    if ece < 0.05:
        quality = "EXCELLENT — probabilities are trustworthy"
    elif ece < 0.10:
        quality = "GOOD — minor calibration drift"
    elif ece < 0.15:
        quality = "MODERATE — calibration correction recommended"
    else:
        quality = "POOR — probabilities need recalibration"
    print(f"\n  Calibration quality: {quality}")

    print(f"\n  {'Bin':>8s} {'n':>5s} {'pred_mean':>10s} {'actual':>10s} {'gap':>8s} {'n%':>6s}")
    print(f"  {'-'*49}")
    for b in bins:
        if b["n"] == 0:
            continue
        gap_str = f"{b['gap']:+.3f}"
        direction = " ⚠" if abs(b["gap"]) > 0.10 else ""
        pct = f"{b['weight']*100:.0f}%"
        print(f"  {b['bin_center']:>7.2f} {b['n']:>5d} {b['pred_mean']:>10.3f} "
              f"{b['actual_rate']:>10.3f} {gap_str:>7s} {pct:>5s}{direction}")

    # Save report
    os.makedirs(os.path.dirname(CALIBRATION_REPORT_PATH), exist_ok=True)
    report = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(y_true),
        "n_bins": N_BINS,
        "ece": ece,
        "brier_score": brier,
        "mean_predicted": round(float(y_prob.mean()), 4),
        "mean_actual": round(float(y_true.mean()), 4),
        "calibration_quality": quality,
        "bins": [b for b in bins if b["n"] > 0],
    }
    with open(CALIBRATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to {CALIBRATION_REPORT_PATH}")

    # Plot reliability diagram
    plot_reliability_diagram(bins, ece, RELIABILITY_DIAGRAM_PATH)

    print(f"\n  Bottom line: ECE={ece:.3f} — {quality}")


if __name__ == "__main__":
    main()
