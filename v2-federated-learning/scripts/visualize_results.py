"""Generate all result plots from saved metrics and model artefacts.

Run after training:
    python scripts/visualize_results.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PLOTS_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def load_fl_history() -> list:
    path = MODELS_DIR / "fl_metrics_history.json"
    if not path.exists():
        logger.warning("FL metrics not found at %s. Run training first.", path)
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("fl_metrics_history", [])


def load_comparison() -> dict:
    path = REPORTS_DIR / "v1_v2_comparison.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

STYLE = {"linewidth": 2, "marker": "o", "markersize": 4}


def plot_convergence(history: list) -> None:
    if not history:
        return

    rounds = [h["round"] for h in history]
    accuracies = [h.get("accuracy", 0.0) for h in history]
    losses = [h.get("loss") or 0.0 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Federated Learning Convergence", fontsize=14, fontweight="bold")

    axes[0].plot(rounds, accuracies, color="steelblue", **STYLE)
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Global Accuracy")
    axes[0].set_title("Accuracy vs. Round")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, losses, color="firebrick", **STYLE)
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_title("Loss vs. Round")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "convergence_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_communication_cost(history: list, n_clients: int = 3) -> None:
    if not history:
        return

    rounds = [h["round"] for h in history]
    # Each round: n_clients send & receive model (21 features LR ≈ tiny)
    # Simulate cumulative MB (illustrative)
    bytes_per_round = n_clients * 2 * (21 + 1) * 8  # coef + intercept, float64
    cum_mb = [r * bytes_per_round / (1024 ** 2) for r in rounds]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, cum_mb, color="darkorange", **STYLE)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cumulative Data Transferred (MB)")
    ax.set_title("Communication Cost vs. Round")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "communication_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_comparison(comparison: dict) -> None:
    if not comparison:
        logger.warning("No comparison data found. Skipping comparison plot.")
        return

    metrics = list(comparison.get("centralized", {}).keys())
    v1_vals = [comparison["centralized"].get(m, 0.0) for m in metrics]
    v2_vals = [comparison["federated"].get(m, 0.0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, v1_vals, width, label="Centralized (V1)", color="steelblue")
    bars2 = ax.bar(x + width / 2, v2_vals, width, label="Federated (V2)", color="darkorange")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("V1 (Centralized) vs V2 (Federated) Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "v1_v2_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_privacy_utility(history: list) -> None:
    """Illustrative privacy-utility trade-off plot across epsilon values."""
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
    # Simulate accuracy degradation with stronger DP (lower epsilon)
    if history:
        base_acc = history[-1].get("accuracy", 0.75)
    else:
        base_acc = 0.75

    # Rough model: accuracy improves as epsilon increases (less noise)
    accs = []
    for eps in epsilons[:-1]:
        noise_factor = 1 - 0.15 * (1 / eps)
        accs.append(max(0.5, base_acc * noise_factor))
    accs.append(base_acc)  # no DP

    x_labels = [str(e) for e in epsilons[:-1]] + ["∞ (no DP)"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(epsilons)), accs, color="purple", **STYLE)
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Model Accuracy")
    ax.set_title("Privacy-Utility Trade-off")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "privacy_utility_tradeoff.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    history = load_fl_history()
    comparison = load_comparison()

    plot_convergence(history)
    plot_communication_cost(history)
    plot_comparison(comparison)
    plot_privacy_utility(history)

    if not history and not comparison:
        print(
            "\n[!] No results found. Run training first:\n"
            "    python scripts/run_federated_training.py\n"
            "Then optionally run notebook 04 to produce the comparison JSON."
        )
    else:
        print(f"\n[✓] Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
