"""Federated metrics calculation and reporting utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str = "model",
) -> Dict[str, float]:
    """Return a dict of standard classification metrics.

    Parameters
    ----------
    model : sklearn estimator
        Fitted classifier with ``predict`` and ``predict_proba``.
    X_test : np.ndarray
    y_test : np.ndarray
    label : str
        Display label used in log messages.

    Returns
    -------
    Dict[str, float]
        Keys: accuracy, f1, roc_auc, log_loss.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    loss = float(log_loss(y_test, y_prob))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = 0.0

    metrics = {"accuracy": accuracy, "f1": f1, "roc_auc": auc, "log_loss": loss}
    logger.info(
        "%s — acc=%.4f, f1=%.4f, auc=%.4f, loss=%.4f",
        label,
        accuracy,
        f1,
        auc,
        loss,
    )
    return metrics


# ---------------------------------------------------------------------------
# FL-specific metrics
# ---------------------------------------------------------------------------


def compute_convergence_round(
    history: List[Dict],
    metric: str = "accuracy",
    threshold_fraction: float = 0.90,
) -> Optional[int]:
    """Return the first round where metric reaches *threshold_fraction* of its final value.

    Parameters
    ----------
    history : list of dicts with keys 'round' and the metric name.
    metric : str
    threshold_fraction : float
        e.g. 0.90 means "90% of the final accuracy".

    Returns
    -------
    int or None
        Round number, or None if the threshold was never reached.
    """
    if not history:
        return None
    values = [h[metric] for h in history if metric in h]
    if not values:
        return None
    target = threshold_fraction * max(values)
    for h in history:
        if h.get(metric, 0.0) >= target:
            return int(h["round"])
    return None


def compute_communication_cost(
    n_clients: int,
    num_rounds: int,
    model_params: List[np.ndarray],
    bytes_per_float: int = 8,
) -> Dict[str, float]:
    """Estimate total communication cost in bytes.

    Parameters
    ----------
    n_clients : int
    num_rounds : int
    model_params : list of np.ndarray
        Model parameter arrays (used to count total floats).
    bytes_per_float : int
        Bytes per floating-point value (default 8 for float64).

    Returns
    -------
    Dict with keys: n_params, bytes_per_round_per_client, total_bytes, total_mb.
    """
    n_params = sum(p.size for p in model_params)
    bytes_per_transmission = n_params * bytes_per_float
    # Each round: n_clients download + n_clients upload
    total_bytes = 2 * n_clients * num_rounds * bytes_per_transmission
    return {
        "n_params": n_params,
        "bytes_per_round_per_client": bytes_per_transmission,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 ** 2),
    }


def federated_vs_centralized_report(
    fl_metrics: Dict[str, float],
    centralized_metrics: Dict[str, float],
) -> pd.DataFrame:
    """Build a comparison DataFrame of FL vs centralised metrics.

    Parameters
    ----------
    fl_metrics : dict
        Metrics from the federated global model.
    centralized_metrics : dict
        Metrics from the V1 centralised model.

    Returns
    -------
    pd.DataFrame
        Rows = metric names, cols = [Centralized, Federated, Difference].
    """
    shared_keys = set(fl_metrics) & set(centralized_metrics)
    rows = []
    for key in sorted(shared_keys):
        v1 = centralized_metrics[key]
        v2 = fl_metrics[key]
        rows.append({
            "Metric": key,
            "Centralized (V1)": round(v1, 4),
            "Federated (V2)": round(v2, 4),
            "Difference": round(v2 - v1, 4),
        })
    return pd.DataFrame(rows).set_index("Metric")


# ---------------------------------------------------------------------------
# Saving / loading metrics
# ---------------------------------------------------------------------------


def save_metrics(metrics: Dict, path: Path) -> None:
    """Serialise metrics dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", path)


def load_metrics(path: Path) -> Dict:
    """Load metrics dict from JSON."""
    with open(path) as f:
        return json.load(f)
