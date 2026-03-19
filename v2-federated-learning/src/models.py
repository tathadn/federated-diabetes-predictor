"""Shared model definitions for federated and centralized training."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import LOGISTIC_PARAMS, RANDOM_STATE

logger = logging.getLogger(__name__)


def get_logistic_regression(params: Optional[dict] = None) -> LogisticRegression:
    """Return a LogisticRegression instance suitable for federated learning.

    ``warm_start=True`` allows incremental weight updates between FL rounds.
    """
    defaults = dict(LOGISTIC_PARAMS)
    if params:
        defaults.update(params)
    return LogisticRegression(**defaults)


def get_random_forest(params: Optional[dict] = None) -> RandomForestClassifier:
    """Return a RandomForestClassifier (for baseline comparison)."""
    defaults = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    if params:
        defaults.update(params)
    return RandomForestClassifier(**defaults)


# ---------------------------------------------------------------------------
# Parameter extraction / injection helpers for Flower
# ---------------------------------------------------------------------------


def get_model_parameters(model: LogisticRegression) -> List[np.ndarray]:
    """Extract Logistic Regression weights as a list of numpy arrays.

    Returns [coef_, intercept_] if the model has been fitted, otherwise
    returns zero-initialised arrays of the expected shape.
    """
    if hasattr(model, "coef_"):
        return [model.coef_, model.intercept_]
    # Return empty placeholders — shape will be set after first fit
    return []


def set_model_parameters(
    model: LogisticRegression, parameters: List[np.ndarray]
) -> LogisticRegression:
    """Inject [coef_, intercept_] into a LogisticRegression model."""
    if len(parameters) == 2:
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        model.classes_ = np.array([0.0, 1.0])
    return model


def initialise_model(n_features: int) -> LogisticRegression:
    """Create a LogisticRegression with zero-initialised weights.

    Flower needs the server to broadcast initial parameters before round 1.
    """
    model = get_logistic_regression()
    # Manually set attributes so get_model_parameters works before any fit
    model.coef_ = np.zeros((1, n_features), dtype=np.float64)
    model.intercept_ = np.zeros(1, dtype=np.float64)
    model.classes_ = np.array([0.0, 1.0])
    return model
