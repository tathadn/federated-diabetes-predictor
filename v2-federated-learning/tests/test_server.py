"""Tests for src/server.py and src/strategies.py."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters

from src.config import FEATURE_COLUMNS
from src.models import get_model_parameters, initialise_model
from src.metrics import (
    compute_communication_cost,
    compute_convergence_round,
    federated_vs_centralized_report,
)
from src.strategies import LoggingFedAvg


N_FEATURES = len(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Strategy initialisation
# ---------------------------------------------------------------------------


def _make_strategy() -> LoggingFedAvg:
    model = initialise_model(N_FEATURES)
    initial_params = ndarrays_to_parameters(get_model_parameters(model))
    return LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_params,
    )


def test_strategy_instantiation():
    strategy = _make_strategy()
    assert strategy is not None


def test_strategy_has_metrics_history():
    strategy = _make_strategy()
    assert hasattr(strategy, "metrics_history")
    assert isinstance(strategy.metrics_history, list)


def test_strategy_initial_params_not_none():
    strategy = _make_strategy()
    # initial_parameters should be set
    assert strategy.initial_parameters is not None


# ---------------------------------------------------------------------------
# build_strategy
# ---------------------------------------------------------------------------


def test_build_fedavg_strategy():
    from src.server import build_strategy
    strategy = build_strategy(n_features=N_FEATURES, strategy_name="fedavg")
    assert isinstance(strategy, LoggingFedAvg)


def test_build_fedprox_strategy():
    from src.server import build_strategy
    from src.strategies import FedProxStrategy
    strategy = build_strategy(n_features=N_FEATURES, strategy_name="fedprox", mu=0.1)
    assert isinstance(strategy, FedProxStrategy)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def test_compute_convergence_round():
    history = [
        {"round": i, "accuracy": 0.5 + i * 0.03}
        for i in range(1, 11)
    ]
    cr = compute_convergence_round(history, metric="accuracy", threshold_fraction=0.90)
    assert cr is not None
    assert isinstance(cr, int)


def test_compute_convergence_round_empty():
    assert compute_convergence_round([]) is None


def test_compute_communication_cost():
    model = initialise_model(N_FEATURES)
    params = get_model_parameters(model)
    cost = compute_communication_cost(
        n_clients=3, num_rounds=10, model_params=params
    )
    assert "n_params" in cost
    assert "total_bytes" in cost
    assert cost["total_bytes"] > 0
    assert cost["total_mb"] < 1.0  # LR model is tiny


def test_federated_vs_centralized_report():
    fl_metrics = {"accuracy": 0.74, "f1": 0.70, "roc_auc": 0.80}
    cent_metrics = {"accuracy": 0.76, "f1": 0.72, "roc_auc": 0.82}
    df = federated_vs_centralized_report(fl_metrics, cent_metrics)
    assert "Centralized (V1)" in df.columns
    assert "Federated (V2)" in df.columns
    assert "Difference" in df.columns
    assert len(df) == 3
