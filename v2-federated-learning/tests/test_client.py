"""Tests for src/client.py — DiabetesClient."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.client import DiabetesClient
from src.models import get_model_parameters, initialise_model, set_model_parameters
from src.config import FEATURE_COLUMNS


N_FEATURES = len(FEATURE_COLUMNS)
N_TRAIN = 200
N_TEST = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    X_train = np.random.randn(N_TRAIN, N_FEATURES).astype(np.float32)
    y_train = np.random.randint(0, 2, N_TRAIN).astype(np.float32)
    X_test = np.random.randn(N_TEST, N_FEATURES).astype(np.float32)
    y_test = np.random.randint(0, 2, N_TEST).astype(np.float32)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def client(synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data
    model = initialise_model(N_FEATURES)
    return DiabetesClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        client_id="0",
        local_epochs=1,
    )


# ---------------------------------------------------------------------------
# get_parameters
# ---------------------------------------------------------------------------


def test_get_parameters_returns_list(client):
    params = client.get_parameters(config={})
    assert isinstance(params, list)
    assert len(params) == 2  # coef_ and intercept_


def test_get_parameters_shapes(client):
    params = client.get_parameters(config={})
    coef, intercept = params
    assert coef.shape == (1, N_FEATURES)
    assert intercept.shape == (1,)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_returns_tuple(client):
    params = client.get_parameters(config={})
    result = client.fit(params, config={"local_epochs": 1})
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_fit_updated_params_shape(client):
    params = client.get_parameters(config={})
    updated_params, num_examples, metrics = client.fit(params, config={})
    assert updated_params[0].shape == (1, N_FEATURES)
    assert updated_params[1].shape == (1,)


def test_fit_num_examples(client, synthetic_data):
    X_train, y_train, *_ = synthetic_data
    params = client.get_parameters(config={})
    _, num_examples, _ = client.fit(params, config={})
    assert num_examples == N_TRAIN


def test_fit_metrics_contains_accuracy(client):
    params = client.get_parameters(config={})
    _, _, metrics = client.fit(params, config={})
    assert "train_accuracy" in metrics
    assert 0.0 <= metrics["train_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_tuple(client):
    params = client.get_parameters(config={})
    client.fit(params, config={})
    params = client.get_parameters(config={})
    result = client.evaluate(params, config={})
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_evaluate_num_examples(client):
    params = client.get_parameters(config={})
    client.fit(params, config={})
    _, num_examples, _ = client.evaluate(params, config={})
    assert num_examples == N_TEST


def test_evaluate_loss_positive(client):
    params = client.get_parameters(config={})
    client.fit(params, config={})
    loss, _, _ = client.evaluate(params, config={})
    assert loss >= 0.0


def test_evaluate_metrics_keys(client):
    params = client.get_parameters(config={})
    client.fit(params, config={})
    _, _, metrics = client.evaluate(params, config={})
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_evaluate_accuracy_range(client):
    params = client.get_parameters(config={})
    client.fit(params, config={})
    _, _, metrics = client.evaluate(params, config={})
    assert 0.0 <= metrics["accuracy"] <= 1.0
