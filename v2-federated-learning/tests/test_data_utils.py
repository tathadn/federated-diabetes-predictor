"""Tests for src/data_utils.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_utils import (
    compute_client_statistics,
    create_noniid_splits,
    get_global_scaler,
    prepare_client_tensors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic dataset with the expected schema."""
    np.random.seed(0)
    n = 300
    data = {col: np.random.randint(0, 5, size=n).astype(float) for col in FEATURE_COLUMNS}
    data["BMI"] = np.random.uniform(18, 45, size=n)
    data[TARGET_COLUMN] = np.random.randint(0, 2, size=n).astype(float)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# create_noniid_splits
# ---------------------------------------------------------------------------


def test_split_total_rows(sample_df):
    """All rows must be assigned across clients."""
    splits = create_noniid_splits(sample_df, n_clients=3, alpha=0.5)
    total = sum(len(df) for df in splits.values())
    assert total == len(sample_df)


def test_split_n_clients(sample_df):
    splits = create_noniid_splits(sample_df, n_clients=3, alpha=0.5)
    assert len(splits) == 3


def test_split_no_empty_client(sample_df):
    splits = create_noniid_splits(sample_df, n_clients=3, alpha=0.5)
    for cid, df in splits.items():
        assert len(df) > 0, f"Client {cid} has no samples"


def test_split_heterogeneity(sample_df):
    """With alpha=0.1 the clients should have different positive rates."""
    splits = create_noniid_splits(sample_df, n_clients=3, alpha=0.1, random_state=7)
    rates = [df[TARGET_COLUMN].mean() for df in splits.values()]
    # At least two clients should differ by more than 0.02
    diffs = [abs(rates[i] - rates[j]) for i in range(3) for j in range(i + 1, 3)]
    assert max(diffs) > 0.01


def test_split_reproducible(sample_df):
    splits_a = create_noniid_splits(sample_df, n_clients=3, alpha=0.5, random_state=42)
    splits_b = create_noniid_splits(sample_df, n_clients=3, alpha=0.5, random_state=42)
    for cid in splits_a:
        pd.testing.assert_frame_equal(
            splits_a[cid].reset_index(drop=True),
            splits_b[cid].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# prepare_client_tensors
# ---------------------------------------------------------------------------


def test_prepare_tensors_shapes(sample_df):
    X_train, y_train, X_test, y_test, scaler = prepare_client_tensors(
        sample_df, test_size=0.2
    )
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(sample_df)


def test_prepare_tensors_scaled(sample_df):
    X_train, _, _, _, scaler = prepare_client_tensors(sample_df)
    # Training data should be approximately zero-mean
    assert abs(X_train.mean()) < 0.5


# ---------------------------------------------------------------------------
# compute_client_statistics
# ---------------------------------------------------------------------------


def test_statistics_shape(sample_df):
    splits = create_noniid_splits(sample_df, n_clients=3)
    stats = compute_client_statistics(splits)
    assert len(stats) == 3
    assert "n_samples" in stats.columns
    assert "positive_rate" in stats.columns


def test_statistics_positive_rate_range(sample_df):
    splits = create_noniid_splits(sample_df, n_clients=3)
    stats = compute_client_statistics(splits)
    assert (stats["positive_rate"] >= 0).all()
    assert (stats["positive_rate"] <= 1).all()


# ---------------------------------------------------------------------------
# get_global_scaler
# ---------------------------------------------------------------------------


def test_global_scaler(sample_df):
    splits = create_noniid_splits(sample_df, n_clients=3)
    scaler = get_global_scaler(splits)
    features = [c for c in FEATURE_COLUMNS if c in sample_df.columns]
    assert scaler.mean_.shape[0] == len(features)
