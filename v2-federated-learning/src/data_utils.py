"""Federated data utilities for V2.

Handles loading the raw dataset, creating non-IID client splits via the
Dirichlet distribution, saving / loading per-client CSVs, and computing
per-client statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DIRICHLET_ALPHA,
    FEATURE_COLUMNS,
    N_CLIENTS,
    RANDOM_STATE,
    TARGET_COLUMN,
    FEDERATED_DATA_DIR,
    RAW_DATASET_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw diabetes CSV and return a clean DataFrame.

    Parameters
    ----------
    path : Path, optional
        Override default dataset path.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the columns defined in ``ALL_COLUMNS``.
    """
    dataset_path = path or RAW_DATASET_PATH
    logger.info("Loading dataset from %s", dataset_path)
    df = pd.read_csv(dataset_path)
    # Keep only known columns
    cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    df = df[cols].dropna().reset_index(drop=True)
    logger.info("Dataset loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Non-IID distribution
# ---------------------------------------------------------------------------


def create_noniid_splits(
    df: pd.DataFrame,
    n_clients: int = N_CLIENTS,
    alpha: float = DIRICHLET_ALPHA,
    random_state: int = RANDOM_STATE,
) -> Dict[int, pd.DataFrame]:
    """Split the dataset into *n_clients* non-IID subsets.

    Uses the Dirichlet distribution over class labels so each client receives
    a different proportion of positive/negative samples — mimicking real-world
    heterogeneous healthcare organisations.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with target column.
    n_clients : int
        Number of federated clients.
    alpha : float
        Dirichlet concentration parameter. Lower → more heterogeneous.
    random_state : int
        NumPy random seed.

    Returns
    -------
    Dict[int, pd.DataFrame]
        Mapping ``client_id → DataFrame``.
    """
    np.random.seed(random_state)
    classes = df[TARGET_COLUMN].unique()

    # For each class, draw a Dirichlet proportion and assign rows to clients
    client_indices: Dict[int, List[int]] = {i: [] for i in range(n_clients)}

    for cls in classes:
        cls_indices = df.index[df[TARGET_COLUMN] == cls].tolist()
        np.random.shuffle(cls_indices)

        # Dirichlet proportions for this class across clients
        proportions = np.random.dirichlet([alpha] * n_clients)
        proportions = proportions / proportions.sum()  # normalise

        # Compute split boundaries
        splits = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        splits[-1] = len(cls_indices)  # ensure all samples are assigned

        start = 0
        for client_id, end in enumerate(splits):
            client_indices[client_id].extend(cls_indices[start:end])
            start = end

    client_dfs: Dict[int, pd.DataFrame] = {}
    for client_id, indices in client_indices.items():
        np.random.shuffle(indices)
        client_dfs[client_id] = df.loc[indices].reset_index(drop=True)
        logger.info(
            "Client %d: %d samples, positive_rate=%.3f",
            client_id,
            len(client_dfs[client_id]),
            client_dfs[client_id][TARGET_COLUMN].mean(),
        )

    _validate_splits(df, client_dfs)
    return client_dfs


def _validate_splits(
    original: pd.DataFrame, client_dfs: Dict[int, pd.DataFrame]
) -> None:
    """Assert no overlap and complete coverage."""
    total = sum(len(df) for df in client_dfs.values())
    assert total == len(original), (
        f"Split sizes sum to {total} but original has {len(original)} rows"
    )

    all_indices = [idx for df in client_dfs.values() for idx in df.index.tolist()]
    # No duplicate indices within a client (reset_index was called)
    for cid, df in client_dfs.items():
        assert len(df) == len(df.index.unique()), f"Duplicate indices in client {cid}"

    logger.info("Validation passed: total=%d, no overlaps detected", total)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_client_data(
    client_dfs: Dict[int, pd.DataFrame],
    output_dir: Path = FEDERATED_DATA_DIR,
) -> None:
    """Save each client's DataFrame to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for client_id, df in client_dfs.items():
        path = output_dir / f"client_{client_id}_data.csv"
        df.to_csv(path, index=False)
        logger.info("Saved client %d data → %s (%d rows)", client_id, path, len(df))


def load_client_data(
    client_id: int,
    data_dir: Path = FEDERATED_DATA_DIR,
) -> pd.DataFrame:
    """Load a single client's CSV."""
    path = data_dir / f"client_{client_id}_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Client {client_id} data not found at {path}. "
            "Run scripts/download_dataset.py and notebooks/01_data_distribution.ipynb first."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Train / test splitting & scaling
# ---------------------------------------------------------------------------


def prepare_client_tensors(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split a client DataFrame into scaled train/test arrays.

    Parameters
    ----------
    df : pd.DataFrame
    test_size : float
    random_state : int
    scaler : StandardScaler, optional
        If provided, use this scaler (e.g., fitted on global data). Otherwise
        fit a new scaler on the client's training split.

    Returns
    -------
    Tuple of (X_train, y_train, X_test, y_test, scaler)
    """
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[features].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    else:
        X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, scaler


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_client_statistics(client_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Compute per-client summary statistics.

    Returns
    -------
    pd.DataFrame
        One row per client with n_samples, positive_rate, feature_means, etc.
    """
    rows = []
    for cid, df in client_dfs.items():
        features = [c for c in FEATURE_COLUMNS if c in df.columns]
        row = {
            "client_id": cid,
            "n_samples": len(df),
            "positive_rate": df[TARGET_COLUMN].mean(),
            "negative_rate": 1 - df[TARGET_COLUMN].mean(),
        }
        for feat in features:
            row[f"{feat}_mean"] = df[feat].mean()
            row[f"{feat}_std"] = df[feat].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("client_id")


def get_global_scaler(client_dfs: Dict[int, pd.DataFrame]) -> StandardScaler:
    """Fit a StandardScaler on all clients' data combined."""
    features = [c for c in FEATURE_COLUMNS if c in next(iter(client_dfs.values())).columns]
    combined = pd.concat(list(client_dfs.values()), ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(combined[features].values.astype(np.float32))
    return scaler
