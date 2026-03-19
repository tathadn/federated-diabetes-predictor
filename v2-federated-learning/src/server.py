"""Flower server implementation for federated diabetes prediction."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig

from src.config import (
    FEATURE_COLUMNS,
    FRACTION_EVALUATE,
    FRACTION_FIT,
    MIN_AVAILABLE_CLIENTS,
    MIN_EVALUATE_CLIENTS,
    MIN_FIT_CLIENTS,
    MODELS_DIR,
    NUM_ROUNDS,
    SERVER_ADDRESS,
)
from src.models import initialise_model, get_model_parameters
from src.strategies import LoggingFedAvg

logger = logging.getLogger(__name__)


def build_strategy(
    n_features: int,
    strategy_name: str = "fedavg",
    mu: float = 0.1,
) -> LoggingFedAvg:
    """Build the aggregation strategy with initial model parameters.

    Parameters
    ----------
    n_features : int
        Input dimensionality (needed to initialise global model weights).
    strategy_name : str
        ``'fedavg'`` or ``'fedprox'``.
    mu : float
        FedProx proximal term (ignored for FedAvg).

    Returns
    -------
    LoggingFedAvg (or FedProxStrategy)
    """
    global_model = initialise_model(n_features)
    initial_params = ndarrays_to_parameters(get_model_parameters(global_model))

    common_kwargs = dict(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        initial_parameters=initial_params,
    )

    if strategy_name == "fedprox":
        from src.strategies import FedProxStrategy
        strategy = FedProxStrategy(mu=mu, **common_kwargs)
        logger.info("Using FedProx strategy (mu=%.3f)", mu)
    else:
        strategy = LoggingFedAvg(**common_kwargs)
        logger.info("Using FedAvg strategy")

    return strategy


def start_server(
    server_address: str = SERVER_ADDRESS,
    num_rounds: int = NUM_ROUNDS,
    n_features: Optional[int] = None,
    strategy_name: str = "fedavg",
) -> None:
    """Start the Flower server for distributed training.

    Parameters
    ----------
    server_address : str
        Host:port string.
    num_rounds : int
        Number of FL rounds.
    n_features : int, optional
        Feature count — derived from FEATURE_COLUMNS if not provided.
    strategy_name : str
    """
    if n_features is None:
        n_features = len(FEATURE_COLUMNS)

    strategy = build_strategy(n_features, strategy_name=strategy_name)

    logger.info(
        "Starting FL server at %s for %d rounds", server_address, num_rounds
    )
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )

    # Save the metrics history produced by the strategy
    _save_server_metrics(strategy, MODELS_DIR)


def _save_server_metrics(strategy: LoggingFedAvg, output_dir: Path) -> None:
    """Persist strategy metrics history to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "server_metrics_history.pkl"
    with open(path, "wb") as f:
        pickle.dump(strategy.metrics_history, f)
    logger.info("Server metrics history saved → %s", path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Start FL server")
    parser.add_argument("--address", default=SERVER_ADDRESS)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox"])
    args = parser.parse_args()

    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        strategy_name=args.strategy,
    )
