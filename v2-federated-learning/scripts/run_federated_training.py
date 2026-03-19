"""Main entry point for federated training.

Supports two modes:
    --mode simulation   Run all clients on a single machine (default).
    --mode distributed  Start only the server; clients connect separately.

Usage
-----
Single-machine simulation (recommended for development):
    python scripts/run_federated_training.py

Distributed (start in separate terminals):
    python scripts/run_federated_training.py --mode distributed
    python src/client.py 0
    python src/client.py 1
    python src/client.py 2
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters

from src.config import (
    DIRICHLET_ALPHA,
    FEATURE_COLUMNS,
    FEDERATED_DATA_DIR,
    LOCAL_EPOCHS,
    MODELS_DIR,
    N_CLIENTS,
    NUM_ROUNDS,
    RANDOM_STATE,
    SERVER_ADDRESS,
)
from src.data_utils import (
    load_dataset,
    create_noniid_splits,
    save_client_data,
    prepare_client_tensors,
    get_global_scaler,
)
from src.models import initialise_model, get_model_parameters
from src.strategies import LoggingFedAvg
from src.client import DiabetesClient
from src.metrics import save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def prepare_data(force: bool = False) -> None:
    """Load raw dataset and create non-IID client splits if not present."""
    client_files = [
        FEDERATED_DATA_DIR / f"client_{i}_data.csv" for i in range(N_CLIENTS)
    ]
    if all(f.exists() for f in client_files) and not force:
        logger.info("Client data already exists — skipping split creation.")
        return

    logger.info("Creating non-IID data splits (alpha=%.2f) …", DIRICHLET_ALPHA)
    df = load_dataset()
    client_dfs = create_noniid_splits(df, n_clients=N_CLIENTS, alpha=DIRICHLET_ALPHA)
    save_client_data(client_dfs)
    logger.info("Client data saved to %s", FEDERATED_DATA_DIR)


def make_client_fn(client_dfs, scaler, n_features):
    """Return a client_fn closure for Flower simulation."""

    def client_fn(cid: str) -> DiabetesClient:
        df = client_dfs[int(cid)]
        X_train, y_train, X_test, y_test, _ = prepare_client_tensors(
            df, scaler=scaler
        )
        model = initialise_model(n_features)
        return DiabetesClient(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            client_id=cid,
            local_epochs=LOCAL_EPOCHS,
        )

    return client_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_simulation(num_rounds: int, strategy_name: str) -> None:
    """Run federated learning on a single machine using Flower simulation."""
    prepare_data()

    # Load all client dataframes
    from src.data_utils import load_client_data
    client_dfs = {i: load_client_data(i) for i in range(N_CLIENTS)}

    # Fit a shared scaler on the combined data
    scaler = get_global_scaler(client_dfs)
    n_features = len(FEATURE_COLUMNS)

    # Build initial parameters
    global_model = initialise_model(n_features)
    initial_params = ndarrays_to_parameters(get_model_parameters(global_model))

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=N_CLIENTS,
        min_evaluate_clients=N_CLIENTS,
        min_available_clients=N_CLIENTS,
        initial_parameters=initial_params,
    )

    if strategy_name == "fedprox":
        from src.strategies import FedProxStrategy
        strategy = FedProxStrategy(mu=0.1, **common_kwargs)
    else:
        strategy = LoggingFedAvg(**common_kwargs)

    client_fn = make_client_fn(client_dfs, scaler, n_features)

    logger.info(
        "Starting simulation: %d clients, %d rounds, strategy=%s",
        N_CLIENTS,
        num_rounds,
        strategy_name,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=N_CLIENTS,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    # Persist results
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_history = strategy.metrics_history
    save_metrics(
        {"fl_metrics_history": metrics_history},
        MODELS_DIR / "fl_metrics_history.json",
    )

    # Save the final global model
    if metrics_history:
        logger.info(
            "Final round metrics: %s", metrics_history[-1]
        )

    # Reconstruct and save global model from last aggregated params
    _save_global_model(strategy, n_features)

    logger.info("Simulation complete. Results saved to %s", MODELS_DIR)


def _save_global_model(strategy: LoggingFedAvg, n_features: int) -> None:
    """Extract and pickle the final global model."""
    try:
        params = strategy.parameters  # type: ignore[attr-defined]
        if params is None:
            return
        from flwr.common import parameters_to_ndarrays
        arrays = parameters_to_ndarrays(params)
        from src.models import set_model_parameters, initialise_model
        model = initialise_model(n_features)
        set_model_parameters(model, arrays)

        path = MODELS_DIR / "global_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Global model saved → %s", path)
    except Exception as exc:
        logger.warning("Could not save global model: %s", exc)


def run_distributed(num_rounds: int, strategy_name: str) -> None:
    """Start only the server; clients must connect separately."""
    from src.server import start_server
    prepare_data()
    start_server(
        server_address=SERVER_ADDRESS,
        num_rounds=num_rounds,
        strategy_name=strategy_name,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated learning training")
    parser.add_argument(
        "--mode",
        choices=["simulation", "distributed"],
        default="simulation",
        help="'simulation' runs everything locally; 'distributed' starts the server only.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help=f"Number of FL rounds (default: {NUM_ROUNDS})",
    )
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox"],
        default="fedavg",
        help="Aggregation strategy (default: fedavg)",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Recreate client data splits even if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "simulation":
        run_simulation(num_rounds=args.rounds, strategy_name=args.strategy)
    else:
        run_distributed(num_rounds=args.rounds, strategy_name=args.strategy)
