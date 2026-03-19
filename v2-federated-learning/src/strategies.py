"""Custom Flower aggregation strategies with enhanced logging."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class LoggingFedAvg(FedAvg):
    """FedAvg strategy that logs per-round aggregated metrics.

    Inherits all FedAvg behaviour and overrides the aggregate callbacks
    to record round-level statistics, which are accessible via
    ``self.metrics_history`` after training.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metrics_history: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and log per-client training metrics."""
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if results:
            train_accs = [
                res.metrics.get("train_accuracy", 0.0)
                for _, res in results
                if res.metrics
            ]
            logger.info(
                "Round %d | fit — clients=%d, mean_train_acc=%.4f, failures=%d",
                server_round,
                len(results),
                float(np.mean(train_accs)) if train_accs else 0.0,
                len(failures),
            )

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and record to history."""
        loss_aggregated, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if results:
            accs = [res.metrics.get("accuracy", 0.0) for _, res in results if res.metrics]
            f1s = [res.metrics.get("f1", 0.0) for _, res in results if res.metrics]
            aucs = [res.metrics.get("roc_auc", 0.0) for _, res in results if res.metrics]

            round_metrics = {
                "round": server_round,
                "loss": float(loss_aggregated) if loss_aggregated is not None else None,
                "accuracy": float(np.mean(accs)),
                "f1": float(np.mean(f1s)),
                "roc_auc": float(np.mean(aucs)),
                "n_clients": len(results),
                "n_failures": len(failures),
            }
            self.metrics_history.append(round_metrics)

            logger.info(
                "Round %d | eval — loss=%.4f, acc=%.4f, f1=%.4f, auc=%.4f",
                server_round,
                round_metrics["loss"] or 0.0,
                round_metrics["accuracy"],
                round_metrics["f1"],
                round_metrics["roc_auc"],
            )

        return loss_aggregated, metrics


class FedProxStrategy(FedAvg):
    """FedProx strategy — FedAvg with a proximal regularisation term.

    The proximal term ``mu * ||w - w_global||^2`` is applied on the client
    side. This strategy is otherwise identical to FedAvg on the server side;
    clients must be aware of ``mu`` and implement the regularisation during
    local training.

    Parameters
    ----------
    mu : float
        Proximal term coefficient.
    """

    def __init__(self, mu: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mu = mu
        self.metrics_history: List[Dict] = []

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject mu into the fit config so clients can use it."""
        config_pairs = super().configure_fit(server_round, parameters, client_manager)
        # Append mu to each client's config
        updated = []
        for client, fit_ins in config_pairs:
            new_config = dict(fit_ins.config)
            new_config["mu"] = self.mu
            updated.append((client, fl.common.FitIns(fit_ins.parameters, new_config)))
        return updated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        if results:
            accs = [res.metrics.get("accuracy", 0.0) for _, res in results if res.metrics]
            round_metrics = {
                "round": server_round,
                "loss": float(loss_aggregated) if loss_aggregated is not None else None,
                "accuracy": float(np.mean(accs)),
            }
            self.metrics_history.append(round_metrics)
        return loss_aggregated, metrics
