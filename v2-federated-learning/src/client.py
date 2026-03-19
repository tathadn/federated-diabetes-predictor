"""Flower client implementation for federated diabetes prediction."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score

from src.models import get_model_parameters, set_model_parameters

logger = logging.getLogger(__name__)


class DiabetesClient(fl.client.NumPyClient):
    """Federated learning client for diabetes prediction.

    Each instance holds its own local train/test split and a LogisticRegression
    model. The client participates in each FL round by:

    1. Receiving the global model parameters from the server.
    2. Training locally for ``local_epochs`` passes.
    3. Returning the updated parameters alongside local metrics.

    Parameters
    ----------
    model : LogisticRegression
        Local model (warm_start=True recommended).
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        Test labels.
    client_id : str
        Identifier used for logging.
    local_epochs : int
        Number of local training epochs per round.
    """

    def __init__(
        self,
        model: LogisticRegression,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        client_id: str,
        local_epochs: int = 1,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id
        self.local_epochs = local_epochs

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return local model parameters as a list of numpy arrays."""
        logger.debug("Client %s: get_parameters called", self.client_id)
        return get_model_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Update local model with global params, train, return updated params.

        Parameters
        ----------
        parameters : list of np.ndarray
            Global model weights broadcast by the server.
        config : dict
            Round-specific config (e.g., ``local_epochs``).

        Returns
        -------
        Tuple of (updated_parameters, num_train_samples, metrics_dict)
        """
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        set_model_parameters(self.model, parameters)

        for _ in range(local_epochs):
            self.model.fit(self.X_train, self.y_train)

        updated_params = get_model_parameters(self.model)
        y_pred = self.model.predict(self.X_train)
        train_accuracy = float(accuracy_score(self.y_train, y_pred))

        logger.info(
            "Client %s | fit round — samples=%d, train_acc=%.4f",
            self.client_id,
            len(self.y_train),
            train_accuracy,
        )

        return updated_params, len(self.y_train), {"train_accuracy": train_accuracy}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate global model parameters on local test data.

        Parameters
        ----------
        parameters : list of np.ndarray
            Global model weights from the server.
        config : dict
            Evaluation configuration.

        Returns
        -------
        Tuple of (loss, num_test_samples, metrics_dict)
        """
        set_model_parameters(self.model, parameters)

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = float(accuracy_score(self.y_test, y_pred))
        f1 = float(f1_score(self.y_test, y_pred, zero_division=0))
        loss = float(log_loss(self.y_test, y_prob))

        try:
            auc = float(roc_auc_score(self.y_test, y_prob))
        except ValueError:
            auc = 0.0

        logger.info(
            "Client %s | evaluate — loss=%.4f, acc=%.4f, f1=%.4f, auc=%.4f",
            self.client_id,
            loss,
            accuracy,
            f1,
            auc,
        )

        return loss, len(self.y_test), {
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": auc,
        }


# ---------------------------------------------------------------------------
# Client entry-point (distributed mode)
# ---------------------------------------------------------------------------


def start_client(client_id: int, server_address: str) -> None:
    """Create and start a Flower client in distributed mode.

    Parameters
    ----------
    client_id : int
        Which client shard to load.
    server_address : str
        gRPC address of the Flower server.
    """
    from src.config import LOCAL_EPOCHS, FEATURE_COLUMNS
    from src.data_utils import load_client_data, prepare_client_tensors
    from src.models import initialise_model

    df = load_client_data(client_id)
    X_train, y_train, X_test, y_test, _ = prepare_client_tensors(df)

    n_features = len([c for c in FEATURE_COLUMNS if c in df.columns])
    model = initialise_model(n_features)

    client = DiabetesClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        client_id=str(client_id),
        local_epochs=LOCAL_EPOCHS,
    )

    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    import sys
    from src.config import SERVER_ADDRESS

    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(cid, SERVER_ADDRESS)
