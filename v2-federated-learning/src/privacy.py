"""Privacy utilities for federated learning.

Provides optional Gaussian differential privacy (DP) noise addition to
model gradients / weight updates before they are sent to the server.
This is a lightweight implementation for demonstration purposes.

For production DP guarantees, use TensorFlow Privacy or Opacus.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian mechanism
# ---------------------------------------------------------------------------


def compute_gaussian_sigma(
    epsilon: float,
    delta: float,
    sensitivity: float = 1.0,
) -> float:
    """Compute the Gaussian noise scale for (epsilon, delta)-DP.

    Uses the analytic Gaussian mechanism formula:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Parameters
    ----------
    epsilon : float
        Privacy budget (lower = stronger privacy, higher noise).
    delta : float
        Failure probability (typically 1e-5).
    sensitivity : float
        L2 sensitivity of the query (clipping norm).

    Returns
    -------
    float
        Standard deviation of the Gaussian noise to add.
    """
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    logger.debug(
        "DP sigma=%.4f for epsilon=%.2f, delta=%.2e, sensitivity=%.2f",
        sigma, epsilon, delta, sensitivity,
    )
    return sigma


def clip_and_add_noise(
    parameters: List[np.ndarray],
    clip_norm: float = 1.0,
    sigma: float = 0.0,
    random_state: int = 42,
) -> List[np.ndarray]:
    """Clip parameter updates by L2 norm and add Gaussian noise.

    Parameters
    ----------
    parameters : list of np.ndarray
        Model weight arrays to be privatised.
    clip_norm : float
        Maximum L2 norm of the update vector (clipping threshold).
    sigma : float
        Standard deviation of the Gaussian noise. 0 = no noise.
    random_state : int
        NumPy seed for reproducibility.

    Returns
    -------
    list of np.ndarray
        Clipped (and optionally noisy) parameter arrays.
    """
    rng = np.random.default_rng(random_state)

    # Flatten all parameters into one vector
    flat = np.concatenate([p.flatten() for p in parameters])

    # L2 clipping
    l2_norm = np.linalg.norm(flat)
    if l2_norm > clip_norm:
        scale = clip_norm / l2_norm
        flat = flat * scale
        logger.debug("Clipped update: l2_norm=%.4f → %.4f", l2_norm, clip_norm)

    # Gaussian noise
    if sigma > 0.0:
        noise = rng.normal(0, sigma, size=flat.shape)
        flat = flat + noise
        logger.debug("Added Gaussian noise (sigma=%.4f)", sigma)

    # Reconstruct original shapes
    result = []
    offset = 0
    for p in parameters:
        size = p.size
        result.append(flat[offset: offset + size].reshape(p.shape))
        offset += size

    return result


def privatise_parameters(
    parameters: List[np.ndarray],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    random_state: int = 42,
) -> Tuple[List[np.ndarray], float]:
    """Apply differential privacy to model parameters.

    Parameters
    ----------
    parameters : list of np.ndarray
    epsilon : float
        DP epsilon budget.
    delta : float
        DP delta.
    clip_norm : float
    random_state : int

    Returns
    -------
    Tuple of (noisy_parameters, sigma_used)
    """
    sigma = compute_gaussian_sigma(epsilon=epsilon, delta=delta, sensitivity=clip_norm)
    noisy = clip_and_add_noise(
        parameters, clip_norm=clip_norm, sigma=sigma, random_state=random_state
    )
    logger.info(
        "Applied DP: epsilon=%.2f, delta=%.2e, sigma=%.4f", epsilon, delta, sigma
    )
    return noisy, sigma


# ---------------------------------------------------------------------------
# Privacy analysis helpers
# ---------------------------------------------------------------------------


def estimate_privacy_budget_spent(
    num_rounds: int,
    n_clients: int,
    delta: float = 1e-5,
    sigma: float = 1.0,
    clip_norm: float = 1.0,
    n_samples: int = 1000,
    batch_size: int = 32,
) -> float:
    """Rough epsilon estimate using moments accountant approximation.

    This is a simplified estimate. For accurate privacy accounting use
    Google's privacy-accountant library.

    Returns
    -------
    float
        Approximate epsilon spent.
    """
    q = batch_size / n_samples  # sampling rate
    T = num_rounds  # number of compositions

    # Rough bound: epsilon ≈ q * T * sqrt(2 * log(1/delta)) / sigma
    epsilon_approx = q * T * math.sqrt(2 * math.log(1.0 / delta)) / sigma
    logger.info(
        "Approximate DP budget spent: epsilon≈%.4f (rounds=%d, sigma=%.2f)",
        epsilon_approx, T, sigma,
    )
    return epsilon_approx
