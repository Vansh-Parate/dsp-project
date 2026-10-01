"""Transparent objective metrics; no pretrained metric dependency."""

import numpy as np

from .config import Config
from .stft import stft


def si_sdr(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-10) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("Signals must have equal length")
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    target = np.dot(estimate, reference) / (np.dot(reference, reference) + epsilon) * reference
    residual = estimate - target
    return float(10 * np.log10((np.dot(target, target) + epsilon) / (np.dot(residual, residual) + epsilon)))


def log_spectral_distance(reference: np.ndarray, estimate: np.ndarray, config: Config = Config()) -> float:
    clean = np.abs(stft(reference, config))
    enhanced = np.abs(stft(estimate, config))
    distance = 20 * np.log10((clean + config.epsilon) / (enhanced + config.epsilon))
    return float(np.mean(np.sqrt(np.mean(distance ** 2, axis=1))))


def evaluate_pair(reference: np.ndarray, estimate: np.ndarray, config: Config = Config()) -> dict[str, float]:
    return {"si_sdr_db": si_sdr(reference, estimate),
            "log_spectral_distance_db": log_spectral_distance(reference, estimate, config)}
