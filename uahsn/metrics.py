"""Reference-based speech quality metrics at 16 kHz."""

import numpy as np
import warnings
from pesq import PesqError, pesq
from pystoi import stoi

from .config import Config


def _pair(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 1 or estimate.shape != reference.shape or not len(reference):
        raise ValueError("Reference and estimate must be nonempty equal-length mono signals")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError("Signals must contain only finite samples")
    return reference, estimate


def si_sdr(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-10) -> float:
    """Scale-invariant SDR from orthogonal projection onto zero-mean clean speech."""
    reference, estimate = _pair(reference, estimate)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    target = np.dot(estimate, reference) / (np.dot(reference, reference) + epsilon) * reference
    residual = estimate - target
    return float(10 * np.log10((np.dot(target, target) + epsilon) / (np.dot(residual, residual) + epsilon)))


def segmental_snr(
    reference: np.ndarray, estimate: np.ndarray, config: Config = Config(),
    lower_db: float = -10.0, upper_db: float = 35.0,
) -> float:
    """Mean of individually clipped 20 ms / 10 ms frame SNRs."""
    reference, estimate = _pair(reference, estimate)
    frame, hop = config.win_length, config.hop_length
    padded_length = max(frame, frame + int(np.ceil(max(0, len(reference) - frame) / hop)) * hop)
    clean = np.pad(reference, (0, padded_length - len(reference)))
    error = np.pad(reference - estimate, (0, padded_length - len(reference)))
    values = []
    for start in range(0, padded_length - frame + 1, hop):
        speech_power = np.sum(clean[start:start + frame] ** 2)
        error_power = np.sum(error[start:start + frame] ** 2)
        frame_snr = 10 * np.log10((speech_power + config.epsilon) / (error_power + config.epsilon))
        values.append(np.clip(frame_snr, lower_db, upper_db))
    return float(np.mean(values))


def stoi_score(reference: np.ndarray, estimate: np.ndarray, sample_rate: int = 16_000) -> float | None:
    reference, estimate = _pair(reference, estimate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        score = float(stoi(reference, estimate, sample_rate, extended=False))
    if any("Not enough STFT frames" in str(warning.message) for warning in caught):
        return None
    return score


def pesq_wideband(reference: np.ndarray, estimate: np.ndarray, sample_rate: int = 16_000) -> float | None:
    reference, estimate = _pair(reference, estimate)
    if sample_rate != 16_000:
        raise ValueError("Wideband PESQ requires 16 kHz input")
    try:
        return float(pesq(sample_rate, reference.astype(np.float32),
                          estimate.astype(np.float32), mode="wb"))
    except (PesqError, ValueError):
        # PESQ is undefined for some very short or otherwise unsupported clips.
        return None


def evaluate_pair(
    reference: np.ndarray, estimate: np.ndarray, config: Config = Config(),
) -> dict[str, float | None]:
    return {"si_sdr_db": si_sdr(reference, estimate),
            "segmental_snr_db": segmental_snr(reference, estimate, config),
            "stoi": stoi_score(reference, estimate, config.sample_rate),
            "pesq_wb": pesq_wideband(reference, estimate, config.sample_rate)}
