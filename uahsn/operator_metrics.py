"""Operator measures using clean/noise references and saved gain sequences."""

import numpy as np

from .baselines import EnhancementResult, overlap_add_fir_blocks
from .config import Config
from .stft import istft, stft


def noise_only_frames(
    clean: np.ndarray, config: Config = Config(), threshold_dbfs: float = -40.0,
) -> np.ndarray:
    """A frame is noise-only when its full clean-reference window is quiet."""
    signal = np.asarray(clean, dtype=np.float64)
    n_hops = int(np.ceil(len(signal) / config.hop_length))
    padded = np.pad(signal, (config.hop_length, (n_hops + 1) * config.hop_length - len(signal)))
    threshold_power = 10 ** (threshold_dbfs / 10)
    mask = np.asarray([
        np.mean(padded[m * config.hop_length:m * config.hop_length + config.win_length] ** 2)
        <= threshold_power for m in range(n_hops + 1)
    ])
    mask[-1] = False  # final analysis frame only flushes the input tail
    return mask


def gain_variance(gains: np.ndarray, noise_only: np.ndarray) -> float | None:
    """Mean squared gain change at time-frequency entries in noise-only frames."""
    gains = np.asarray(gains)
    if gains.ndim != 2 or len(gains) != len(noise_only):
        raise ValueError("Gain matrix and noise-only mask have incompatible shapes")
    eligible = noise_only[1:]
    if not eligible.any():
        return None
    return float(np.mean(np.diff(gains, axis=0)[eligible] ** 2))


def isolated_peak_count(
    power: np.ndarray, local_floor: np.ndarray, noise_only: np.ndarray,
    theta_db: float = 6.0,
) -> int:
    """Count strict 3x3 local maxima above neighbors and local noise floor."""
    if power.shape != local_floor.shape or power.ndim != 2 or len(noise_only) != len(power):
        raise ValueError("Power, floor, and frame mask have incompatible shapes")
    threshold = 10 ** (theta_db / 10)
    count = 0
    for m in range(1, len(power) - 1):
        if not noise_only[m]:
            continue
        for k in range(1, power.shape[1] - 1):
            center = power[m, k]
            if center <= local_floor[m, k]:
                continue
            neighbors = power[m - 1:m + 2, k - 1:k + 2].copy()
            neighbors[1, 1] = -np.inf
            if center >= threshold * np.max(neighbors) and center > 0:
                count += 1
    return count


def residual_peak_density(
    result: EnhancementResult, clean: np.ndarray, config: Config = Config(),
    theta_db: float = 6.0, threshold_dbfs: float = -40.0,
) -> float | None:
    mask = noise_only_frames(clean, config, threshold_dbfs)
    duration = float(mask.sum() * config.hop_length / config.sample_rate)
    if duration <= 0:
        return None
    residual_power = np.abs(stft(result.audio, config)) ** 2
    if residual_power.shape != result.gains.shape or result.noise_psd.shape != result.gains.shape:
        raise ValueError("Result gain/PSD dimensions do not match its STFT")
    local_floor = result.noise_psd * result.gains ** 2
    peaks = isolated_peak_count(residual_power, local_floor, mask, theta_db)
    return float(peaks / duration)


def aliasing_ratio(gains: np.ndarray, central_radius: int = 31) -> float:
    """Average circular IDFT energy outside lags [-radius, radius]."""
    gains = np.asarray(gains, dtype=np.float64)
    if gains.ndim != 2:
        raise ValueError("Expected frames x positive-frequency gain matrix")
    n_fft = (gains.shape[1] - 1) * 2
    if not 0 <= central_radius < n_fft // 2:
        raise ValueError("central_radius must be between zero and N/2 - 1")
    impulse = np.fft.irfft(gains, n=n_fft, axis=1)
    energy = impulse ** 2
    outside = energy.copy()
    outside[:, :central_radius + 1] = 0
    if central_radius:
        outside[:, -central_radius:] = 0
    total = energy.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Aliasing ratio is undefined for a zero-energy gain vector")
    return float(np.mean(outside.sum(axis=1) / total))


def shadow_filter_noise(
    noise: np.ndarray, result: EnhancementResult, config: Config = Config(),
) -> np.ndarray:
    """Pass known additive noise through the fixed operator used on the mixture."""
    if result.fir_coefficients is not None:
        return overlap_add_fir_blocks(noise, result.fir_coefficients, config.hop_length)
    noise_spectra = stft(noise, config)
    if noise_spectra.shape != result.gains.shape:
        raise ValueError("Noise STFT and saved gain sequence have incompatible shapes")
    return istft(noise_spectra * result.gains, len(noise), config)


def noise_reduction_db(
    clean: np.ndarray, mixture: np.ndarray, result: EnhancementResult,
    config: Config = Config(),
) -> float | None:
    clean = np.asarray(clean, dtype=np.float64)
    mixture = np.asarray(mixture, dtype=np.float64)
    if clean.shape != mixture.shape:
        raise ValueError("Clean reference and mixture need equal lengths")
    noise = mixture - clean
    input_power = np.sum(noise ** 2)
    if input_power <= 0:
        return None
    processed = shadow_filter_noise(noise, result, config)
    output_power = np.sum(np.asarray(processed, dtype=np.float64) ** 2)
    if output_power <= 0:
        return None
    return float(10 * np.log10(input_power / output_power))


def operator_metrics(
    result: EnhancementResult, clean: np.ndarray, mixture: np.ndarray,
    config: Config = Config(), threshold_dbfs: float = -40.0,
    theta_db: float = 6.0, central_radius: int = 31,
) -> dict[str, float | None]:
    mask = noise_only_frames(clean, config, threshold_dbfs)
    return {
        "gain_variance": gain_variance(result.gains, mask),
        "residual_peak_density_per_second": residual_peak_density(result, clean, config,
                                                                   theta_db, threshold_dbfs),
        "aliasing_ratio": aliasing_ratio(result.gains, central_radius),
        "noise_reduction_db": noise_reduction_db(clean, mixture, result, config),
    }
