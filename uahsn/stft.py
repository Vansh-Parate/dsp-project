"""Causal STFT with a one-hop algorithmic reconstruction delay."""

import numpy as np

from .config import Config


def sqrt_hann(config: Config) -> np.ndarray:
    n = np.arange(config.win_length)
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * n / config.win_length))


def stft(audio: np.ndarray, config: Config = Config()) -> np.ndarray:
    """Frame m sees samples through the end of hop m, never future samples."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n_hops = int(np.ceil(len(x) / config.hop_length))
    padded = np.pad(x, (config.hop_length, (n_hops + 1) * config.hop_length - len(x)))
    window = sqrt_hann(config)
    return np.stack([
        np.fft.rfft(padded[m * config.hop_length:m * config.hop_length + config.win_length] * window,
                     n=config.n_fft)
        for m in range(n_hops + 1)
    ])


def istft(spectra: np.ndarray, length: int, config: Config = Config()) -> np.ndarray:
    frames = np.asarray(spectra)
    if frames.ndim != 2 or frames.shape[1] != config.n_fft // 2 + 1:
        raise ValueError("Expected frames x positive-frequency bins")
    output = np.zeros((len(frames) - 1) * config.hop_length + config.win_length)
    normalization = np.zeros_like(output)
    window = sqrt_hann(config)
    for m, spectrum in enumerate(frames):
        offset = m * config.hop_length
        frame = np.fft.irfft(spectrum, n=config.n_fft)[:config.win_length]
        output[offset:offset + config.win_length] += frame * window
        normalization[offset:offset + config.win_length] += window ** 2
    output /= np.maximum(normalization, 1e-12)
    return output[config.hop_length:config.hop_length + length].astype(np.float32)
