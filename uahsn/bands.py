"""ERB-spaced triangles and partition-of-unity gain interpolation."""

import numpy as np

from .config import Config


def erb_rate(frequency_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1 + 0.00437 * frequency_hz)


def inverse_erb_rate(rate: np.ndarray) -> np.ndarray:
    return (10 ** (rate / 21.4) - 1) / 0.00437


class Bands:
    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        hz = np.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1)
        self.centers_hz = inverse_erb_rate(np.linspace(0, erb_rate(np.array(config.sample_rate / 2)), config.n_bands))
        centers = erb_rate(self.centers_hz)
        points = erb_rate(hz)
        weights = np.zeros((len(hz), config.n_bands))
        for b in range(config.n_bands):
            if b == 0:
                triangle = (centers[1] - points) / (centers[1] - centers[0])
            elif b == config.n_bands - 1:
                triangle = (points - centers[-2]) / (centers[-1] - centers[-2])
            else:
                rise = (points - centers[b - 1]) / (centers[b] - centers[b - 1])
                fall = (centers[b + 1] - points) / (centers[b + 1] - centers[b])
                triangle = np.minimum(rise, fall)
            weights[:, b] = np.clip(triangle, 0, 1)
        self.weights = weights / weights.sum(axis=1, keepdims=True)
        self.analysis_weights = self.weights / np.maximum(self.weights.sum(axis=0, keepdims=True), 1e-12)

    def energy(self, power: np.ndarray) -> np.ndarray:
        return np.asarray(power) @ self.analysis_weights

    def interpolate(self, gains: np.ndarray) -> np.ndarray:
        return np.asarray(gains) @ self.weights.T
