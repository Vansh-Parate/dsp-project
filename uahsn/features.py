"""96-dimensional causal features and classical decision-directed state."""

import numpy as np

from .bands import Bands
from .config import Config
from .minstats import MinimumStatistics


class FeatureExtractor:
    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.bands = Bands(config)
        self.noise = MinimumStatistics(config)
        self.previous_wiener = np.ones(config.n_fft // 2 + 1)
        self.previous_posterior = np.ones(config.n_fft // 2 + 1)

    def update(self, spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        power = np.abs(spectrum) ** 2
        noise_psd = self.noise.update(power)
        gamma = power / noise_psd
        xi = (self.config.dd_beta * self.previous_wiener ** 2 * self.previous_posterior
              + (1 - self.config.dd_beta) * np.maximum(gamma - 1, 0))
        wiener = xi / (1 + xi)
        self.previous_wiener = wiener
        self.previous_posterior = gamma
        band_energy = self.bands.energy(power)
        band_noise = np.maximum(self.bands.energy(noise_psd), self.config.epsilon)
        band_gamma = band_energy / band_noise
        band_xi = self.bands.energy(xi * noise_psd) / band_noise
        features = np.stack((
            np.log10(band_energy + self.config.epsilon),
            10 * np.log10(band_gamma + self.config.epsilon),
            10 * np.log10(band_xi + self.config.epsilon),
        ), axis=-1).reshape(-1)
        return features.astype(np.float32), noise_psd, gamma, wiener


def extract_features(spectra: np.ndarray, config: Config = Config()) -> np.ndarray:
    extractor = FeatureExtractor(config)
    return np.stack([extractor.update(frame)[0] for frame in spectra])
