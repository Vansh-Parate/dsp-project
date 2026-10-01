"""Stateful causal UA-HSN enhancement."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .bands import Bands
from .baselines import EnhancementResult, finish_result
from .config import Config
from .features import FeatureExtractor
from .model import UAHSN
from .stft import istft, stft


def smooth_band_gains(
    predicted: np.ndarray, standard_deviation: np.ndarray,
    sigma0: float, config: Config = Config(),
) -> np.ndarray:
    """Apply the specified uncertainty-adaptive recursion across frames."""
    u = standard_deviation / (standard_deviation + sigma0)
    alpha = config.alpha_min + (config.alpha_max - config.alpha_min) * u
    smoothed = np.empty_like(predicted)
    smoothed[0] = predicted[0]
    for m in range(1, len(predicted)):
        smoothed[m] = alpha[m] * smoothed[m - 1] + (1 - alpha[m]) * predicted[m]
    return smoothed


class UAHSNEnhancer:
    name = "uahsn"

    def __init__(
        self, model: UAHSN, sigma0: float, config: Config = Config(),
        feature_mean: np.ndarray | None = None, feature_std: np.ndarray | None = None,
    ) -> None:
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError("sigma0 must be the positive validation median standard deviation")
        self.model = model.cpu().eval()
        self.sigma0 = float(sigma0)
        self.config = config
        self.bands = Bands(config)
        self.feature_mean = np.zeros(96, dtype=np.float32) if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.ones(96, dtype=np.float32) if feature_std is None else np.asarray(feature_std, dtype=np.float32)
        if self.feature_mean.shape != (96,) or self.feature_std.shape != (96,) or np.any(self.feature_std <= 0):
            raise ValueError("Feature normalization needs 96 means and positive standard deviations")

    @classmethod
    def from_checkpoint(cls, path: str | Path, config: Config = Config()) -> "UAHSNEnhancer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = UAHSN()
        model.load_state_dict(checkpoint["model"])
        return cls(model, checkpoint["sigma0"], config,
                   checkpoint.get("feature_mean"), checkpoint.get("feature_std"))

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        start = perf_counter()
        spectra = stft(audio, self.config)
        extractor = FeatureExtractor(self.config)
        predicted_gains = []
        standard_deviations = []
        noise_psds = []
        state = None
        with torch.no_grad():
            for spectrum in spectra:
                feature, noise_psd, _, _ = extractor.update(spectrum)
                noise_psds.append(noise_psd)
                normalized = (feature - self.feature_mean) / self.feature_std
                gain, log_variance, state = self.model(torch.from_numpy(normalized)[None, None], state)
                predicted = gain[0, 0].numpy()
                sigma = np.exp(0.5 * np.clip(log_variance[0, 0].numpy(), -40, 40))
                predicted_gains.append(predicted)
                standard_deviations.append(sigma)
        std_array = np.asarray(standard_deviations)
        band_gains = smooth_band_gains(np.asarray(predicted_gains), std_array, self.sigma0, self.config)
        gains_array = np.maximum(self.config.gain_floor, self.bands.interpolate(band_gains))
        result = EnhancementResult(
            istft(spectra * gains_array, len(audio), self.config),
            gains_array,
            np.asarray(noise_psds),
            band_gains=band_gains,
            standard_deviation=std_array,
        )
        return finish_result(result, len(audio), self.config, start)
