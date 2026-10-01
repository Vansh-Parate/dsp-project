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


class UAHSNEnhancer:
    name = "uahsn"

    def __init__(self, model: UAHSN, sigma0: float, config: Config = Config()) -> None:
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError("sigma0 must be the positive validation median standard deviation")
        self.model = model.cpu().eval()
        self.sigma0 = float(sigma0)
        self.config = config
        self.bands = Bands(config)

    @classmethod
    def from_checkpoint(cls, path: str | Path, config: Config = Config()) -> "UAHSNEnhancer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = UAHSN()
        model.load_state_dict(checkpoint["model"])
        return cls(model, checkpoint["sigma0"], config)

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        start = perf_counter()
        spectra = stft(audio, self.config)
        extractor = FeatureExtractor(self.config)
        gains = []
        band_gains = []
        standard_deviations = []
        noise_psds = []
        state = None
        previous = None
        with torch.no_grad():
            for spectrum in spectra:
                feature, noise_psd, _, _ = extractor.update(spectrum)
                noise_psds.append(noise_psd)
                gain, log_variance, state = self.model(torch.from_numpy(feature)[None, None], state)
                predicted = gain[0, 0].numpy()
                sigma = np.exp(0.5 * np.clip(log_variance[0, 0].numpy(), -40, 40))
                u = sigma / (sigma + self.sigma0)
                alpha = self.config.alpha_min + (self.config.alpha_max - self.config.alpha_min) * u
                smoothed = predicted if previous is None else alpha * previous + (1 - alpha) * predicted
                previous = smoothed
                band_gains.append(smoothed)
                standard_deviations.append(sigma)
                gains.append(np.maximum(self.config.gain_floor, self.bands.interpolate(smoothed)))
        gains_array = np.asarray(gains)
        result = EnhancementResult(
            istft(spectra * gains_array, len(audio), self.config),
            gains_array,
            np.asarray(noise_psds),
            band_gains=np.asarray(band_gains),
            standard_deviation=np.asarray(standard_deviations),
        )
        return finish_result(result, len(audio), self.config, start)
