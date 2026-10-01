"""Classical enhancement methods with a shared result and interface."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import Config
from .features import FeatureExtractor
from .stft import istft, stft


@dataclass
class EnhancementResult:
    audio: np.ndarray
    gains: np.ndarray  # frames x 257, magnitude response
    band_gains: np.ndarray | None = None
    standard_deviation: np.ndarray | None = None


class Enhancer(Protocol):
    name: str

    def enhance(self, audio: np.ndarray) -> EnhancementResult: ...


class Noisy:
    name = "noisy"

    def __init__(self, config: Config = Config()) -> None:
        self.config = config

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        frames = stft(audio, self.config)
        return EnhancementResult(np.asarray(audio, dtype=np.float32).copy(), np.ones_like(frames.real))


class SpectralSubtraction:
    name = "spectral_subtraction"

    def __init__(self, config: Config = Config()) -> None:
        self.config = config

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        spectra = stft(audio, self.config)
        features = FeatureExtractor(self.config)
        gains = []
        for spectrum in spectra:
            _, noise_psd, _, _ = features.update(spectrum)
            gain = 1 - np.sqrt(noise_psd / (np.abs(spectrum) ** 2 + self.config.epsilon))
            gains.append(np.clip(gain, self.config.gain_floor, 1))
        gains_array = np.asarray(gains)
        return EnhancementResult(istft(spectra * gains_array, len(audio), self.config), gains_array)


class DecisionDirectedWiener:
    name = "minimum_statistics_dd_wiener"

    def __init__(self, config: Config = Config()) -> None:
        self.config = config

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        spectra = stft(audio, self.config)
        features = FeatureExtractor(self.config)
        gains = np.asarray([
            np.maximum(self.config.gain_floor, features.update(spectrum)[3]) for spectrum in spectra
        ])
        return EnhancementResult(istft(spectra * gains, len(audio), self.config), gains)


class LinearPhaseSpectralSubtraction:
    """Causal symmetric 63-tap FIR projected from each subtraction gain curve.

    Coefficients are frozen for each 160-sample output block. A symmetric FIR
    contributes a fixed 31-sample delay; no future input samples are accessed.
    """

    name = "linear_phase_spectral_subtraction"

    def __init__(self, config: Config = Config(), taps: int = 63) -> None:
        if taps % 2 != 1 or taps > config.n_fft:
            raise ValueError("taps must be odd and no larger than n_fft")
        self.config = config
        self.taps = taps

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        spectra = stft(audio, self.config)
        features = FeatureExtractor(self.config)
        hop = self.config.hop_length
        x = np.asarray(audio, dtype=np.float64)
        padded = np.pad(x, (self.taps - 1, int(np.ceil(len(x) / hop)) * hop - len(x)))
        output = np.zeros(int(np.ceil(len(x) / hop)) * hop)
        responses = []
        radius = self.taps // 2
        for m, spectrum in enumerate(spectra):
            _, noise_psd, _, _ = features.update(spectrum)
            target = np.clip(
                1 - np.sqrt(noise_psd / (np.abs(spectrum) ** 2 + self.config.epsilon)),
                self.config.gain_floor, 1,
            )
            zero_phase = np.fft.irfft(target, n=self.config.n_fft)
            coefficients = np.concatenate((zero_phase[-radius:], zero_phase[:radius + 1]))
            coefficients *= np.hanning(self.taps)
            response = np.fft.rfft(coefficients, n=self.config.n_fft)
            responses.append(np.abs(response))
            if m < len(spectra) - 1:
                start = m * hop
                for offset in range(hop):
                    index = start + offset + self.taps - 1
                    output[start + offset] = np.dot(coefficients, padded[index - self.taps + 1:index + 1][::-1])
        return EnhancementResult(output[:len(x)].astype(np.float32), np.asarray(responses))
