"""Classical speech enhancers and a clearly labeled reconstructed FIR baseline."""

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Config
from .features import FeatureExtractor
from .minstats import MinimumStatistics
from .stft import istft, stft


@dataclass
class EnhancementResult:
    audio: np.ndarray
    gains: np.ndarray  # frames x positive-frequency bins; FIR uses |H|
    noise_psd: np.ndarray  # frames x positive-frequency bins
    band_gains: np.ndarray | None = None
    standard_deviation: np.ndarray | None = None
    fir_coefficients: np.ndarray | None = None
    runtime_seconds: float = 0.0
    real_time_factor: float = 0.0
    frame_times_ms: np.ndarray | None = None


def finish_result(result: EnhancementResult, samples: int, config: Config, start: float) -> EnhancementResult:
    result.runtime_seconds = perf_counter() - start
    duration = samples / config.sample_rate
    result.real_time_factor = result.runtime_seconds / duration if duration else 0.0
    return result


class Enhancer(Protocol):
    name: str

    def enhance(self, audio: np.ndarray) -> EnhancementResult: ...


def spectral_subtraction_gain(
    spectrum: np.ndarray, noise_psd: np.ndarray, oversubtraction: float, config: Config,
) -> np.ndarray:
    """Amplitude subtraction with noisy phase retained at reconstruction."""
    magnitude = np.abs(spectrum)
    gain = 1 - oversubtraction * np.sqrt(noise_psd) / np.maximum(magnitude, np.sqrt(config.epsilon))
    return np.clip(gain, config.gain_floor, 1.0)


class Noisy:
    name = "noisy"

    def __init__(self, config: Config = Config()) -> None:
        self.config = config

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        start = perf_counter()
        frames = stft(audio, self.config)
        estimator = MinimumStatistics(self.config)
        psds, times = [], []
        for frame in frames:
            tick = perf_counter()
            psds.append(estimator.update(np.abs(frame) ** 2))
            times.append((perf_counter() - tick) * 1000)
        result = EnhancementResult(np.asarray(audio, dtype=np.float32).copy(),
                                   np.ones_like(frames.real), np.asarray(psds),
                                   frame_times_ms=np.asarray(times))
        return finish_result(result, len(audio), self.config, start)


class SpectralSubtraction:
    name = "spectral_subtraction"

    def __init__(self, config: Config = Config(), oversubtraction: float = 1.0) -> None:
        if not np.isfinite(oversubtraction) or oversubtraction < 0:
            raise ValueError("oversubtraction must be finite and nonnegative")
        self.config = config
        self.oversubtraction = float(oversubtraction)

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        start = perf_counter()
        spectra = stft(audio, self.config)
        estimator = MinimumStatistics(self.config)
        gains, psds, times = [], [], []
        for spectrum in spectra:
            tick = perf_counter()
            psd = estimator.update(np.abs(spectrum) ** 2)
            psds.append(psd)
            gains.append(spectral_subtraction_gain(spectrum, psd, self.oversubtraction, self.config))
            times.append((perf_counter() - tick) * 1000)
        gain_matrix = np.asarray(gains)
        result = EnhancementResult(istft(spectra * gain_matrix, len(audio), self.config),
                                   gain_matrix, np.asarray(psds), frame_times_ms=np.asarray(times))
        return finish_result(result, len(audio), self.config, start)


class DecisionDirectedWiener:
    name = "minimum_statistics_dd_wiener"

    def __init__(self, config: Config = Config()) -> None:
        self.config = config

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        start = perf_counter()
        spectra = stft(audio, self.config)
        extractor = FeatureExtractor(self.config)
        gains, psds, times = [], [], []
        for spectrum in spectra:
            tick = perf_counter()
            _, psd, _, wiener = extractor.update(spectrum)
            psds.append(psd)
            gains.append(wiener)
            times.append((perf_counter() - tick) * 1000)
        gain_matrix = np.asarray(gains)
        result = EnhancementResult(istft(spectra * gain_matrix, len(audio), self.config),
                                   gain_matrix, np.asarray(psds), frame_times_ms=np.asarray(times))
        return finish_result(result, len(audio), self.config, start)


def symmetric_fir_from_gain(gain: np.ndarray, taps: int, n_fft: int) -> np.ndarray:
    """IDFT, circular symmetrization, Fejer-window truncation, causal delay.

    The triangular Fejer window keeps the zero-phase response inside the
    original gain range, since its frequency kernel is nonnegative.
    """
    if taps % 2 != 1 or taps < 3 or taps > n_fft:
        raise ValueError("Filter length must be odd, >= 3, and <= FFT size")
    impulse = np.fft.irfft(gain, n=n_fft)
    indices = np.arange(n_fft)
    symmetric = 0.5 * (impulse + impulse[(-indices) % n_fft])
    radius = (taps - 1) // 2
    # Index -radius..radius and move this centered window to causal 0..L-1.
    offsets = np.arange(-radius, radius + 1)
    window = 1 - np.abs(offsets) / (radius + 1)
    coefficients = symmetric[offsets % n_fft] * window
    return coefficients


def overlap_add_fir_blocks(
    audio: np.ndarray, coefficients: np.ndarray, hop: int,
    block_times_ms: list[float] | None = None,
) -> np.ndarray:
    """Apply one causal FIR per input hop using FFT overlap-add."""
    samples = len(audio)
    n_blocks = int(np.ceil(samples / hop))
    if coefficients.ndim != 2 or len(coefficients) < n_blocks:
        raise ValueError("Expected one FIR coefficient vector per input block")
    taps = coefficients.shape[1]
    fft_size = 1 << (hop + taps - 2).bit_length()
    padded = np.pad(np.asarray(audio, dtype=np.float64), (0, n_blocks * hop - samples))
    output = np.zeros(n_blocks * hop + taps - 1, dtype=np.float64)
    for m in range(n_blocks):
        tick = perf_counter()
        block = padded[m * hop:(m + 1) * hop]
        convolution = np.fft.irfft(
            np.fft.rfft(block, n=fft_size) * np.fft.rfft(coefficients[m], n=fft_size), n=fft_size,
        )[:hop + taps - 1]
        output[m * hop:m * hop + len(convolution)] += convolution
        if block_times_ms is not None:
            block_times_ms.append((perf_counter() - tick) * 1000)
    return output[:samples].astype(np.float32)


class ReconstructedLinearPhaseBaseline:
    """Reconstruction from a prose method description, not an exact paper reproduction.

    Each frame designs a causal Type-I symmetric FIR from the subtraction gain.
    The current hop is convolved with that FIR using FFT overlap-add. Filter
    changes occur at hop boundaries; its fixed group delay is (L-1)/2 samples.
    """

    name = "reconstructed_linear_phase_baseline"

    def __init__(
        self, config: Config = Config(), taps: int = 63, oversubtraction: float = 1.0,
    ) -> None:
        if config.win_length not in (256, 320, 512):
            raise ValueError("Baseline experiment frame length must be 256, 320, or 512")
        if taps % 2 != 1 or taps < 3 or taps > config.n_fft:
            raise ValueError("Filter length must be odd, >= 3, and <= FFT size")
        if not np.isfinite(oversubtraction) or oversubtraction < 0:
            raise ValueError("oversubtraction must be finite and nonnegative")
        self.config = config
        self.taps = taps
        self.oversubtraction = float(oversubtraction)

    def enhance(self, audio: np.ndarray) -> EnhancementResult:
        start = perf_counter()
        spectra = stft(audio, self.config)
        estimator = MinimumStatistics(self.config)
        hop = self.config.hop_length
        responses, coefficients_by_frame, psds, times = [], [], [], []
        for spectrum in spectra:
            tick = perf_counter()
            psd = estimator.update(np.abs(spectrum) ** 2)
            psds.append(psd)
            target = spectral_subtraction_gain(spectrum, psd, self.oversubtraction, self.config)
            coefficients = symmetric_fir_from_gain(target, self.taps, self.config.n_fft)
            coefficients_by_frame.append(coefficients)
            responses.append(np.abs(np.fft.rfft(coefficients, n=self.config.n_fft)))
            times.append((perf_counter() - tick) * 1000)
        coefficients_array = np.asarray(coefficients_by_frame)
        block_times: list[float] = []
        output = overlap_add_fir_blocks(audio, coefficients_array, hop, block_times)
        times[:len(block_times)] = (np.asarray(times[:len(block_times)])
                                   + np.asarray(block_times)).tolist()
        result = EnhancementResult(output, np.asarray(responses),
                                   np.asarray(psds), fir_coefficients=coefficients_array,
                                   frame_times_ms=np.asarray(times))
        return finish_result(result, len(audio), self.config, start)


# Retained as an import alias for existing callers; result.name uses the explicit label.
LinearPhaseSpectralSubtraction = ReconstructedLinearPhaseBaseline
