"""Audio I/O, random noise excerpts, and power-exact SNR mixing."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def load_audio(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    if sr != sample_rate:
        divisor = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, sr // divisor)
    if not np.isfinite(audio).all():
        raise ValueError(f"Invalid resampled audio: {path}")
    return np.asarray(audio, dtype=np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int = 16_000) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")


def random_noise_segment(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Select a uniform start; wrap circularly only when the recording is short."""
    if len(noise) == 0 or length <= 0:
        raise ValueError("Noise and requested segment must be nonempty")
    for _ in range(32):
        if len(noise) >= length:
            start = int(rng.integers(0, len(noise) - length + 1))
            segment = noise[start:start + length]
        else:
            start = int(rng.integers(0, len(noise)))
            segment = noise[(start + np.arange(length)) % len(noise)]
        if np.mean(np.asarray(segment, dtype=np.float64) ** 2) > 1e-12:
            return np.asarray(segment, dtype=np.float32)
    # Long recordings can contain mostly silence. Include the strongest sample.
    center = int(np.argmax(np.abs(noise)))
    start = min(max(0, center - length // 2), max(0, len(noise) - length))
    segment = noise[start:start + length] if len(noise) >= length else noise[(start + np.arange(length)) % len(noise)]
    if np.mean(np.asarray(segment, dtype=np.float64) ** 2) <= 1e-12:
        raise ValueError("Noise has no usable non-silent segment")
    return np.asarray(segment, dtype=np.float32)


def mix_components(
    clean: np.ndarray, noise: np.ndarray, snr_db: float, peak_limit: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned clean, scaled noise, mixture with one common peak scale."""
    if len(clean) != len(noise) or not len(clean):
        raise ValueError("Clean and noise must have equal nonzero length")
    speech = np.asarray(clean, dtype=np.float64)
    disturbance = np.asarray(noise, dtype=np.float64)
    ps = np.mean(speech ** 2)
    pv = np.mean(disturbance ** 2)
    if not np.isfinite(ps) or not np.isfinite(pv) or ps <= 0 or pv <= 0:
        raise ValueError("Clean and noise need finite positive power")
    scale = np.sqrt(ps / (pv * 10 ** (snr_db / 10)))
    scaled_noise = scale * disturbance
    mixture = speech + scaled_noise
    peak = max(np.max(np.abs(speech)), np.max(np.abs(scaled_noise)), np.max(np.abs(mixture)))
    common = min(1.0, peak_limit / peak)
    return (np.asarray(common * speech, dtype=np.float32),
            np.asarray(common * scaled_noise, dtype=np.float32),
            np.asarray(common * mixture, dtype=np.float32))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / max(len(noise), 1))))
    noise = np.asarray(noise[: len(clean)], dtype=np.float32)
    return mix_components(clean, noise, snr_db)[2]
