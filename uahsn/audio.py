"""Audio I/O and deterministic mixing."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def load_audio(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim != 1:
        raise ValueError(f"Single-channel audio required: {path}")
    if sr != sample_rate:
        divisor = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, sr // divisor)
    return np.asarray(audio, dtype=np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int = 16_000) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / max(len(noise), 1))))
    noise = np.asarray(noise[: len(clean)], dtype=np.float32)
    clean_power = np.mean(np.square(clean))
    noise_power = np.mean(np.square(noise))
    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10) + 1e-10))
    return np.asarray(clean + scale * noise, dtype=np.float32)
