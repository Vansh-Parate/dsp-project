"""Experiment constants. Defaults follow the UA-HSN specification."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16_000
    win_length: int = 320
    hop_length: int = 160
    n_fft: int = 512
    n_bands: int = 32
    gain_floor: float = 0.1
    psd_alpha: float = 0.9
    minstat_subwindows: int = 8
    minstat_frames: int = 12
    dd_beta: float = 0.98
    alpha_min: float = 0.2
    alpha_max: float = 0.9
    compressed_exponent: float = 0.3
    lambda_mag: float = 1.0
    lambda_nll: float = 0.1
    lambda_tc: float = 0.1
    tc_tau: float = 0.3
    epsilon: float = 1e-10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.win_length != 2 * self.hop_length or self.n_fft < self.win_length:
            raise ValueError("Expected 50% overlap and FFT size >= window length")


DEFAULT = Config()
