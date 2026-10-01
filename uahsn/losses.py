"""Specified band gain targets and three-term training objective."""

import torch

from .config import Config


def ideal_ratio_gain(clean_energy: torch.Tensor, noisy_energy: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    return torch.clamp(torch.sqrt(clean_energy / (noisy_energy + epsilon)), max=1.0)


def enhancement_loss(
    gain: torch.Tensor,
    log_variance: torch.Tensor,
    target_gain: torch.Tensor,
    noisy_magnitude: torch.Tensor,
    clean_magnitude: torch.Tensor,
    bin_weights: torch.Tensor,
    config: Config = Config(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Inputs have B,T,32 bands or B,T,257 bins; W has 257,32."""
    bin_gain = torch.clamp(gain @ bin_weights.T, min=config.gain_floor)
    enhanced = (bin_gain * noisy_magnitude + config.epsilon) ** config.compressed_exponent
    clean = (clean_magnitude + config.epsilon) ** config.compressed_exponent
    mag = torch.mean(torch.abs(enhanced - clean))
    variance = torch.exp(log_variance)
    nll = torch.mean(0.5 * (log_variance + (gain - target_gain) ** 2 / variance))
    adjacent_noise = (target_gain[:, 1:] < config.tc_tau) & (target_gain[:, :-1] < config.tc_tau)
    squared_change = (gain[:, 1:] - gain[:, :-1]) ** 2
    tc = (squared_change * adjacent_noise).sum() / adjacent_noise.sum().clamp_min(1)
    total = config.lambda_mag * mag + config.lambda_nll * nll + config.lambda_tc * tc
    return total, {"mag": mag, "nll": nll, "tc": tc}
