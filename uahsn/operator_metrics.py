"""Gain-based operator measurements from saved enhancement results."""

import numpy as np

from .baselines import EnhancementResult


def operator_metrics(result: EnhancementResult) -> dict[str, float]:
    gain = np.asarray(result.gains)
    metrics = {
        "mean_gain": float(gain.mean()),
        "mean_absolute_gain_change": float(np.mean(np.abs(np.diff(gain, axis=0)))) if len(gain) > 1 else 0.0,
        "gain_below_0_1_fraction": float(np.mean(gain < 0.1 - 1e-6)),
    }
    if result.standard_deviation is not None:
        metrics["median_predicted_standard_deviation"] = float(np.median(result.standard_deviation))
    return metrics
