"""Per-method summaries, bootstrap intervals, and paired Wilcoxon tests."""

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


METRICS = (
    "si_sdr_db", "segmental_snr_db", "stoi", "pesq_wb",
    "gain_variance", "residual_peak_density_per_second", "aliasing_ratio", "noise_reduction_db",
)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.number)) and np.isfinite(value)


def bootstrap_mean_interval(
    values: np.ndarray, rng: np.random.Generator, samples: int = 2000,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty set")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def summarize_results(
    records: list[dict[str, Any]], seed: int = 0, bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Pair methods by mixture_id, retaining missing metrics as null in records."""
    rng = np.random.default_rng(seed)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[str(record["method"])].append(record)
    summaries: dict[str, dict[str, Any]] = {}
    for method, method_records in sorted(by_method.items()):
        summaries[method] = {}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in method_records if _finite(row.get(metric))], dtype=float)
            if not len(values):
                summaries[method][metric] = {"n": 0, "mean": None, "std": None, "ci95": None}
                continue
            summaries[method][metric] = {
                "n": int(len(values)), "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95": list(bootstrap_mean_interval(values, rng, bootstrap_samples)),
            }
    proposed = {row["mixture_id"]: row for row in by_method.get("uahsn", [])}
    comparisons: dict[str, dict[str, Any]] = {}
    for baseline, baseline_records in sorted(by_method.items()):
        if baseline == "uahsn":
            continue
        comparisons[baseline] = {}
        for metric in METRICS:
            differences = [
                float(proposed[row["mixture_id"]][metric] - row[metric])
                for row in baseline_records
                if row["mixture_id"] in proposed
                and _finite(row.get(metric))
                and _finite(proposed[row["mixture_id"]].get(metric))
            ]
            nonzero = np.count_nonzero(np.asarray(differences))
            if len(differences) < 3 or nonzero < 3:
                comparisons[baseline][metric] = {"n_pairs": len(differences),
                                                 "statistic": None, "p_value": None}
                continue
            statistic, p_value = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
            comparisons[baseline][metric] = {"n_pairs": len(differences),
                                             "statistic": float(statistic), "p_value": float(p_value)}
    return {"summary": summaries, "paired_wilcoxon": comparisons,
            "bootstrap_samples": bootstrap_samples, "seed": seed}
