"""Wall-clock benchmark including feature extraction and reconstruction."""

from time import perf_counter

import numpy as np

from .baselines import Enhancer
from .config import Config


def benchmark(method: Enhancer, audio: np.ndarray, config: Config = Config()) -> dict[str, float]:
    start = perf_counter()
    method.enhance(audio)
    elapsed = perf_counter() - start
    duration = len(audio) / config.sample_rate
    return {"seconds": elapsed, "real_time_factor": elapsed / duration}
