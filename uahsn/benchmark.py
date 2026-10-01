"""Wall-clock benchmark including feature extraction and reconstruction."""

import numpy as np

from .baselines import Enhancer
from .config import Config


def benchmark(method: Enhancer, audio: np.ndarray, config: Config = Config()) -> dict[str, float]:
    result = method.enhance(audio)
    return {"runtime_seconds": result.runtime_seconds,
            "real_time_factor": result.real_time_factor}
