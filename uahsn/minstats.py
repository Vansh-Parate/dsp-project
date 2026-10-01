"""Causal smoothed-periodogram minimum-statistics noise PSD."""

from collections import deque

import numpy as np

from .config import Config


class MinimumStatistics:
    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.smoothed: np.ndarray | None = None
        self.current_min: np.ndarray | None = None
        self.completed: deque[np.ndarray] = deque(maxlen=config.minstat_subwindows - 1)
        self.frame_in_subwindow = 0

    def update(self, power: np.ndarray) -> np.ndarray:
        power = np.asarray(power, dtype=np.float64)
        if self.smoothed is None:
            self.smoothed = power.copy()
        else:
            a = self.config.psd_alpha
            self.smoothed = a * self.smoothed + (1 - a) * power
        self.current_min = self.smoothed.copy() if self.current_min is None else np.minimum(self.current_min, self.smoothed)
        candidates = list(self.completed) + [self.current_min]
        estimate = np.minimum.reduce(candidates).copy()
        self.frame_in_subwindow += 1
        if self.frame_in_subwindow == self.config.minstat_frames:
            self.completed.append(self.current_min.copy())
            self.current_min = None
            self.frame_in_subwindow = 0
        return np.maximum(estimate, self.config.epsilon)
