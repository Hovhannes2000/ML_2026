"""ALS с авторским знаковым целевым сигналом и затуханием давности."""

import numpy as np
from .ials import IALS


class CustomTargetALS(IALS):
    def __init__(self, half_life_days=730.0, **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def targets(self, frame, reference_time):
        if reference_time is None:
            raise ValueError("Нужна граница обучающего периода")
        age_days = (reference_time - frame.timestamp.to_numpy()) / 86400.0
        if np.any(age_days < 0):
            raise ValueError("Обучающие события не могут находиться в будущем")
        signed_rating = (frame.rating.to_numpy(dtype=float) - 3.0) / 2.0
        decay = np.exp2(-age_days / self.half_life_days)
        return signed_rating * decay
