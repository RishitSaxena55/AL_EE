"""Random sampling baseline."""

import numpy as np
from .base_strategy import BaseStrategy


class RandomSampling(BaseStrategy):
    """Uniformly random selection from unlabeled pool."""

    def score_unlabeled(self, dataset) -> np.ndarray:
        return np.random.rand(len(self.idxs_unlb))
