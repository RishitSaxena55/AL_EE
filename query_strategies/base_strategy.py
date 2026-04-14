"""Base Active Learning Strategy."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseStrategy(ABC):
    """
    Abstract base class for all AL query strategies.

    All strategies receive:
      - model: the multi-exit segmentation model
      - idxs_lb (np.ndarray): current labeled pool indices (into full train list)
      - idxs_unlb (np.ndarray): current unlabeled pool indices
      - cfg (dict): full config
      - device (torch.device)
    """

    def __init__(self, model, idxs_lb: np.ndarray, idxs_unlb: np.ndarray,
                 cfg: dict, device):
        self.model = model
        self.idxs_lb = np.array(idxs_lb)
        self.idxs_unlb = np.array(idxs_unlb)
        self.cfg = cfg
        self.device = device

    @abstractmethod
    def score_unlabeled(self, dataset) -> np.ndarray:
        """
        Compute a score for every sample in idxs_unlb.
        Higher score → higher priority for labeling.

        Returns:
            scores: np.ndarray of shape [len(idxs_unlb)]
        """

    def query(self, dataset, n_query: int) -> np.ndarray:
        """
        Default: rank by score descending, pick top n_query.
        Returns absolute dataset indices (from idxs_unlb).
        """
        scores = self.score_unlabeled(dataset)
        ranked = np.argsort(scores)[::-1]  # highest first
        chosen_relative = ranked[:n_query]
        return self.idxs_unlb[chosen_relative]

    def update(self, new_idxs: np.ndarray):
        """Move new_idxs from unlabeled to labeled pool."""
        self.idxs_lb = np.concatenate([self.idxs_lb, new_idxs])
        unlb_set = set(self.idxs_unlb.tolist()) - set(new_idxs.tolist())
        self.idxs_unlb = np.array(sorted(unlb_set))
