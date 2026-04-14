"""
Standard Entropy Sampling Baseline.
Scores each unlabeled image by mean pixel-wise entropy of the FINAL exit softmax.
Images with high entropy = the model is uncertain everywhere → query them.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .base_strategy import BaseStrategy


class EntropySampling(BaseStrategy):
    """
    Entropy-based query strategy (standard baseline for segmentation AL).
    Score = mean pixel entropy of p(y|x) at the final exit.
    """

    @torch.no_grad()
    def score_unlabeled(self, dataset) -> np.ndarray:
        self.model.eval()
        subset = Subset(dataset, self.idxs_unlb)
        loader = DataLoader(subset, batch_size=1, shuffle=False,
                            num_workers=self.cfg['training'].get('num_workers', 4),
                            pin_memory=True)
        scores = []
        for batch in tqdm(loader, desc='[Entropy] scoring', leave=False):
            img = batch[0].to(self.device, dtype=torch.float32)
            out = self.model(img)
            logits = out['final_logits']              # [1, C, H, W]
            probs = F.softmax(logits, dim=1)          # [1, C, H, W]
            # pixel-wise entropy: -sum(p * log p)
            ent = -(probs * (probs + 1e-12).log()).sum(dim=1)  # [1, H, W]
            scores.append(ent.mean().item())
        return np.array(scores)
