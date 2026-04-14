"""
OUR NOVEL METHOD – Step U: Early-Exit Spatial Disagreement Uncertainty.

Computes per-image uncertainty as mean pixel-wise KL-divergence between
each early-exit softmax distribution and the final-exit softmax distribution.

Key insight: A well-calibrated multi-exit network "agrees" on easy images
across all exits. Disagreement (high KL-div between early and final exits)
signals that the early representation is still uncertain → the image is hard.

This is computed in a SINGLE forward pass (no MC-Dropout needed),
making it efficient at query time.

Score(x) = (1/N_exits) * sum_i [ mean_pixels( KL( softmax(exit_i) || softmax(final) ) ) ]
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .base_strategy import BaseStrategy


class EEUncertainty(BaseStrategy):
    """
    Early-Exit Spatial Disagreement uncertainty scorer (Step U of EE-AL).
    """

    @torch.no_grad()
    def score_unlabeled(self, dataset) -> np.ndarray:
        self.model.eval()
        subset = Subset(dataset, self.idxs_unlb)
        loader = DataLoader(subset, batch_size=1, shuffle=False,
                            num_workers=self.cfg['training'].get('num_workers', 4),
                            pin_memory=True)
        scores = []

        for batch in tqdm(loader, desc='[EE-U] scoring', leave=False):
            img = batch[0].to(self.device, dtype=torch.float32)
            out = self.model(img)     # dict

            final_logits = out['final_logits']       # [1, C, H, W]
            exit_logits_list = out['exit_logits']    # list of [1, C, H, W]

            final_probs = F.softmax(final_logits, dim=1)                  # [1, C, H, W]
            final_log_probs = F.log_softmax(final_logits, dim=1)

            if len(exit_logits_list) == 0:
                # Fallback: plain entropy if no exits
                ent = -(final_probs * (final_probs + 1e-12).log()).sum(1).mean()
                scores.append(ent.item())
                continue

            kl_sum = 0.0
            n_exits = len(exit_logits_list)
            for exit_logits in exit_logits_list:
                # Upsample exit to final resolution if needed
                if exit_logits.shape[-2:] != final_logits.shape[-2:]:
                    exit_logits = F.interpolate(exit_logits, size=final_logits.shape[-2:],
                                                mode='bilinear', align_corners=False)
                exit_probs = F.softmax(exit_logits, dim=1)  # [1, C, H, W]
                # KL( exit || final ): sum over classes, mean over pixels
                # F.kl_div expects log-probs as input, probs as target
                exit_log_probs = F.log_softmax(exit_logits, dim=1)
                # KL(P||Q) = sum P * (log P - log Q) = sum P * log P - sum P * log Q
                kl = (exit_probs * (exit_log_probs - final_log_probs)).sum(dim=1)  # [1, H, W]
                kl_sum += kl.mean().item()

            scores.append(kl_sum / n_exits)

        return np.array(scores)
