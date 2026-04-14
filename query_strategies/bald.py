"""
BALD (Bayesian Active Learning by Disagreement) baseline.
Uses MC-Dropout to estimate mutual information I[y; w | x].

BALD = H[y|x] - E_w[H[y|x,w]]
     = H(mean_prediction) - mean(H(each_prediction))

For segmentation: computed pixel-wise then mean-reduced per image.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .base_strategy import BaseStrategy


def _enable_dropout(model):
    """Enable dropout layers in eval mode for MC-Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.train()


class BALDSampling(BaseStrategy):
    """
    BALD via MC-Dropout on the final exit.
    T stochastic forward passes → estimate mutual information.
    """

    def __init__(self, model, idxs_lb, idxs_unlb, cfg, device):
        super().__init__(model, idxs_lb, idxs_unlb, cfg, device)
        self.T = cfg.get('bald', {}).get('mc_passes', 10)
        self.dropout_p = cfg.get('bald', {}).get('dropout_p', 0.5)
        # Inject dropout into the final head's ASPP if not already present
        self._inject_dropout()

    def _inject_dropout(self):
        """Add a Dropout2d layer before the final Conv2d in exit heads and final head."""
        import torch.nn as nn
        for module in self.model.modules():
            if isinstance(module, nn.Sequential):
                children = list(module.children())
                if children and isinstance(children[-1], nn.Conv2d):
                    # Check if dropout already exists
                    has_dropout = any(isinstance(c, (nn.Dropout, nn.Dropout2d)) for c in children)
                    if not has_dropout:
                        children.insert(-1, nn.Dropout2d(p=self.dropout_p))
                        module._modules.clear()
                        for i, c in enumerate(children):
                            module.add_module(str(i), c)

    @torch.no_grad()
    def score_unlabeled(self, dataset) -> np.ndarray:
        self.model.eval()
        _enable_dropout(self.model)

        subset = Subset(dataset, self.idxs_unlb)
        loader = DataLoader(subset, batch_size=1, shuffle=False,
                            num_workers=self.cfg['training'].get('num_workers', 4),
                            pin_memory=True)
        scores = []

        for batch in tqdm(loader, desc='[BALD] scoring', leave=False):
            img = batch[0].to(self.device, dtype=torch.float32)
            mc_probs = []
            for _ in range(self.T):
                out = self.model(img)
                probs = F.softmax(out['final_logits'], dim=1)  # [1, C, H, W]
                mc_probs.append(probs.unsqueeze(0))            # [1, 1, C, H, W]
            mc_probs = torch.cat(mc_probs, dim=0)              # [T, 1, C, H, W]

            mean_probs = mc_probs.mean(0)                       # [1, C, H, W]
            # H[E[p]]
            H_mean = -(mean_probs * (mean_probs + 1e-12).log()).sum(dim=1)  # [1, H, W]
            # E[H[p]]
            H_each = -(mc_probs * (mc_probs + 1e-12).log()).sum(dim=2)     # [T, 1, H, W]
            E_H = H_each.mean(0)                                            # [1, H, W]

            bald_score = (H_mean - E_H).mean().item()
            scores.append(max(bald_score, 0.0))

        return np.array(scores)
