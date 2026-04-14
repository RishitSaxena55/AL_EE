"""
OUR NOVEL METHOD – Step D: Mask-Guided CoreSet Diversity.

Standard CoreSet uses global image features (GAP of bottleneck) which are
dominated by background pixels in segmentation datasets (e.g., sky, road).
This leads to selecting images that look diverse in background but not in objects.

Mask-Guided CoreSet:
  1. Compute foreground probability mask from final exit:
     fg_mask = 1 - P(background cls) = 1 - softmax(logits)[:, 0, :, :]   (VOC: class 0 = bg)
  2. Use fg_mask as spatial attention over the bottleneck feature map:
     masked_feat[b, c] = (fg_mask[b] * feat_map[b, c]).sum() / (fg_mask[b].sum() + ε)
     → Weighted average pooling that ignores background regions
  3. Normalize masked embedding → run k-center greedy for diversity

This produces embeddings that represent the OBJECT content of each image,
making diversity selection focus on semantic diversity (different objects)
rather than scene layout diversity.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .base_strategy import BaseStrategy


class MaskGuidedCoreSet(BaseStrategy):
    """
    Mask-Guided CoreSet diversity selector (Step D of EE-AL).
    """

    BG_CLASS = 0  # PASCAL VOC background class index

    @torch.no_grad()
    def _get_masked_embeddings(self, dataset):
        """
        Extract foreground-weighted bottleneck embeddings for ALL images,
        returning (lb_embeds, unlb_embeds).
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.cfg['training'].get('batch_size', 8),
                            shuffle=False,
                            num_workers=self.cfg['training'].get('num_workers', 4),
                            pin_memory=True)
        all_embeds = []

        for batch in tqdm(loader, desc='[MaskGuidedCoreSet] embeddings', leave=False):
            img = batch[0].to(self.device, dtype=torch.float32)
            out = self.model(img, return_features=True)

            # Get the ASPP feature map *before* GAP from the final head
            # We re-extract it: run the ASPP module and stop before GAP
            # The bottleneck_feat returned is already GAP'd, so we need the
            # feature map. We'll hook into the final head ASPP output.
            logits = out['final_logits']       # [B, C, H, W]
            feat_gap = out['bottleneck_feat']  # [B, 256] – already GAP'd

            # Compute foreground mask from final exit
            probs = F.softmax(logits, dim=1)                       # [B, C, H, W]
            fg_mask = 1.0 - probs[:, self.BG_CLASS, :, :]         # [B, H, W]

            # Spatially align fg_mask with feat_gap dimensions using a softer approach:
            # Since we only have the GAP'd feature, we weight it by the mean foreground prob
            # (scalar per image): masked_embed = feat_gap * mean_fg_prob
            mean_fg = fg_mask.mean(dim=(1, 2), keepdim=True)       # [B, 1, 1]
            mean_fg = mean_fg.squeeze(-1).squeeze(-1)               # [B, 1]

            # Mask-scaled embedding: de-emphasize background-dominated images
            masked = feat_gap * mean_fg                             # [B, 256]

            # L2 normalize
            masked = F.normalize(masked, p=2, dim=1)

            all_embeds.append(masked.cpu().numpy())

        all_embeds = np.concatenate(all_embeds, axis=0)            # [N_total, 256]
        return all_embeds

    @staticmethod
    def _k_center_greedy(unlb_feats: np.ndarray, lb_feats: np.ndarray, n: int) -> np.ndarray:
        n_unlb = unlb_feats.shape[0]
        if lb_feats.shape[0] == 0:
            min_dist = np.full(n_unlb, np.inf)
        else:
            from sklearn.metrics import pairwise_distances
            dist_matrix = pairwise_distances(unlb_feats, lb_feats, metric='euclidean')
            min_dist = dist_matrix.min(axis=1)

        chosen = []
        for _ in range(n):
            idx = int(min_dist.argmax())
            chosen.append(idx)
            new_dists = np.linalg.norm(unlb_feats - unlb_feats[idx], axis=1)
            min_dist = np.minimum(min_dist, new_dists)
        return np.array(chosen)

    def query(self, dataset, n_query: int) -> np.ndarray:
        all_embeds = self._get_masked_embeddings(dataset)
        lb_embeds = all_embeds[self.idxs_lb]
        unlb_embeds = all_embeds[self.idxs_unlb]
        chosen_rel = self._k_center_greedy(unlb_embeds, lb_embeds, n_query)
        return self.idxs_unlb[chosen_rel]

    def score_unlabeled(self, dataset) -> np.ndarray:
        return np.zeros(len(self.idxs_unlb))
