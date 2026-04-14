"""
Vanilla CoreSet baseline (k-center greedy on bottleneck embeddings).
Sener & Savarese, ICLR 2018.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .base_strategy import BaseStrategy


class CoreSetSampling(BaseStrategy):
    """
    Vanilla k-center greedy CoreSet using bottleneck ASPP features
    (GAP of ASPP output → [B, 256] vectors).
    """

    @torch.no_grad()
    def _get_embeddings(self, dataset) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.cfg['training'].get('batch_size', 8),
                            shuffle=False,
                            num_workers=self.cfg['training'].get('num_workers', 4),
                            pin_memory=True)
        all_feats = []
        for batch in tqdm(loader, desc='[CoreSet] embeddings', leave=False):
            img = batch[0].to(self.device, dtype=torch.float32)
            out = self.model(img, return_features=True)
            feat = out['bottleneck_feat']  # [B, 256]
            all_feats.append(feat.cpu().numpy())
        return np.concatenate(all_feats, axis=0)  # [N_total, D]

    @staticmethod
    def _k_center_greedy(unlb_feats: np.ndarray, lb_feats: np.ndarray, n: int) -> np.ndarray:
        """
        Greedy k-center: iteratively pick the unlabeled point with
        maximum distance to the nearest labeled center.
        """
        from sklearn.metrics import pairwise_distances

        n_unlb = unlb_feats.shape[0]
        if lb_feats.shape[0] == 0:
            min_dist = np.full(n_unlb, np.inf)
        else:
            dist_matrix = pairwise_distances(unlb_feats, lb_feats, metric='euclidean')
            min_dist = dist_matrix.min(axis=1)

        chosen = []
        for _ in range(n):
            idx = int(min_dist.argmax())
            chosen.append(idx)
            # Update min distances
            new_dists = np.linalg.norm(unlb_feats - unlb_feats[idx], axis=1)
            min_dist = np.minimum(min_dist, new_dists)
        return np.array(chosen)

    def query(self, dataset, n_query: int) -> np.ndarray:
        all_feats = self._get_embeddings(dataset)
        lb_feats = all_feats[self.idxs_lb]
        unlb_feats = all_feats[self.idxs_unlb]
        chosen_rel = self._k_center_greedy(unlb_feats, lb_feats, n_query)
        return self.idxs_unlb[chosen_rel]

    def score_unlabeled(self, dataset) -> np.ndarray:
        return np.zeros(len(self.idxs_unlb))
