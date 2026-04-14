"""
BADGE (Batch Active Learning by Diverse Gradient Embeddings) baseline.
Kim et al., ICLR 2021.

For segmentation: we use the gradient of the cross-entropy loss w.r.t.
the final linear layer weights as embedding, then run k-means++ seeding.

Approximation: since true per-pixel gradients are huge, we use a
gradient embedding approach on pooled features (following the approach
in the original paper's image-classification approximation extended to
segmentation via spatial mean-pooling).
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
from .base_strategy import BaseStrategy


class BADGESampling(BaseStrategy):
    """
    BADGE: use gradient w.r.t. last-layer weights as embeddings,
    then select a diverse batch via k-means++ seeding.
    """

    @torch.no_grad()
    def _get_grad_embeddings(self, dataset) -> np.ndarray:
        """
        Returns grad embeddings [N_unlb, D] where D = num_classes * bottleneck_dim.
        """
        self.model.eval()
        subset = Subset(dataset, self.idxs_unlb)
        loader = DataLoader(subset, batch_size=1, shuffle=False,
                            num_workers=self.cfg['training'].get('num_workers', 4),
                            pin_memory=True)
        embeddings = []

        # Identify final classifier weight for gradient embedding
        # We use the final head's last Conv2d as the "linear layer"
        # and bottleneck ASPP features as the "penultimate rep"

        for batch in tqdm(loader, desc='[BADGE] embeddings', leave=False):
            img = batch[0].to(self.device, dtype=torch.float32)
            out = self.model(img, return_features=True)
            feat = out['bottleneck_feat']          # [1, 256]
            logits = out['final_logits']           # [1, C, H, W]

            # Predicted class per pixel → pseudo-label
            probs = F.softmax(logits, dim=1)       # [1, C, H, W]
            pred_cls = probs.mean(dim=(2, 3))      # [1, C] (image-level class prob)
            pred_label = pred_cls.argmax(dim=1)    # [1]

            # Gradient of cross-entropy w.r.t. feature ([B, C*D])
            C = probs.shape[1]
            one_hot = torch.zeros_like(pred_cls)
            one_hot.scatter_(1, pred_label.unsqueeze(1), 1.0)
            # ∂L/∂feat ≈ (p - one_hot) ⊗ feat  (outer product)
            grad_weight = (pred_cls - one_hot).cpu().numpy()  # [1, C]
            feat_np = feat.cpu().numpy()                       # [1, D]
            # Outer product as embedding: [1, C*D]
            grad_embed = (grad_weight.T @ feat_np).flatten()  # [C*D]
            embeddings.append(grad_embed)

        return np.array(embeddings)   # [N_unlb, C*D]

    def query(self, dataset, n_query: int) -> np.ndarray:
        """k-means++ seeding on gradient embeddings."""
        embeddings = self._get_grad_embeddings(dataset)
        chosen = self._kmeans_pp(embeddings, n_query)
        return self.idxs_unlb[chosen]

    def score_unlabeled(self, dataset) -> np.ndarray:
        # Not used (query is overridden)
        return np.zeros(len(self.idxs_unlb))

    def _kmeans_pp(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        k-means++ seeding: greedily select k points that are far from each other.
        Returns indices into X.
        """
        n = X.shape[0]
        chosen = []
        # Random first center
        first = np.random.randint(n)
        chosen.append(first)
        # Squared distances to nearest center
        min_sq_dists = np.sum((X - X[first]) ** 2, axis=1)
        for _ in range(1, k):
            # Sample proportional to min squared distance
            probs = min_sq_dists / (min_sq_dists.sum() + 1e-12)
            idx = np.random.choice(n, p=probs)
            chosen.append(idx)
            new_dists = np.sum((X - X[idx]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, new_dists)
        return np.array(chosen)
