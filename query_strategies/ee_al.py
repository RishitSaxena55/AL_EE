"""
OUR NOVEL METHOD – Full EE-AL: Uncertainty × Diversity (UxD) Combined Strategy.

Algorithm: Uncertainty-Weighted k-Center Greedy
================================================
Step 1 (U): Run EE Spatial Disagreement → uncertainty score u_i per unlabeled image (single pass)
Step 2 (D): Extract Mask-Guided CoreSet embeddings → foreground-aware feature vectors
Step 3 (UxD): Uncertainty-Weighted k-Center Greedy:
    - For each candidate, effective_distance = distance_to_nearest_labeled_center * u_i
    - Greedy selection picks argmax over effective_distance iteratively
    - This biases selection toward images that are (1) far from labeled set AND (2) uncertain

Why this is novel and effective:
  - Single forward pass for U (unlike BALD's T passes)
  - Uncertainty from architectural disagreement (EE-specific signal, not pixel entropy)
  - Diversity mask focuses on object regions (not background noise)
  - UxD combination avoids: picking uncertain-but-redundant examples (all-entropy suffers)
    and diverse-but-easy examples (all-coreset suffers)
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .base_strategy import BaseStrategy


class EEActiveLearning(BaseStrategy):
    """
    Our main proposed method: EE-AL (UxD).
    Combines EE-Uncertainty and Mask-Guided CoreSet via uncertainty-weighted
    k-center greedy selection.
    """

    BG_CLASS = 0  # PASCAL VOC

    @torch.no_grad()
    def _compute_scores_and_embeddings(self, dataset):
        """
        Single pass over ALL images (labeled + unlabeled) to compute:
          - uncertainty scores for unlabeled images [N_unlb]
          - mask-guided embeddings for all images [N_total, D]
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=self.cfg['training'].get('num_workers', 4),
                            pin_memory=True)

        all_embeds = []
        unlb_set = set(self.idxs_unlb.tolist())
        # Map global idx → position in idxs_unlb for uncertainty assignment
        unlb_idx_to_pos = {int(v): i for i, v in enumerate(self.idxs_unlb)}
        uncertainty_scores = np.zeros(len(self.idxs_unlb))

        for batch in tqdm(loader, desc='[EE-AL] computing U and D', leave=False):
            img = batch[0].to(self.device, dtype=torch.float32)
            global_idx = batch[4].item() if len(batch) > 4 else -1

            out = self.model(img, return_features=True)

            # ── Uncertainty (U): EE Spatial Disagreement ─────────────────────
            if global_idx in unlb_set:
                final_logits = out['final_logits']
                exit_logits_list = out['exit_logits']
                final_log_probs = F.log_softmax(final_logits, dim=1)

                if exit_logits_list:
                    kl_sum = 0.0
                    for exit_logits in exit_logits_list:
                        if exit_logits.shape[-2:] != final_logits.shape[-2:]:
                            exit_logits = F.interpolate(exit_logits, size=final_logits.shape[-2:],
                                                        mode='bilinear', align_corners=False)
                        exit_probs = F.softmax(exit_logits, dim=1)
                        exit_log_probs = F.log_softmax(exit_logits, dim=1)
                        kl = (exit_probs * (exit_log_probs - final_log_probs)).sum(1).mean()
                        kl_sum += kl.item()
                    uncertainty_scores[unlb_idx_to_pos[global_idx]] = kl_sum / len(exit_logits_list)
                else:
                    # Fallback: pixel entropy
                    final_probs = F.softmax(final_logits, dim=1)
                    ent = -(final_probs * (final_probs + 1e-12).log()).sum(1).mean()
                    uncertainty_scores[unlb_idx_to_pos[global_idx]] = ent.item()

            # ── Diversity (D): Mask-Guided Embedding ─────────────────────────
            feat_gap = out['bottleneck_feat']                 # [1, 256]
            final_logits_d = out['final_logits']
            probs_d = F.softmax(final_logits_d, dim=1)
            fg_mask = 1.0 - probs_d[:, self.BG_CLASS, :, :]  # [1, H, W]
            mean_fg = fg_mask.mean(dim=(1, 2), keepdim=True).squeeze(-1).squeeze(-1)  # [1, 1]
            masked = feat_gap * mean_fg                        # [1, 256]
            masked = F.normalize(masked, p=2, dim=1)
            all_embeds.append(masked.cpu().numpy())

        all_embeds = np.concatenate(all_embeds, axis=0)       # [N_total, D]
        return uncertainty_scores, all_embeds

    def query(self, dataset, n_query: int) -> np.ndarray:
        """
        Uncertainty-Weighted k-Center Greedy:
          effective_dist[i] = min_dist_to_labeled_center[i] * uncertainty[i]
          Iteratively pick argmax(effective_dist), update distances, repeat.
        """
        uncertainty_scores, all_embeds = self._compute_scores_and_embeddings(dataset)

        lb_embeds = all_embeds[self.idxs_lb]      # [N_lb, D]
        unlb_embeds = all_embeds[self.idxs_unlb]  # [N_unlb, D]

        # Initialize min-distance to labeled set
        if lb_embeds.shape[0] == 0:
            min_dist = np.full(len(self.idxs_unlb), np.inf)
        else:
            # Vectorized L2 distance (CPU, manageable for typical AL pool sizes)
            diff = unlb_embeds[:, None, :] - lb_embeds[None, :, :]  # [N_unlb, N_lb, D]
            dist_matrix = np.linalg.norm(diff, axis=-1)              # [N_unlb, N_lb]
            min_dist = dist_matrix.min(axis=1)                       # [N_unlb]

        # Normalize uncertainty scores to [0, 1] for stable weighting
        u = uncertainty_scores
        u_range = u.max() - u.min()
        if u_range > 1e-8:
            u_norm = (u - u.min()) / u_range
        else:
            u_norm = np.ones_like(u)
        # Add small floor to prevent zero-weight (diversity alone when u≈0)
        u_weight = u_norm + 0.1

        chosen_rel = []
        _min_dist = min_dist.copy()

        for step in range(n_query):
            # Effective score: distance × uncertainty weight
            effective = _min_dist * u_weight
            # Exclude already chosen
            for idx in chosen_rel:
                effective[idx] = -np.inf
            idx = int(np.argmax(effective))
            chosen_rel.append(idx)
            # Update min distances based on newly chosen point
            new_dists = np.linalg.norm(unlb_embeds - unlb_embeds[idx], axis=1)
            _min_dist = np.minimum(_min_dist, new_dists)

        chosen_rel = np.array(chosen_rel)
        return self.idxs_unlb[chosen_rel]

    def score_unlabeled(self, dataset) -> np.ndarray:
        # Not used directly (query is overridden)
        return np.zeros(len(self.idxs_unlb))
