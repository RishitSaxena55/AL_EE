"""
mIoU Evaluator for multi-exit segmentation.
Evaluates the FINAL exit only (standard protocol).
Also reports per-exit mIoU for analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:

    def __init__(self, num_classes: int, ignore_label: int = 255, device='cuda'):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.device = device

    def _compute_iou(self, conf_matrix: np.ndarray) -> dict:
        """Compute per-class IoU and mIoU from confusion matrix."""
        iu = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            denom = (conf_matrix[c, :].sum() + conf_matrix[:, c].sum()
                     - conf_matrix[c, c])
            if denom > 0:
                iu[c] = conf_matrix[c, c] / denom
        valid = conf_matrix.sum(axis=1) > 0
        miou = iu[valid].mean()
        return {'miou': float(miou), 'per_class_iou': iu.tolist()}

    @torch.no_grad()
    def evaluate(self, model, val_dataset, batch_size: int = 1,
                 num_workers: int = 4, exit_idx: int = -1) -> dict:
        """
        Evaluate on val_dataset.
        exit_idx: -1 = final exit, 0..N-1 = early exits
        Returns dict with 'miou', 'per_class_iou', 'conf_matrix'.
        """
        model.eval()
        model.to(self.device)

        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        for batch in tqdm(loader, desc='[Eval] mIoU', leave=False):
            imgs = batch[0].to(self.device, dtype=torch.float32)
            labels = batch[1].numpy()     # [B, H, W] or [B, crop_h, crop_w]
            sizes = batch[2].numpy()      # original sizes

            out = model(imgs)
            if exit_idx == -1:
                logits = out['final_logits']
            else:
                logits = out['exit_logits'][exit_idx]

            # Upsample to label size
            logits = F.interpolate(logits, size=labels.shape[-2:],
                                   mode='bilinear', align_corners=False)
            preds = logits.argmax(dim=1).cpu().numpy()  # [B, H, W]

            for b in range(preds.shape[0]):
                pred_b = preds[b]
                label_b = labels[b].astype(np.int32)
                # Mask out ignore label
                mask = (label_b != self.ignore_label)
                pred_b = pred_b[mask]
                label_b = label_b[mask]
                # Accumulate confusion matrix
                idx = np.ravel_multi_index(
                    (label_b.clip(0, self.num_classes - 1),
                     pred_b.clip(0, self.num_classes - 1)),
                    (self.num_classes, self.num_classes)
                )
                conf_flat = np.bincount(idx, minlength=self.num_classes ** 2)
                conf += conf_flat.reshape(self.num_classes, self.num_classes)

        result = self._compute_iou(conf)
        result['conf_matrix'] = conf.tolist()
        return result

    @torch.no_grad()
    def evaluate_all_exits(self, model, val_dataset, batch_size=1, num_workers=4) -> dict:
        """Evaluate all exits and return dict with mIoU per exit."""
        results = {}
        n_exits = model.num_ee
        for i in range(n_exits):
            r = self.evaluate(model, val_dataset, batch_size, num_workers, exit_idx=i)
            results[f'exit_{i}'] = r['miou']
            print(f"    Exit {i} mIoU: {r['miou']:.4f}")
        r_final = self.evaluate(model, val_dataset, batch_size, num_workers, exit_idx=-1)
        results['final'] = r_final['miou']
        print(f"    Final mIoU:   {r_final['miou']:.4f}")
        return results
