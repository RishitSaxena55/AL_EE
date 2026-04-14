"""
Training loop for the multi-exit segmentation model.

Loss: Equal-weight cross-entropy over ALL exits + final head.
  L = (1/(N_exits+1)) * sum_i [ CE(exit_i, label) ] + CE(final, label)
  Ignore label = 255 (PASCAL VOC void).

Optimizer: SGD with poly-LR schedule (standard DeepLab practice).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class MultiExitCELoss(nn.Module):
    """
    Combined cross-entropy over all exits.
    Supports configurable per-exit weights (default: uniform).
    """

    def __init__(self, ignore_index: int = 255, exit_weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.exit_weights = exit_weights  # if None, equal weights

    def forward(self, model_output: dict, labels: torch.Tensor) -> dict:
        labels = labels.long()
        ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        exit_logits = model_output['exit_logits']
        final_logits = model_output['final_logits']

        total_exits = len(exit_logits) + 1  # exits + final
        weights = self.exit_weights if self.exit_weights else [1.0 / total_exits] * total_exits

        loss = torch.tensor(0.0, device=labels.device, requires_grad=True)

        exit_losses = []
        for i, e_logits in enumerate(exit_logits):
            l_resized = F.interpolate(e_logits, size=labels.shape[-2:],
                                      mode='bilinear', align_corners=False)
            el = ce(l_resized, labels)
            exit_losses.append(el.item())
            loss = loss + weights[i] * el

        final_resized = F.interpolate(final_logits, size=labels.shape[-2:],
                                      mode='bilinear', align_corners=False)
        final_loss = ce(final_resized, labels)
        loss = loss + weights[-1] * final_loss

        return {
            'total_loss': loss,
            'exit_losses': exit_losses,
            'final_loss': final_loss.item(),
        }


class Trainer:
    """
    Poly-LR SGD trainer for multi-exit segmentation.
    """

    def __init__(self, model, cfg: dict, device, checkpoint_dir: str):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.criterion = MultiExitCELoss(
            ignore_index=cfg['dataset'].get('ignore_label', 255)
        )

    def _poly_lr(self, base_lr: float, step: int, max_steps: int, power: float = 0.9) -> float:
        return base_lr * ((1 - step / max_steps) ** power)

    def _build_optimizer(self):
        tcfg = self.cfg['training']
        # Separate backbone (1x LR) and head (10x LR)
        backbone_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'layer' in name and 'exit' not in name and 'final_head' not in name:
                backbone_params.append(p)
            else:
                head_params.append(p)
        return optim.SGD(
            [
                {'params': backbone_params, 'lr': tcfg['lr']},
                {'params': head_params,     'lr': tcfg['lr'] * 10},
            ],
            momentum=tcfg.get('momentum', 0.9),
            weight_decay=tcfg.get('weight_decay', 1e-4),
        )

    def train_round(self, dataset, idxs_lb: np.ndarray, rd: int) -> dict:
        """
        Train the model on the current labeled pool for one AL round.
        Returns training stats dict.
        """
        tcfg = self.cfg['training']
        bs = tcfg.get('batch_size', 8)
        epochs = tcfg.get('epochs_per_round', 50)

        subset = Subset(dataset, idxs_lb)
        loader = DataLoader(subset, batch_size=bs, shuffle=True,
                            num_workers=tcfg.get('num_workers', 4),
                            pin_memory=True, drop_last=True)

        num_steps = epochs * len(loader)
        optimizer = self._build_optimizer()
        base_lr = tcfg['lr']

        self.model.train()
        self.model.to(self.device)

        step = 0
        total_loss_accum = 0.0
        log_interval = max(1, len(loader) // 5)

        for epoch in range(epochs):
            for batch in loader:
                # Poly LR update
                lr = self._poly_lr(base_lr, step, max(num_steps, 1))
                optimizer.param_groups[0]['lr'] = lr
                optimizer.param_groups[1]['lr'] = lr * 10

                imgs = batch[0].to(self.device, dtype=torch.float32)
                labels = batch[1].to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                out = self.model(imgs)
                loss_dict = self.criterion(out, labels)
                loss = loss_dict['total_loss']
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss_accum += loss.item()
                step += 1

            if (epoch + 1) % max(1, epochs // 5) == 0:
                avg_loss = total_loss_accum / max(step, 1)
                print(f"  [Round {rd}] Epoch {epoch+1}/{epochs} | "
                      f"LR={lr:.6f} | AvgLoss={avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, f'round_{rd}.pth')
        torch.save({
            'round': rd,
            'model_state_dict': self.model.state_dict(),
            'n_labeled': len(idxs_lb),
        }, ckpt_path)
        print(f"  [Round {rd}] Checkpoint saved → {ckpt_path}")

        return {
            'avg_loss': total_loss_accum / max(step, 1),
            'checkpoint': ckpt_path,
        }
