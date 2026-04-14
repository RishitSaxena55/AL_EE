"""
Logging and result saving utilities.
Saves:
  - Per-round mIoU + selected indices to CSV
  - Full JSON summary
  - TensorBoard (optional)
"""

import os
import csv
import json
import numpy as np
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


class ResultLogger:
    """
    Logs AL experiment results per round.
    Creates:
      {save_dir}/
        round_results.csv     – one row per round
        summary.json          – final summary
        selected_indices/     – .npy file of selected indices per round
        tensorboard/          – TensorBoard events (if available)
    """

    CSV_FIELDS = [
        'round', 'n_labeled', 'n_unlabeled',
        'final_miou', 'timestamp', 'strategy',
        'avg_train_loss',
    ]

    def __init__(self, save_dir: str, strategy_name: str, exp_name: str = ''):
        self.save_dir = save_dir
        self.strategy_name = strategy_name
        self.exp_name = exp_name
        self.rows = []

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'selected_indices'), exist_ok=True)

        self.csv_path = os.path.join(save_dir, 'round_results.csv')
        self.json_path = os.path.join(save_dir, 'summary.json')

        # Write CSV header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()

        if HAS_TB:
            tb_dir = os.path.join(save_dir, 'tensorboard')
            self.writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.writer = None

        print(f"[Logger] Results will be saved to: {save_dir}")

    def log_round(self, rd: int, n_labeled: int, n_unlabeled: int,
                  final_miou: float, avg_train_loss: float,
                  selected_idxs: np.ndarray = None,
                  all_exit_mious: dict = None):
        row = {
            'round': rd,
            'n_labeled': n_labeled,
            'n_unlabeled': n_unlabeled,
            'final_miou': round(final_miou, 6),
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'avg_train_loss': round(avg_train_loss, 6),
        }
        self.rows.append(row)

        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writerow(row)

        # Save selected indices
        if selected_idxs is not None:
            idx_path = os.path.join(self.save_dir, 'selected_indices', f'round_{rd}.npy')
            np.save(idx_path, selected_idxs)

        # TensorBoard
        if self.writer:
            self.writer.add_scalar('mIoU/final', final_miou, rd)
            self.writer.add_scalar('labeled_count', n_labeled, rd)
            if all_exit_mious:
                for k, v in all_exit_mious.items():
                    self.writer.add_scalar(f'mIoU/{k}', v, rd)

        print(f"  [Round {rd}] mIoU={final_miou:.4f} | labeled={n_labeled}")

    def save_summary(self, extra: dict = None):
        summary = {
            'strategy': self.strategy_name,
            'exp_name': self.exp_name,
            'rounds': self.rows,
            'best_miou': max((r['final_miou'] for r in self.rows), default=0),
            'miou_curve': [r['final_miou'] for r in self.rows],
            'labeled_curve': [r['n_labeled'] for r in self.rows],
        }
        if extra:
            summary.update(extra)
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[Logger] Summary saved → {self.json_path}")
        if self.writer:
            self.writer.close()
        return summary
