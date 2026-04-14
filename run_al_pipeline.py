#!/usr/bin/env python3
"""
EE-AL Active Learning Pipeline for Semantic Segmentation
=========================================================
Main entry point. Orchestrates the full AL loop:
  1. Initial training on seed labeled set
  2. For each round:
     a. Query strategy selects K images
     b. "Oracle" reveals labels (simulated via ground-truth)
     c. Model retrained on expanded labeled pool
     d. Evaluate mIoU on val set
     e. Save results

Usage:
  # Run our novel EE-AL method:
  python run_al_pipeline.py --config configs/pascal_voc.yaml --strategy ee_al

  # Run entropy baseline:
  python run_al_pipeline.py --config configs/pascal_voc.yaml --strategy entropy

  # Dry-run (no real data needed — uses synthetic data):
  python run_al_pipeline.py --config configs/pascal_voc.yaml --strategy ee_al --dry-run

  # Resume from checkpoint:
  python run_al_pipeline.py --config configs/pascal_voc.yaml --strategy ee_al \
      --resume results/ee_al/checkpoints/round_2.pth --start-round 3
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import random
from pathlib import Path

# ─── Make sure the pipeline root is in PYTHONPATH ───────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from models import build_multi_exit_model
from data.pascal_voc_dataset import VOCDataSet, VOCGTDataSet, DryRunDataset
from query_strategies import build_strategy, STRATEGY_REGISTRY
from training import Trainer, Evaluator
from utils import ResultLogger


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='EE-AL: Early-Exit Active Learning')
    p.add_argument('--config', type=str, default='configs/pascal_voc.yaml',
                   help='Path to YAML config file')
    p.add_argument('--strategy', type=str, default='ee_al',
                   choices=list(STRATEGY_REGISTRY.keys()),
                   help='AL query strategy to use')
    p.add_argument('--dry-run', action='store_true',
                   help='Use synthetic data (no real VOC dataset required)')
    p.add_argument('--n-rounds', type=int, default=None,
                   help='Override number of AL rounds from config')
    p.add_argument('--initial-budget', type=float, default=None,
                   help='Override initial labeled budget fraction')
    p.add_argument('--query-budget', type=float, default=None,
                   help='Override query budget fraction per round')
    p.add_argument('--seed', type=int, default=None, help='Override random seed')
    p.add_argument('--gpu', type=int, default=0, help='GPU device index')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--start-round', type=int, default=0,
                   help='Round index to start from (used with --resume)')
    p.add_argument('--exp-name', type=str, default=None,
                   help='Experiment name suffix for results directory')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override epochs per round')
    p.add_argument('--dry-run-size', type=int, default=300,
                   help='Number of synthetic images in dry-run mode')
    return p.parse_args()


# ─── Seeding ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    al = cfg['active_learning']
    if args.n_rounds is not None:       al['n_rounds'] = args.n_rounds
    if args.initial_budget is not None: al['initial_budget'] = args.initial_budget
    if args.query_budget is not None:   al['query_budget'] = args.query_budget
    if args.seed is not None:           al['seed'] = args.seed
    if args.epochs is not None:         cfg['training']['epochs_per_round'] = args.epochs

    strategy_name = args.strategy
    seed = al.get('seed', 42)
    set_seed(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
        print('[WARNING] CUDA not available – running on CPU (slow!)')
    print(f'[EE-AL] Device: {device}')

    # ── Results directory ────────────────────────────────────────────────────
    exp_suffix = f'_{args.exp_name}' if args.exp_name else ''
    exp_tag = 'dryrun_' if args.dry_run else ''
    # Note: argparse converts --dry-run flag to args.dry_run automatically
    results_dir = os.path.join(
        cfg['output'].get('results_dir', './results'),
        f'{exp_tag}{strategy_name}{exp_suffix}'
    )
    checkpoint_dir = os.path.join(
        cfg['output'].get('checkpoint_dir', './checkpoints'),
        f'{exp_tag}{strategy_name}{exp_suffix}'
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = ResultLogger(results_dir, strategy_name=strategy_name,
                          exp_name=f'{exp_tag}{strategy_name}{exp_suffix}')

    # ── Dataset ──────────────────────────────────────────────────────────────
    crop_size = tuple(cfg['dataset']['input_size'])
    num_classes = cfg['dataset']['num_classes']
    ignore_label = cfg['dataset'].get('ignore_label', 255)

    if args.dry_run:
        N = args.dry_run_size
        print(f'[EE-AL] DRY-RUN MODE: using {N} synthetic images')
        train_dataset = DryRunDataset(length=N, crop_size=crop_size,
                                      num_classes=num_classes)
        val_dataset = DryRunDataset(length=50, crop_size=crop_size,
                                    num_classes=num_classes)
        n_total = N
    else:
        data_dir = cfg['dataset']['data_dir']
        train_list = cfg['dataset']['train_list']
        val_list = cfg['dataset']['val_list']

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Dataset not found at '{data_dir}'. "
                "Please run scripts/download_voc.sh or set data_dir in the config."
            )
        train_dataset = VOCDataSet(data_dir, train_list, crop_size=crop_size)
        val_dataset = VOCGTDataSet(data_dir, val_list, crop_size=(505, 505))
        n_total = len(train_dataset)

    print(f'[EE-AL] Train pool: {n_total} | Val: {len(val_dataset)}')

    # ── Initial labeled / unlabeled split ────────────────────────────────────
    all_idxs = np.arange(n_total)
    np.random.shuffle(all_idxs)

    n_initial = max(1, int(n_total * al['initial_budget']))
    n_query = max(1, int(n_total * al['query_budget']))
    idxs_lb = all_idxs[:n_initial].copy()
    idxs_unlb = all_idxs[n_initial:].copy()
    print(f'[EE-AL] Initial labeled: {len(idxs_lb)} | Unlabeled: {len(idxs_unlb)}')
    print(f'[EE-AL] Query per round: {n_query} | Rounds: {al["n_rounds"]}')

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = cfg['model']
    model_cfg['num_classes'] = num_classes
    model = build_multi_exit_model(model_cfg)
    model = model.to(device)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_round = args.start_round
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'[EE-AL] Resumed from {args.resume} (round {ckpt["round"]})')
        start_round = ckpt['round'] + 1

    # ── Trainer & Evaluator ───────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device, checkpoint_dir)
    evaluator = Evaluator(num_classes, ignore_label=ignore_label, device=device)

    # ── Strategy (initialized with seed labeled set) ──────────────────────────
    strategy = build_strategy(strategy_name, model, idxs_lb, idxs_unlb, cfg, device)

    print(f'\n{"="*60}')
    print(f'  EE-AL Experiment')
    print(f'  Strategy : {strategy_name}')
    print(f'  Dataset  : {cfg["dataset"]["name"]} ({"DRY-RUN" if args.dry_run else "real"})')
    print(f'  Rounds   : {al["n_rounds"]}')
    print(f'  Exits    : {model.num_ee} (distribution={model_cfg.get("ee_distribution", "fine")})')
    print(f'{"="*60}\n')

    # ── AL Loop ───────────────────────────────────────────────────────────────
    for rd in range(start_round, al['n_rounds'] + 1):
        print(f'\n── Round {rd} / {al["n_rounds"]} '
              f'| Labeled: {len(strategy.idxs_lb)} '
              f'| Unlabeled: {len(strategy.idxs_unlb)} ──')

        # Step 1: Train on current labeled pool
        train_stats = trainer.train_round(train_dataset, strategy.idxs_lb, rd)

        # Step 2: Evaluate on val set
        print(f'  Evaluating...')
        val_result = evaluator.evaluate(model, val_dataset,
                                        batch_size=1,
                                        num_workers=cfg['training'].get('num_workers', 4))
        final_miou = val_result['miou']

        # Also report per-exit mIoU (for analysis)
        if rd == al['n_rounds'] or rd == 0:
            exit_mious = evaluator.evaluate_all_exits(model, val_dataset)
        else:
            exit_mious = None

        # Log results
        logger.log_round(
            rd=rd,
            n_labeled=len(strategy.idxs_lb),
            n_unlabeled=len(strategy.idxs_unlb),
            final_miou=final_miou,
            avg_train_loss=train_stats['avg_loss'],
            selected_idxs=strategy.idxs_lb[-n_query:] if rd > 0 else strategy.idxs_lb,
            all_exit_mious=exit_mious,
        )

        # Step 3: Query (select next batch to label) — skip on last round
        if rd < al['n_rounds']:
            print(f'  Querying {n_query} samples via {strategy_name}...')
            new_idxs = strategy.query(train_dataset, n_query)
            strategy.update(new_idxs)
            print(f'  → Selected {len(new_idxs)} samples. '
                  f'New labeled pool size: {len(strategy.idxs_lb)}')

    # ── Final summary ─────────────────────────────────────────────────────────
    summary = logger.save_summary({
        'config': cfg,
        'strategy': strategy_name,
        'n_exits': model.num_ee,
        'exit_positions': model.exit_positions,
        'exit_cost_ratios': model.exit_cost_ratios,
    })

    print(f'\n{"="*60}')
    print(f'  Experiment Complete!')
    print(f'  Best mIoU: {summary["best_miou"]:.4f}')
    print(f'  mIoU curve: {[f"{v:.4f}" for v in summary["miou_curve"]]}')
    print(f'  Results → {results_dir}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
