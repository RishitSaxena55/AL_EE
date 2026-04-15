#!/usr/bin/env python3
"""
Comprehensive plotting script for AL experiment results.
Analyzes and visualizes BALD, EE-AL, Entropy, and Random strategies.

Usage:
    python plot_results.py  [--output-dir results/figures] [--dpi 300]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
from scipy import interpolate

# ─── Configuration ───────────────────────────────────────────────────────────

RESULTS_DIR = Path("./results")
STRATEGIES = ['ee_al', 'bald', 'entropy', 'random']
STRATEGY_LABELS = {
    'ee_al':    'EE-AL (Ours)',
    'bald':     'BALD',
    'entropy':  'Entropy',
    'random':   'Random',
}
COLORS = {
    'ee_al':    '#E74C3C',  # Red
    'bald':     '#3498DB',  # Blue
    'entropy':  '#2ECC71',  # Green
    'random':   '#95A5A6',  # Gray
}

# ─── Load Data ────────────────────────────────────────────────────────────────

def load_results(strategies=STRATEGIES):
    """Load round_results.csv for each strategy."""
    data = {}
    for strategy in strategies:
        csv_path = RESULTS_DIR / strategy / "round_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data[strategy] = df
            print(f"✓ Loaded {strategy:15s} ({len(df)} rounds)")
        else:
            print(f"✗ Not found: {csv_path}")
    return data


def calculate_auc(df):
    """Calculate area under learning curve (normalized)."""
    x = df['n_labeled'].values
    y = df['final_miou'].values
    # Normalize by max mIoU to make it comparable
    if len(x) > 1:
        auc = np.trapz(y, x) / (x[-1] - x[0]) - y[0]
        return auc
    return 0


# ─── Plotting Functions ──────────────────────────────────────────────────────

def plot_learning_curves(data, output_dir, dpi=300):
    """Plot mIoU vs Number of Labeled Samples (main learning curve)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            ax.plot(df['n_labeled'], df['final_miou'], 
                   marker='o', linewidth=2.5, markersize=8,
                   label=STRATEGY_LABELS[strategy],
                   color=COLORS[strategy])
    
    ax.set_xlabel('Number of Labeled Samples', fontsize=13, fontweight='bold')
    ax.set_ylabel('mIoU Score', fontsize=13, fontweight='bold')
    ax.set_title('Active Learning Learning Curves\n(PASCAL VOC Augmented Split)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    
    # Add background shading for AL rounds
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    path = output_dir / "01_learning_curves.png"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_rounds(data, output_dir, dpi=300):
    """Plot mIoU vs AL Round number."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            ax.plot(df['round'], df['final_miou'], 
                   marker='s', linewidth=2.5, markersize=8,
                   label=STRATEGY_LABELS[strategy],
                   color=COLORS[strategy])
    
    ax.set_xlabel('Active Learning Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('mIoU Score', fontsize=13, fontweight='bold')
    ax.set_title('mIoU Performance per Active Learning Round', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.set_xticks(range(6))
    
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    path = output_dir / "02_rounds.png"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_training_loss(data, output_dir, dpi=300):
    """Plot average training loss vs round."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            ax.plot(df['round'], df['avg_train_loss'], 
                   marker='D', linewidth=2.5, markersize=7,
                   label=STRATEGY_LABELS[strategy],
                   color=COLORS[strategy], alpha=0.8)
    
    ax.set_xlabel('Active Learning Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Training Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Loss Progression Over AL Rounds', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.set_xticks(range(6))
    
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    path = output_dir / "03_training_loss.png"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_miou_improvement(data, output_dir, dpi=300):
    """Plot per-round mIoU improvement (delta)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            improvements = df['final_miou'].diff().fillna(0) * 100  # Convert to percentage points
            ax.plot(df['round'][1:], improvements[1:], 
                   marker='^', linewidth=2.5, markersize=8,
                   label=STRATEGY_LABELS[strategy],
                   color=COLORS[strategy])
    
    ax.set_xlabel('Active Learning Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('mIoU Improvement (percentage points)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Round mIoU Improvement', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.set_xticks(range(1, 6))
    
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    path = output_dir / "04_per_round_improvement.png"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_auc_comparison(data, output_dir, dpi=300):
    """Bar plot comparing AUC (area under learning curve)."""
    aucs = {}
    for strategy in STRATEGIES:
        if strategy in data:
            aucs[strategy] = calculate_auc(data[strategy])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies_list = list(aucs.keys())
    auc_values = list(aucs.values())
    colors_list = [COLORS[s] for s in strategies_list]
    labels_list = [STRATEGY_LABELS[s] for s in strategies_list]
    
    bars = ax.bar(labels_list, auc_values, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, auc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('AUC (Normalized)', fontsize=13, fontweight='bold')
    ax.set_title('Area Under Learning Curve (AUC) Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    path = output_dir / "05_auc_comparison.png"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_sample_efficiency(data, output_dir, dpi=300):
    """Plot samples needed to reach performance milestones."""
    milestones = [0.55, 0.60, 0.65, 0.67, 0.68]  # mIoU thresholds
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    milestone_data = {m: {} for m in milestones}
    
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            for milestone in milestones:
                # Find first round where performance >= milestone
                mask = df['final_miou'] >= milestone
                if mask.any():
                    n_samples = df[mask]['n_labeled'].iloc[0]
                    milestone_data[milestone][strategy] = n_samples
    
    x_pos = np.arange(len(milestones))
    bar_width = 0.2
    
    for i, strategy in enumerate(STRATEGIES):
        if strategy in data:
            values = [milestone_data[m].get(strategy, np.nan) for m in milestones]
            ax.bar(x_pos + i*bar_width, values, bar_width,
                  label=STRATEGY_LABELS[strategy], color=COLORS[strategy], alpha=0.8,
                  edgecolor='black', linewidth=1)
    
    ax.set_xlabel('mIoU Performance Target', fontsize=13, fontweight='bold')
    ax.set_ylabel('Labeled Samples Required', fontsize=13, fontweight='bold')
    ax.set_title('Sample Efficiency: Samples Needed to Reach Performance Targets', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + bar_width * 1.5)
    ax.set_xticklabels([f'{m:.2f}' for m in milestones])
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    path = output_dir / "06_sample_efficiency.png"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_final_performance(data, output_dir, dpi=300):
    """Bar plot of final mIoU scores."""
    final_scores = {}
    for strategy in STRATEGIES:
        if strategy in data:
            final_scores[strategy] = data[strategy]['final_miou'].iloc[-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies_list = list(final_scores.keys())
    score_values = list(final_scores.values())
    colors_list = [COLORS[s] for s in strategies_list]
    labels_list = [STRATEGY_LABELS[s] for s in strategies_list]
    
    bars = ax.bar(labels_list, score_values, color=colors_list, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, score_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Final mIoU Score', fontsize=13, fontweight='bold')
    ax.set_title('Final Performance Comparison (After All AL Rounds)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0.6, 0.7] if max(score_values) < 0.7 else None)
    
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    path = output_dir / "07_final_performance.png"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_combined_metrics(data, output_dir, dpi=300):
    """Create a comprehensive 2x2 subplot figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Learning Curves
    ax = axes[0, 0]
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            ax.plot(df['n_labeled'], df['final_miou'], 
                   marker='o', linewidth=2, markersize=6,
                   label=STRATEGY_LABELS[strategy], color=COLORS[strategy])
    ax.set_xlabel('Number of Labeled Samples', fontsize=11, fontweight='bold')
    ax.set_ylabel('mIoU Score', fontsize=11, fontweight='bold')
    ax.set_title('(A) Learning Curves', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9)
    ax.set_facecolor('#F8F9FA')
    
    # 2. Per-Round Improvement
    ax = axes[0, 1]
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            improvements = df['final_miou'].diff().fillna(0) * 100
            ax.plot(df['round'][1:], improvements[1:], 
                   marker='^', linewidth=2, markersize=6,
                   label=STRATEGY_LABELS[strategy], color=COLORS[strategy])
    ax.set_xlabel('AL Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('mIoU Improvement (pp)', fontsize=11, fontweight='bold')
    ax.set_title('(B) Per-Round Improvement', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_facecolor('#F8F9FA')
    
    # 3. Training Loss
    ax = axes[1, 0]
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            ax.plot(df['round'], df['avg_train_loss'], 
                   marker='s', linewidth=2, markersize=6,
                   label=STRATEGY_LABELS[strategy], color=COLORS[strategy], alpha=0.8)
    ax.set_xlabel('AL Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Avg Training Loss', fontsize=11, fontweight='bold')
    ax.set_title('(C) Training Loss Progression', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9)
    ax.set_facecolor('#F8F9FA')
    
    # 4. AUC Comparison
    ax = axes[1, 1]
    aucs = {}
    for strategy in STRATEGIES:
        if strategy in data:
            aucs[strategy] = calculate_auc(data[strategy])
    
    strategies_list = list(aucs.keys())
    auc_values = list(aucs.values())
    colors_list = [COLORS[s] for s in strategies_list]
    labels_list = [STRATEGY_LABELS[s] for s in strategies_list]
    
    bars = ax.bar(labels_list, auc_values, color=colors_list, alpha=0.8,
                  edgecolor='black', linewidth=1)
    for bar, val in zip(bars, auc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11, fontweight='bold')
    ax.set_title('(D) Area Under Curve', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor('#F8F9FA')
    
    fig.suptitle('Comprehensive AL Strategy Comparison\n(PASCAL VOC Augmented Split)', 
                fontsize=15, fontweight='bold', y=0.995)
    fig.patch.set_facecolor('white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    path = output_dir / "08_combined_metrics.png"
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


# ─── Summary Statistics ──────────────────────────────────────────────────────

def print_summary(data):
    """Print summary statistics for each strategy."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for strategy in STRATEGIES:
        if strategy in data:
            df = data[strategy]
            print(f"\n{STRATEGY_LABELS[strategy]:20s}")
            print("-" * 70)
            print(f"  Rounds:              {len(df)}")
            print(f"  Initial mIoU:        {df['final_miou'].iloc[0]:.6f}")
            print(f"  Final mIoU:          {df['final_miou'].iloc[-1]:.6f}")
            print(f"  Total Improvement:   {(df['final_miou'].iloc[-1] - df['final_miou'].iloc[0]):.6f} ({(df['final_miou'].iloc[-1] - df['final_miou'].iloc[0])*100:.2f}pp)")
            print(f"  Avg per-round gain:  {df['final_miou'].diff().mean():.6f}")
            print(f"  Max per-round gain:  {df['final_miou'].diff().max():.6f}")
            print(f"  Avg Training Loss:   {df['avg_train_loss'].mean():.6f}")
            print(f"  Final Training Loss: {df['avg_train_loss'].iloc[-1]:.6f}")
            print(f"  AUC (normalized):    {calculate_auc(df):.6f}")
            
            # Find round with max gradient
            grad = df['final_miou'].diff()
            max_grad_round = grad.idxmax()
            print(f"  Best round:          Round {max_grad_round} (+{grad.max():.6f})")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot AL experiment results')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                       help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure seaborn/matplotlib
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    
    # Load data
    print("\n" + "="*70)
    print("LOADING RESULTS")
    print("="*70)
    data = load_results()
    
    if not data:
        print("✗ No data found!")
        return
    
    # Print summary
    print_summary(data)
    
    # Plot
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_learning_curves(data, output_dir, dpi=args.dpi)
    plot_rounds(data, output_dir, dpi=args.dpi)
    plot_training_loss(data, output_dir, dpi=args.dpi)
    plot_miou_improvement(data, output_dir, dpi=args.dpi)
    plot_auc_comparison(data, output_dir, dpi=args.dpi)
    plot_sample_efficiency(data, output_dir, dpi=args.dpi)
    plot_final_performance(data, output_dir, dpi=args.dpi)
    plot_combined_metrics(data, output_dir, dpi=args.dpi)
    
    print("\n" + "="*70)
    print(f"✓ All plots saved to: {output_dir.resolve()}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
