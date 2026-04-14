#!/usr/bin/env python3
"""
EE-AL Results Analysis & Visualization
=======================================
Generates all relevant graphs comparing EE-AL (ours) vs 5 baselines:
  1. mIoU Learning Curve          (main result)
  2. mIoU Improvement Over Random  (Δ mIoU vs Random)
  3. Training Loss Curves
  4. Annotation Efficiency        (mIoU per labeled image)
  5. Per-Exit mIoU Breakdown      (from EE-AL final round)
  6. Area Under Curve (AUC) Summary Bar Chart

Usage:
  python analyze_results.py

Output: results/figures/  (PNG files + a single PDF report)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("./results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Strategy display names and colors
STRATEGIES = {
    "ee_al":   {"label": "EE-AL (Ours)",  "color": "#FF6B35", "lw": 2.5, "ls": "-",  "marker": "★", "zorder": 5},
    "badge":   {"label": "BADGE",         "color": "#2196F3", "lw": 1.8, "ls": "--", "marker": "s", "zorder": 4},
    "bald":    {"label": "BALD",          "color": "#9C27B0", "lw": 1.8, "ls": "--", "marker": "^", "zorder": 3},
    "coreset": {"label": "CoreSet",       "color": "#4CAF50", "lw": 1.8, "ls": "--", "marker": "D", "zorder": 3},
    "entropy": {"label": "Entropy",       "color": "#FF9800", "lw": 1.8, "ls": ":",  "marker": "o", "zorder": 2},
    "random":  {"label": "Random",        "color": "#607D8B", "lw": 1.5, "ls": ":",  "marker": "x", "zorder": 1},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Load data ─────────────────────────────────────────────────────────────────
def load_results():
    dfs = {}
    for strat in STRATEGIES:
        csv_path = RESULTS_DIR / strat / "round_results.csv"
        if csv_path.exists():
            dfs[strat] = pd.read_csv(csv_path)
        else:
            print(f"  [WARN] Missing: {csv_path}")
    return dfs


# ── Plot 1: mIoU Learning Curve ───────────────────────────────────────────────
def plot_miou_curve(dfs, ax=None, title_suffix=""):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    for strat, df in dfs.items():
        s = STRATEGIES[strat]
        x = df["n_labeled"].values
        y = df["final_miou"].values * 100
        ax.plot(x, y, color=s["color"], lw=s["lw"], ls=s["ls"],
                marker=s["marker"] if s["marker"] != "★" else "*",
                markersize=8 if s["marker"] == "★" else 6,
                label=s["label"], zorder=s["zorder"])

    ax.set_xlabel("# Labeled Images")
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"Active Learning Curve – PASCAL VOC{title_suffix}")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    if standalone:
        fig.tight_layout()
        out = FIGURES_DIR / "01_miou_curve.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
    return ax


# ── Plot 2: Δ mIoU Over Random ───────────────────────────────────────────────
def plot_delta_random(dfs, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    if "random" not in dfs:
        return ax

    random_miou = dfs["random"]["final_miou"].values
    for strat, df in dfs.items():
        if strat == "random":
            continue
        s = STRATEGIES[strat]
        x = df["n_labeled"].values
        diff = (df["final_miou"].values - random_miou[:len(df)]) * 100
        ax.plot(x, diff, color=s["color"], lw=s["lw"], ls=s["ls"],
                marker=s["marker"] if s["marker"] != "★" else "*",
                markersize=8 if s["marker"] == "★" else 6,
                label=s["label"], zorder=s["zorder"])

    ax.axhline(0, color="#607D8B", lw=1.5, ls=":", label="Random (baseline)")
    ax.set_xlabel("# Labeled Images")
    ax.set_ylabel("Δ mIoU vs Random (%)")
    ax.set_title("mIoU Gain Over Random Sampling")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    if standalone:
        fig.tight_layout()
        out = FIGURES_DIR / "02_delta_vs_random.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
    return ax


# ── Plot 3: Training Loss Curves ──────────────────────────────────────────────
def plot_loss_curves(dfs, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    for strat, df in dfs.items():
        if "avg_train_loss" not in df.columns:
            continue
        s = STRATEGIES[strat]
        ax.plot(df["n_labeled"].values, df["avg_train_loss"].values,
                color=s["color"], lw=s["lw"], ls=s["ls"],
                marker=s["marker"] if s["marker"] != "★" else "*",
                markersize=6, label=s["label"], zorder=s["zorder"])

    ax.set_xlabel("# Labeled Images")
    ax.set_ylabel("Avg Training Loss (CE)")
    ax.set_title("Training Loss by Strategy")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    if standalone:
        fig.tight_layout()
        out = FIGURES_DIR / "03_training_loss.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
    return ax


# ── Plot 4: Annotation Efficiency (mIoU / n_labeled) ─────────────────────────
def plot_annotation_efficiency(dfs, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    for strat, df in dfs.items():
        s = STRATEGIES[strat]
        x = df["n_labeled"].values
        efficiency = df["final_miou"].values / (x / 100)  # mIoU per 1% data
        ax.plot(x, efficiency, color=s["color"], lw=s["lw"], ls=s["ls"],
                marker=s["marker"] if s["marker"] != "★" else "*",
                markersize=8 if s["marker"] == "★" else 6,
                label=s["label"], zorder=s["zorder"])

    ax.set_xlabel("# Labeled Images")
    ax.set_ylabel("mIoU / (% of pool used)")
    ax.set_title("Annotation Efficiency (mIoU per % of labeled data)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    if standalone:
        fig.tight_layout()
        out = FIGURES_DIR / "04_annotation_efficiency.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
    return ax


# ── Plot 5: AUC Summary Bar Chart ─────────────────────────────────────────────
def plot_auc_bar(dfs, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    names, aucs, colors = [], [], []
    for strat in ["ee_al", "badge", "bald", "coreset", "entropy", "random"]:
        if strat not in dfs:
            continue
        df = dfs[strat]
        x = df["n_labeled"].values
        y = df["final_miou"].values * 100
        # Trapezoidal integration
        auc = np.trapz(y, x) / (x[-1] - x[0])
        names.append(STRATEGIES[strat]["label"])
        aucs.append(auc)
        colors.append(STRATEGIES[strat]["color"])

    bars = ax.bar(names, aucs, color=colors, edgecolor="white",
                  linewidth=1.5, zorder=2)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#222222")

    ax.set_ylabel("Avg mIoU (AUC-normalized, %)")
    ax.set_title("Area Under mIoU Curve (Average Performance)")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(aucs) * 1.15)

    if standalone:
        fig.tight_layout()
        out = FIGURES_DIR / "05_auc_bar.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
    return ax


# ── Plot 6: EE-AL Per-Exit Breakdown (from summary.json) ─────────────────────
def plot_exit_breakdown(ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    # Manually collected from the terminal output for all rounds
    exit_data = {
        "Round 0 (29 imgs)": [0.0, 0.0, 0.0, 0.0, 21.3],
        "Round 3 (116 imgs)": [7.5, 10.8, 11.6, 11.7, 44.9],
        "Round 5 (174 imgs)": [9.7, 13.7, 14.7, 14.9, 53.7],
    }
    labels = ["Exit 0\n(shallow)", "Exit 1", "Exit 2", "Exit 3\n(deep)", "Final\n(ASPP)"]
    x = np.arange(len(labels))
    width = 0.25

    palette = ["#BBDEFB", "#64B5F6", "#2196F3", "#FF8A65", "#FF6B35"]
    for i, (rnd, vals) in enumerate(exit_data.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=rnd,
                      color=palette[i*2] if i*2 < len(palette) else palette[-1],
                      edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mIoU (%)")
    ax.set_title("EE-AL: Per-Exit mIoU Across AL Rounds")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    if standalone:
        fig.tight_layout()
        out = FIGURES_DIR / "06_exit_breakdown.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
    return ax


# ── Plot 7: Big Dashboard (all plots together) ────────────────────────────────
def plot_dashboard(dfs):
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])   # spans 2 columns – main result
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    plot_miou_curve(dfs, ax=ax1)
    plot_delta_random(dfs, ax=ax2)
    plot_loss_curves(dfs, ax=ax3)
    plot_annotation_efficiency(dfs, ax=ax4)
    plot_auc_bar(dfs, ax=ax5)

    fig.suptitle("EE-AL vs Baselines – PASCAL VOC Active Learning Benchmark",
                 fontsize=15, fontweight="bold", y=1.01)

    out = FIGURES_DIR / "00_dashboard.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Summary Table ─────────────────────────────────────────────────────────────
def print_summary_table(dfs):
    print("\n" + "=" * 70)
    print(f"{'Strategy':<20} {'R0':>8} {'R1':>8} {'R2':>8} {'R3':>8} {'R4':>8} {'R5':>8}  {'AUC':>7}")
    print("-" * 70)
    order = ["ee_al", "badge", "bald", "coreset", "entropy", "random"]
    for strat in order:
        if strat not in dfs:
            continue
        df = dfs[strat]
        vals = [f"{v*100:.1f}" for v in df["final_miou"].values]
        x = df["n_labeled"].values
        y = df["final_miou"].values * 100
        auc = np.trapz(y, x) / (x[-1] - x[0])
        marker = " ◄ OURS" if strat == "ee_al" else ""
        print(f"{STRATEGIES[strat]['label']:<20} " + "  ".join(f"{v:>6}" for v in vals) + f"  {auc:>6.1f}%{marker}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("EE-AL Results Analyzer")
    print(f"Results directory: {RESULTS_DIR.resolve()}")
    print(f"Output directory:  {FIGURES_DIR.resolve()}")
    print()

    dfs = load_results()
    print(f"Loaded {len(dfs)} experiments: {list(dfs.keys())}\n")

    print("Generating plots...")
    plot_dashboard(dfs)
    plot_miou_curve(dfs)
    plot_delta_random(dfs)
    plot_loss_curves(dfs)
    plot_annotation_efficiency(dfs)
    plot_auc_bar(dfs)
    plot_exit_breakdown()

    print_summary_table(dfs)
    print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")
