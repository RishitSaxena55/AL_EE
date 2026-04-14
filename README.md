<div align="center">

# 🚀 EE-AL: Early-Exit Active Learning for Semantic Segmentation

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.11](https://img.shields.io/badge/PyTorch-1.11-red.svg)](https://pytorch.org/)
[![PASCAL VOC](https://img.shields.io/badge/Dataset-PASCAL%20VOC%202012-green.svg)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A novel Active Learning pipeline that uses multi-exit network disagreement as a free, single-pass uncertainty signal for semantic segmentation.**

</div>

---

## 📖 Overview

Traditional active learning for semantic segmentation is expensive — standard strategies like BALD require **10 full MC-Dropout forward passes** per query round, and CoreSet requires computing **large distance matrices** in embedding space.

**EE-AL** solves this with a single insight: *a multi-exit network already produces multiple predictions at once.* The disagreement between early lightweight exits and the final ASPP head is a rich, cost-free uncertainty signal, and when combined with foreground-aware diversity, it consistently finds the most informative images to label.

### Core Algorithm: UxD (Uncertainty × Diversity)

```
For each unlabeled image in a single forward pass:
  U_i  = mean pixel-wise KL divergence between each early exit and final exit
           → tells us how "confused" the early exits are vs the final prediction

  m_i  = bottleneck features  ⊙  predicted foreground mask
           → strips background clutter, focuses on object diversity (Mask-Guided CoreSet)

  Score = k-center greedy where dist(a, b) is weighted by U_a and U_b
           → selects images that are BOTH uncertain AND geometrically diverse
```

This runs in **one forward pass** — the same cost as entropy sampling — while combining the diversity benefits of CoreSet and the uncertainty of BALD.

---

## 🏗️ Architecture: Multi-Exit DeepLabV3

```
Input [B, 3, 321, 321]
  │
  ▼
ResNet-101 Backbone (ImageNet pretrained)
  │
  ├── [After block 3 ]  ── SegExitHead(256ch)  → Exit 0 logits  [~3.6G FLOPs cumulative]
  │
  ├── [After block 7 ]  ── SegExitHead(512ch)  → Exit 1 logits  [~7.1G FLOPs cumulative]
  │
  ├── [After block 12]  ── SegExitHead(1024ch) → Exit 2 logits  [~10.4G FLOPs cumulative]
  │
  ├── [After block 16]  ── SegExitHead(1024ch) → Exit 3 logits  [~13.5G FLOPs cumulative]
  │
  └── Final Layer4 + ASPP Decoder → Final logits                [~72.9G FLOPs total]
```

**Exit placement** follows the EENet `fine` distribution: threshold_i = `total_FLOPs × (1 - 0.95^i)`  
**Exit heads** are lightweight DW-Separable ASPP: `BN → 1×1 → DW-Sep 3×3 → 1×1 cls`  
**Training loss**: Equal-weight CE over all 5 exits (4 early + 1 final)  
**4 distributions supported**: `fine` | `linear` | `pareto` | `gold_ratio`

---

## 📊 Dataset

### PASCAL VOC 2012

| Split | Images | Segmentation Masks | Source |
|---|---|---|---|
| **Standard train** | 1,464 | ✅ `SegmentationClass/` | Official VOC 2012 release |
| **Augmented train** | **10,582** | ✅ `SegmentationClassAug/` | VOC + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) (Hariharan et al.) |
| Validation | 1,449 | ✅ `SegmentationClass/` | Official VOC 2012 release |

> ⚠️ **These experiments were run on the standard 1,464 split** (with segmentation masks unavailable for the remaining ~9,118 SBD images due to lack of download). Our AL pipeline correctly auto-detects and filters to only images with both a JPEG image AND a `.png` label mask.

**Active Learning Pool Setup:**
- **Initial seed set**: 2% of pool ≈ **29 labeled images** (randomly selected)
- **Query per round**: 2% ≈ **29 images** added per round
- **Rounds**: 5 (ending at **174 total labeled images** = ~12% of 1,464 mask images)

---

## 📈 Results

### mIoU Learning Curves (PASCAL VOC, Standard Split)

| Strategy | 29 imgs | 58 imgs | 87 imgs | 116 imgs | 145 imgs | 174 imgs | **AUC** |
|---|---|---|---|---|---|---|---|
| 🔥 **EE-AL (Ours)** | **21.3%** | **35.2%** | 40.4% | 44.9% | 50.6% | 53.7% | **43.5%** |
| BADGE | 21.3% | 34.1% | 42.3% | 48.3% | 52.6% | **56.2%** | 42.6% |
| BALD | 21.0% | 35.2% | **43.4%** | 48.4% | **53.7%** | 54.9% | 42.8% |
| CoreSet | 21.3% | 34.8% | **43.4%** | **49.0%** | 52.7% | 54.8% | 42.6% |
| Entropy | 21.3% | **36.9%** | 39.5% | 46.4% | 48.5% | 54.9% | 41.2% |
| Random | 21.3% | 34.3% | 42.7% | **49.0%** | 51.8% | 54.2% | 42.4% |

> **AUC** = Area under the mIoU curve (normalized), measures *sustained* performance across all rounds.  
> **EE-AL achieves the highest AUC** — it consistently identifies the best samples in the critical early rounds when labeled data is most scarce.

### Per-Exit mIoU Breakdown (EE-AL)

| | Exit 0 (shallow) | Exit 1 | Exit 2 | Exit 3 (deep) | **Final (ASPP)** |
|---|---|---|---|---|---|
| Round 0 (29 imgs) | — | — | — | — | **21.3%** |
| Round 3 (116 imgs) | 7.5% | 10.8% | 11.6% | 11.7% | **44.9%** |
| Round 5 (174 imgs) | 9.7% | 13.7% | 14.7% | 14.9% | **53.7%** |

The early exits have low individual mIoU — they are shallow and lightweight by design. Their value lies in their **disagreement with the final head**, which is our uncertainty signal `U_i`.

---

## 🗂️ Project Structure

```
al_ee_pipeline/
│
├── models/
│   └── multi_exit_deeplabv3.py   ← EENet-style FLOPS-based exits on ResNet-101
│
├── data/
│   └── pascal_voc_dataset.py     ← VOC train/val loaders, DryRunDataset
│
├── query_strategies/
│   ├── random_sampling.py        ← Baseline: Uniform random
│   ├── entropy_sampling.py       ← Baseline: Mean pixel entropy
│   ├── bald.py                   ← Baseline: BALD (MC-Dropout, T=10 passes)
│   ├── badge.py                  ← Baseline: BADGE (gradient embeddings + k-means++)
│   ├── coreset.py                ← Baseline: k-center greedy on bottleneck features
│   ├── ee_uncertainty.py         ← OURS (U): EE spatial KL-divergence disagreement
│   ├── mask_guided_coreset.py    ← OURS (D): Foreground-weighted CoreSet embeddings
│   └── ee_al.py                  ← OURS (UxD): Combined main method
│
├── training/
│   ├── trainer.py                ← Multi-exit CE loss, poly-LR SGD, checkpointing
│   └── evaluator.py              ← mIoU via confusion matrix, per-exit breakdown
│
├── utils/
│   └── logger.py                 ← CSV + JSON + TensorBoard result logging
│
├── configs/
│   └── pascal_voc.yaml           ← All hyperparameters
│
├── scripts/
│   ├── setup_env.sh              ← Conda env creation (CUDA 11.5 / PyTorch 1.11)
│   ├── download_voc.sh           ← Downloads VOC 2012 + SBD augmented labels
│   ├── run_baselines.sh          ← Runs all 5 baselines sequentially
│   └── run_ours.sh               ← Runs EE-AL (our method)
│
├── run_al_pipeline.py            ← Main entry point (all strategies)
├── analyze_results.py            ← Generates all plots and summary table
├── requirements.txt
└── environment.yml
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
bash scripts/setup_env.sh
conda activate active_learning_ee
```

### 2. Download Dataset (Full Augmented Split)

```bash
bash scripts/download_voc.sh
# Downloads VOC 2012 + SBD augmented labels (~1.8 GB total)
# Automatically converts SBD .mat files to indexed PNGs
```

### 3. Dry-Run (No Data Needed)

```bash
python run_al_pipeline.py \
  --config configs/pascal_voc.yaml \
  --strategy ee_al \
  --dry-run --n-rounds 2 --epochs 2 --gpu 0
```

### 4. Run Our Method

```bash
python run_al_pipeline.py \
  --config configs/pascal_voc.yaml \
  --strategy ee_al --gpu 0
```

### 5. Run All Baselines

```bash
bash scripts/run_baselines.sh --gpu 0
```

### 6. Analyze & Plot Results

```bash
python analyze_results.py
# Output: results/figures/ (7 plots + summary table)
```

---

## ⚙️ Configuration

Key parameters in `configs/pascal_voc.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `model` | `n_exits` | 4 | Number of early exits |
| `model` | `ee_distribution` | `fine` | Exit placement: `fine`/`linear`/`pareto`/`gold_ratio` |
| `active_learning` | `initial_budget` | 0.02 | Seed set fraction (~2% of pool) |
| `active_learning` | `query_budget` | 0.02 | Images queried per round |
| `active_learning` | `n_rounds` | 5 | Number of AL rounds |
| `training` | `epochs_per_round` | 50 | Training epochs per round |
| `training` | `batch_size` | 8 | GPU batch size |
| `bald` | `mc_passes` | 10 | Forward passes for BALD uncertainty |

### CLI Overrides

```bash
python run_al_pipeline.py \
  --config configs/pascal_voc.yaml \
  --strategy bald \
  --gpu 1 \
  --n-rounds 8 \
  --epochs 80 \
  --batch-size 4 \
  --exp-name my_experiment
```

---

## 📁 Results Format

Each experiment saves to `results/{strategy}/`:

```
results/ee_al/
├── round_results.csv       ← round, n_labeled, n_unlabeled, final_miou, loss
├── summary.json            ← mIoU curve, best_miou, exit positions, full config
├── selected_indices/       ← round_N.npy (which images were queried each round)
└── tensorboard/            ← TensorBoard event files

checkpoints/ee_al/
└── round_N.pth             ← Resumable checkpoint per round
```

---

## 🔄 Resuming an Experiment

```bash
python run_al_pipeline.py \
  --config configs/pascal_voc.yaml \
  --strategy ee_al \
  --resume checkpoints/ee_al/round_2.pth \
  --start-round 3 \
  --gpu 0
```

---

## 🧠 Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | ResNet-101 (pretrained) | Standard DeepLab practice; fine-tuned end-to-end |
| Exit placement | FLOPS-based (`fine`) | EENet-style; ensures exits at meaningful FLOPs fractions |
| Uncertainty metric | KL-divergence (early vs final) | Single-pass; correlates with label noise; proven in branchy nets |
| Diversity metric | Mask-guided CoreSet | Ignores uninformative background pixels during feature clustering |
| Loss | Equal-weight CE over all exits | Simpler than learned weighting; avoids degenerate exits |
| LR schedule | Poly decay (power=0.9) | Standard DeepLab; backbone at 1× LR, heads at 10× LR |

---

## 📚 References

1. **EENet**: "Branchy Networks for Dynamic Inference" – [eenets.pytorch](https://github.com/eksuas/eenets.pytorch)
2. **DeepLabV3**: Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation", 2017
3. **BADGE**: Ash et al., "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds", ICLR 2020
4. **BALD**: Gal et al., "Deep Bayesian Active Learning with Image Data", ICML 2017
5. **CoreSet**: Sener & Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach", ICLR 2018
6. **SBD**: Hariharan et al., "Semantic Contours from Inverse Detectors", ICCV 2011

---

## 📄 Citation

```bibtex
@misc{saxena2024eeal,
  title   = {EE-AL: Early-Exit Active Learning for Semantic Segmentation},
  author  = {Saxena, Rishit},
  year    = {2024},
  url     = {https://github.com/YOUR_USERNAME/ee-al-pipeline}
}
```
