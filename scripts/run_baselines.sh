#!/usr/bin/env bash
# run_baselines.sh – Run all 5 AL baselines sequentially
# Usage: bash scripts/run_baselines.sh [--dry-run] [--config path] [--gpu N]
set -e

CONFIG="configs/pascal_voc.yaml"
DRY=""
GPU=0
EPOCHS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY="--dry-run"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --epochs) EPOCHS="--epochs $2"; shift 2 ;;
        *) echo "Unknown arg $1"; exit 1 ;;
    esac
done

STRATEGIES=("random" "entropy" "bald" "badge" "coreset")

echo "======================================================"
echo " Running ALL Baselines"
echo " Config: ${CONFIG} | GPU: ${GPU}"
echo "======================================================"

for STRAT in "${STRATEGIES[@]}"; do
    echo ""
    echo "── Strategy: ${STRAT} ──────────────────────────────"
    python run_al_pipeline.py \
        --config ${CONFIG} \
        --strategy ${STRAT} \
        --gpu ${GPU} \
        ${DRY} ${EPOCHS}
    echo "── ${STRAT} done ──"
done

echo ""
echo "✓ All baselines complete. Results in ./results/"
