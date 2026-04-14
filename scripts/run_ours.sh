#!/usr/bin/env bash
# run_ours.sh – Run our novel EE-AL (UxD) method
# Usage: bash scripts/run_ours.sh [--dry-run] [--config path] [--gpu N]
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

echo "======================================================"
echo " EE-AL (UxD) – Our Proposed Method"
echo " Config: ${CONFIG} | GPU: ${GPU}"
echo "======================================================"

python run_al_pipeline.py \
    --config ${CONFIG} \
    --strategy ee_al \
    --gpu ${GPU} \
    ${DRY} ${EPOCHS}

echo ""
echo "✓ EE-AL complete. Results in ./results/ee_al/"
