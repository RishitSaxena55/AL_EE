#!/usr/bin/env bash
# download_voc.sh – Download and set up PASCAL VOC 2012 + SBD augmentation
# Usage: bash scripts/download_voc.sh [--data-dir /path/to/datasets]
set -e

DATA_DIR="./datasets"
if [ "$1" = "--data-dir" ] && [ -n "$2" ]; then
    DATA_DIR="$2"
fi

VOC_DIR="${DATA_DIR}/pascal_voc"
mkdir -p "${VOC_DIR}"

echo "======================================================"
echo " Downloading PASCAL VOC 2012 + Augmented Annotations"
echo " Target: ${VOC_DIR}"
echo "======================================================"

# Download VOC 2012
if [ ! -d "${VOC_DIR}/JPEGImages" ]; then
    echo "[1/4] Downloading VOC 2012 trainval..."
    wget -q --show-progress -O /tmp/VOCtrainval_11-May-2012.tar \
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    echo "[2/4] Extracting..."
    tar -xf /tmp/VOCtrainval_11-May-2012.tar -C "${DATA_DIR}"
    # Flatten the nested structure
    if [ -d "${DATA_DIR}/VOCdevkit/VOC2012" ]; then
        mv "${DATA_DIR}/VOCdevkit/VOC2012/"* "${VOC_DIR}/"
        rm -rf "${DATA_DIR}/VOCdevkit"
    fi
    rm -f /tmp/VOCtrainval_11-May-2012.tar
    echo "[2/4] VOC 2012 extracted."
else
    echo "[1-2/4] JPEGImages already present, skipping download."
fi

# Download SBD augmented annotations (SegmentationClassAug)
if [ ! -d "${VOC_DIR}/SegmentationClassAug" ]; then
    echo "[3/4] Downloading SBD augmented annotations (benchmark.tgz)..."
    wget -q --show-progress -O /tmp/benchmark.tgz \
        http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
    echo "[4/4] Extracting augmented annotations..."
    mkdir -p /tmp/sbd_tmp
    tar -xf /tmp/benchmark.tgz -C /tmp/sbd_tmp
    mkdir -p "${VOC_DIR}/SegmentationClassAug"
    # SBD stores .mat files, but we need PNGs converted
    # Use pre-converted PNGs if available, otherwise note manual step
    if ls /tmp/sbd_tmp/benchmark/dataset/cls/*.mat >/dev/null 2>&1; then
        echo "Converting SBD .mat files to PNGs..."
        python -c "
import scipy.io, numpy as np, os
from PIL import Image
from tqdm import tqdm
sbd_cls='/tmp/sbd_tmp/benchmark/dataset/cls'
out_dir='${VOC_DIR}/SegmentationClassAug'
files = [f for f in os.listdir(sbd_cls) if f.endswith('.mat')]
for f in tqdm(files, desc='Converting'):
    mat = scipy.io.loadmat(os.path.join(sbd_cls, f))
    arr = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out_dir, f.replace('.mat', '.png')))
"
    fi
    rm -rf /tmp/sbd_tmp /tmp/benchmark.tgz
else
    echo "[3-4/4] SegmentationClassAug already present."
fi

echo ""
echo "✓ Dataset setup complete at: ${VOC_DIR}"
echo "  Update configs/pascal_voc.yaml → data_dir: ${VOC_DIR}"
