"""
PASCAL VOC 2012 (augmented) dataset loader.
Reuses logic from best_practices_ALSS/data/dataloaders/pascal_voc.py
Expected directory structure:
  data_dir/
    JPEGImages/     *.jpg
    SegmentationClassAug/  *.png   (augmented split, 10582 train images)
"""

import os
import os.path as osp
import numpy as np
import random
import torch
from torch.utils import data
import cv2
from PIL import Image

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


class VOCDataSet(data.Dataset):
    """Training dataset with optional augmentation (scale + mirror)."""

    def __init__(self, root, list_path, crop_size=(321, 321),
                 mean=IMG_MEAN, scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mirror = mirror
        self.mean = mean
        self.ignore_label = ignore_label
        self.img_ids = [l.strip() for l in open(list_path) if l.strip()]
        # Auto-detect augmented vs standard segmentation split
        seg_dir = 'SegmentationClassAug' if osp.isdir(osp.join(root, 'SegmentationClassAug')) else 'SegmentationClass'
        self.files = []
        for n in self.img_ids:
            img_path = osp.join(root, 'JPEGImages', f'{n}.jpg')
            lbl_path = osp.join(root, seg_dir, f'{n}.png')
            if osp.exists(img_path) and osp.exists(lbl_path):
                self.files.append({'img': img_path, 'label': lbl_path, 'name': n})

    def __len__(self):
        return len(self.files)

    def _generate_scale_label(self, image, label):
        f = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        d = self.files[index]
        image = cv2.imread(d['img'], cv2.IMREAD_COLOR)
        label = np.array(Image.open(d['label']), dtype=np.uint8)
        size = image.shape
        if self.scale:
            image, label = self._generate_scale_label(image, label)
        image = np.asarray(image, np.float32) - self.mean

        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,))

        h, w = label.shape
        h_off = random.randint(0, h - self.crop_h)
        w_off = random.randint(0, w - self.crop_w)
        image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]

        # BGR → RGB, HWC → CHW
        image = image[:, :, ::-1].transpose(2, 0, 1)
        if self.mirror and np.random.rand() < 0.5:
            image = image[:, :, ::-1]
            label = label[:, ::-1]

        return image.copy(), label.copy(), np.array(size), d['name'], index


class VOCGTDataSet(data.Dataset):
    """Validation dataset (full image, no crop)."""

    def __init__(self, root, list_path, crop_size=(505, 505), mean=IMG_MEAN,
                 scale=False, mirror=False, ignore_label=255):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.ignore_label = ignore_label
        self.img_ids = [l.strip() for l in open(list_path) if l.strip()]
        seg_dir = 'SegmentationClassAug' if osp.isdir(osp.join(root, 'SegmentationClassAug')) else 'SegmentationClass'
        self.files = []
        for n in self.img_ids:
            img_path = osp.join(root, 'JPEGImages', f'{n}.jpg')
            lbl_path = osp.join(root, seg_dir, f'{n}.png')
            if osp.exists(img_path) and osp.exists(lbl_path):
                self.files.append({'img': img_path, 'label': lbl_path, 'name': n})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        d = self.files[index]
        image = cv2.imread(d['img'], cv2.IMREAD_COLOR)
        label = np.array(Image.open(d['label']), dtype=np.uint8)
        size = image.shape
        image = np.asarray(image, np.float32) - self.mean

        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,))

        image = image[:, :, ::-1].transpose(2, 0, 1)
        return image.copy(), label.copy(), np.array(size), d['name'], index


class DryRunDataset(data.Dataset):
    """
    Synthetic dataset for --dry-run mode.
    Returns random tensors with correct shapes (no real data needed).
    """
    def __init__(self, length=200, crop_size=(321, 321), num_classes=21, ignore_label=255):
        self.length = length
        self.h, self.w = crop_size
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = np.random.randn(3, self.h, self.w).astype(np.float32)
        label = np.random.randint(0, self.num_classes, (self.h, self.w)).astype(np.float32)
        return image, label, np.array([self.h, self.w, 3]), f'dry_{index}', index
