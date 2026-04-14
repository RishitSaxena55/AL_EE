"""
Multi-Exit DeepLabV3 for Semantic Segmentation
================================================
Architecture:
  - Backbone: ResNet-101 (torchvision, pretrained ImageNet)
  - Early exits inserted at residual block boundaries based on FLOPS thresholds
    (inspired by EENet: github.com/eksuas/eenets.pytorch)
  - Each exit head: lightweight ASPP (3 depthwise-separable convs) → 1×1 conv → num_classes
  - Final head: full DeepLabV3 ASPP decoder
  - Forward (train): returns list of [B, C, H, W] logits for ALL exits + final
  - Forward (query): returns same + bottleneck features for CoreSet diversity

FLOPS-based exit distributions (from EENet):
  - 'fine':       threshold_i = total_flops * (1 - 0.95^(i+1))   ← default, dense early
  - 'linear':     threshold_i = total_flops * margin * (i+1)
  - 'pareto':     threshold_i = total_flops * (1 - 0.80^(i+1))
  - 'gold_ratio': threshold_i = total_flops * gold_rate^(i - num_ee)
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from typing import List, Optional, Tuple, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# FLOPS counter (lightweight, only counts Conv2d and BN - sufficient for placement)
# ─────────────────────────────────────────────────────────────────────────────

def _count_flops(module: nn.Module, input_shape: Tuple[int, int, int]) -> float:
    """Count forward-pass MAC FLOPs for a module via hooks."""
    total = [0.0]

    handles = []

    def _conv_hook(m, inp, out):
        b, c_in, h_out, w_out = out.shape[0], m.in_channels, out.shape[2], out.shape[3]
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        flops = b * m.out_channels * (c_in // m.groups) * kh * kw * h_out * w_out
        total[0] += flops

    def _bn_hook(m, inp, out):
        total[0] += 2 * out.numel()

    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(_conv_hook))
        elif isinstance(mod, nn.BatchNorm2d):
            handles.append(mod.register_forward_hook(_bn_hook))

    device = next(module.parameters(), torch.zeros(1)).device
    dummy = torch.zeros(1, *input_shape, device=device)
    with torch.no_grad():
        try:
            module.eval()(dummy)
        except Exception:
            pass

    for h in handles:
        h.remove()

    return total[0]


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Segmentation Exit Head
# ─────────────────────────────────────────────────────────────────────────────

class SegExitHead(nn.Module):
    """
    Lightweight exit head for dense prediction.
    Uses:  BN → AdaptiveAvgPool (skip) OR direct Conv path
    Path:  inplanes → bottleneck_ch (1×1) → 3×3 dw-sep → num_classes (1×1)
    Also produces confidence score (scalar per image) via GAP → Linear → Sigmoid
    """

    def __init__(self, in_channels: int, num_classes: int, bottleneck_ch: int = 128):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_ch, 1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_dw = nn.Sequential(
            # depthwise
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1, groups=bottleneck_ch, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
            # pointwise
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(bottleneck_ch, num_classes, 1)

        # confidence head (for EENet-style inference gating, not used in AL query)
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conf = self.conf_head(x)              # [B, 1]
        feat = self.reduce(x)
        feat = self.conv_dw(feat)
        logits = self.classifier(feat)        # [B, C, H', W']
        return logits, conf


# ─────────────────────────────────────────────────────────────────────────────
# Final DeepLabV3 ASPP Decoder Head
# ─────────────────────────────────────────────────────────────────────────────

class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        out = super().forward(x)
        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, atrous_rates=(6, 12, 18), out_ch: int = 256):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            *[ASPPConv(in_ch, out_ch, r) for r in atrous_rates],
            ASPPPooling(in_ch, out_ch),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))


class DeepLabV3Head(nn.Module):
    """Full DeepLabV3 decode head (final exit)."""

    def __init__(self, in_ch: int, num_classes: int, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.aspp = ASPP(in_ch, atrous_rates, out_ch=256)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        # GAP for bottleneck embeddings (CoreSet diversity features)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        aspp_out = self.aspp(x)
        logits = self.final_conv(aspp_out)   # [B, C, H', W']
        if return_feat:
            # [B, 256] bottleneck embedding
            feat = self.gap(aspp_out).flatten(1)
            return logits, feat
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Exit DeepLabV3 (ResNet-101 backbone)
# ─────────────────────────────────────────────────────────────────────────────

class MultiExitDeepLabV3(nn.Module):
    """
    Multi-exit DeepLabV3 with ResNet-101 backbone.

    Exit placement follows EENet's FLOPS-threshold approach:
      For each residual block boundary in the backbone, compute cumulative FLOPs.
      If cumulative_flops >= threshold[next_exit], insert an exit head there.

    This gives num_ee exits + 1 final head. For ResNet-101 (3+4+23+3 blocks),
    using distribution='fine' and num_ee=4, exits are densely placed in the
    earlier layers where most computation happens.

    Args:
        num_classes (int): 21 for PASCAL VOC
        num_ee (int): number of early exits (default 4)
        distribution (str): 'fine' | 'linear' | 'pareto' | 'gold_ratio'
        pretrained (bool): use ImageNet pretrained ResNet-101 backbone
        exit_bottleneck_ch (int): channels in exit head reduce layer
    """

    DISTRIBUTIONS = ('fine', 'linear', 'pareto', 'gold_ratio')

    def __init__(
        self,
        num_classes: int = 21,
        num_ee: int = 4,
        distribution: str = 'fine',
        pretrained: bool = True,
        exit_bottleneck_ch: int = 128,
        input_shape: Tuple[int, int, int] = (3, 321, 321),
    ):
        super().__init__()
        assert distribution in self.DISTRIBUTIONS, f"distribution must be one of {self.DISTRIBUTIONS}"

        self.num_classes = num_classes
        self.num_ee = num_ee
        self.distribution = distribution
        self.input_shape = input_shape

        # ── Build backbone (ResNet-101) with modified output stride ──────────
        # torchvision ResNet-101 with dilated conv for stride=16 (DeepLab-style)
        backbone = tvm.resnet101(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])

        # Stem and pooling
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 3 blocks,  out: 256ch,  stride 4
        self.layer2 = backbone.layer2  # 4 blocks,  out: 512ch,  stride 8
        self.layer3 = backbone.layer3  # 23 blocks, out: 1024ch, stride 8 (dilated)
        self.layer4 = backbone.layer4  # 3 blocks,  out: 2048ch, stride 8 (dilated)

        # Collect all residual blocks in order for FLOPS-based placement
        # We will insert exits AFTER each complete block
        self._all_blocks = (
            list(self.layer1.children()) +  # 3 blocks
            list(self.layer2.children()) +  # 4 blocks
            list(self.layer3.children()) +  # 23 blocks
            list(self.layer4.children())    # 3 blocks
        )
        # Channel widths after each block
        self._block_out_channels = (
            [256] * len(list(self.layer1.children())) +
            [512] * len(list(self.layer2.children())) +
            [1024] * len(list(self.layer3.children())) +
            [2048] * len(list(self.layer4.children()))
        )

        # ── Compute FLOPS thresholds ─────────────────────────────────────────
        print(f"[MultiExitDeepLabV3] Computing backbone FLOPS for exit placement...")
        total_flops = self._estimate_backbone_flops()
        self.exit_flop_thresholds = self._compute_thresholds(total_flops, num_ee, distribution)
        print(f"[MultiExitDeepLabV3] Total backbone FLOPs: {total_flops/1e9:.2f}G")
        print(f"[MultiExitDeepLabV3] Exit FLOPs thresholds: {[f'{t/1e9:.2f}G' for t in self.exit_flop_thresholds]}")

        # ── Place exits ───────────────────────────────────────────────────────
        self.exit_positions: List[int] = []  # block indices where exits are placed
        self.exit_heads = nn.ModuleList()
        self._place_exits(exit_bottleneck_ch)
        print(f"[MultiExitDeepLabV3] Placed {len(self.exit_heads)} exits at block indices: {self.exit_positions}")
        # Update actual num_ee after placement
        self.num_ee = len(self.exit_heads)

        # ── Final DeepLabV3 head ──────────────────────────────────────────────
        self.final_head = DeepLabV3Head(in_ch=2048, num_classes=num_classes)

        # FLOP cost ratio for each exit (used by loss and AL scoring)
        self.exit_cost_ratios: List[float] = []  # filled after placement
        self._compute_exit_cost_ratios(total_flops)

    # ── FLOPS helpers ─────────────────────────────────────────────────────────

    def _estimate_backbone_flops(self) -> float:
        """
        Estimate total backbone FLOPs by running each block INDIVIDUALLY.
        Tracks dummy tensor shape through the network to handle stride changes.
        Much faster than running one growing sequential (O(N^2) → O(N)).
        """
        dummy = torch.zeros(1, *self.input_shape)
        total = 0.0

        # Stem
        total += _count_flops(self.layer0, tuple(dummy.shape[1:]))
        with torch.no_grad():
            dummy = self.layer0(dummy)

        all_blocks = (
            list(self.layer1.children()) +
            list(self.layer2.children()) +
            list(self.layer3.children()) +
            list(self.layer4.children())
        )
        for block in all_blocks:
            total += _count_flops(block, tuple(dummy.shape[1:]))
            with torch.no_grad():
                dummy = block(dummy)

        return total

    def _compute_thresholds(self, total_flops: float, num_ee: int, distribution: str) -> List[float]:
        gold_rate = 1.61803398875
        margin = 1.0 / (num_ee + 1)
        thresholds = []
        for i in range(num_ee):
            if distribution == 'fine':
                thresholds.append(total_flops * (1 - 0.95 ** (i + 1)))
            elif distribution == 'linear':
                thresholds.append(total_flops * margin * (i + 1))
            elif distribution == 'pareto':
                thresholds.append(total_flops * (1 - 0.80 ** (i + 1)))
            else:  # gold_ratio
                thresholds.append(total_flops * (gold_rate ** (i - num_ee)))
        return thresholds

    def _place_exits(self, exit_bottleneck_ch: int):
        """
        Walk through all residual blocks, accumulate FLOPs INCREMENTALLY,
        insert exit when cumulative FLOPs crosses threshold[next_exit_idx].

        KEY OPTIMIZATION vs original: each block is measured INDIVIDUALLY
        (not as a growing sequential), making this O(N) not O(N^2).
        Dummy tensor tracks actual spatial dimensions through stride changes.
        """
        dummy = torch.zeros(1, *self.input_shape)

        # Stem FLOPs
        stem_flops = _count_flops(self.layer0, tuple(dummy.shape[1:]))
        with torch.no_grad():
            dummy = self.layer0(dummy)
        cum_flops = stem_flops

        next_exit_idx = 0
        # Cumulative FLOPs at each exit position (for cost ratio computation)
        self._exit_cum_flops: List[float] = []

        all_blocks = (
            list(self.layer1.children()) +
            list(self.layer2.children()) +
            list(self.layer3.children()) +
            list(self.layer4.children())
        )
        block_channels = (
            [256] * len(list(self.layer1.children())) +
            [512] * len(list(self.layer2.children())) +
            [1024] * len(list(self.layer3.children())) +
            [2048] * len(list(self.layer4.children()))
        )

        for b_idx, (block, ch) in enumerate(zip(all_blocks, block_channels)):
            # Measure this single block with current input shape
            block_flops = _count_flops(block, tuple(dummy.shape[1:]))
            cum_flops += block_flops

            # Advance dummy tensor (track shape changes from stride)
            with torch.no_grad():
                dummy = block(dummy)

            if (next_exit_idx < len(self.exit_flop_thresholds) and
                    cum_flops >= self.exit_flop_thresholds[next_exit_idx]):
                self.exit_positions.append(b_idx)
                self.exit_heads.append(SegExitHead(ch, self.num_classes, exit_bottleneck_ch))
                self._exit_cum_flops.append(cum_flops)
                next_exit_idx += 1

        if next_exit_idx < self.num_ee:
            print(f"[MultiExitDeepLabV3] WARNING: Only placed {next_exit_idx}/{self.num_ee} exits "
                  f"(model may be too shallow). Proceeding with {next_exit_idx} exits.")

    def _compute_exit_cost_ratios(self, total_flops: float):
        """Compute FLOPs fraction at each exit (already computed during _place_exits)."""
        for f in self._exit_cum_flops:
            self.exit_cost_ratios.append(f / max(total_flops, 1.0))
        self.exit_cost_ratios.append(1.0)  # final exit always costs 100%

    # ── Forward  ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            x: [B, 3, H, W]
            return_features: if True, also returns bottleneck features [B, 256]
                             from final ASPP (for CoreSet diversity)
        Returns:
            dict with keys:
              'exit_logits': list of [B, C, H, W] tensors, one per early exit
              'final_logits': [B, C, H, W]
              'exit_confs':  list of [B, 1] confidence scalars (early exits)
              'bottleneck_feat' (optional): [B, 256]
              'exit_cost_ratios': list of float
        """
        inp_size = x.shape[-2:]

        # ── Run backbone block-by-block ───────────────────────────────────────
        x = self.layer0(x)

        all_blocks = (
            list(self.layer1.children()) +
            list(self.layer2.children()) +
            list(self.layer3.children()) +
            list(self.layer4.children())
        )

        exit_logits: List[torch.Tensor] = []
        exit_confs: List[torch.Tensor] = []
        exit_ptr = 0  # index into exit_positions / exit_heads

        for b_idx, block in enumerate(all_blocks):
            x = block(x)
            if exit_ptr < len(self.exit_positions) and b_idx == self.exit_positions[exit_ptr]:
                e_logits, e_conf = self.exit_heads[exit_ptr](x)
                # Upsample to input resolution
                e_logits = F.interpolate(e_logits, size=inp_size, mode='bilinear', align_corners=False)
                exit_logits.append(e_logits)
                exit_confs.append(e_conf)
                exit_ptr += 1

        # ── Final ASPP head ───────────────────────────────────────────────────
        if return_features:
            final_logits, bottleneck_feat = self.final_head(x, return_feat=True)
        else:
            final_logits = self.final_head(x, return_feat=False)
            bottleneck_feat = None

        final_logits = F.interpolate(final_logits, size=inp_size, mode='bilinear', align_corners=False)

        result = {
            'exit_logits': exit_logits,
            'final_logits': final_logits,
            'exit_confs': exit_confs,
            'exit_cost_ratios': self.exit_cost_ratios,
        }
        if return_features:
            result['bottleneck_feat'] = bottleneck_feat
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_multi_exit_model(cfg: dict) -> MultiExitDeepLabV3:
    """Build model from config dict."""
    return MultiExitDeepLabV3(
        num_classes=cfg.get('num_classes', 21),
        num_ee=cfg.get('n_exits', 4),
        distribution=cfg.get('ee_distribution', 'fine'),
        pretrained=cfg.get('pretrained', True),
        exit_bottleneck_ch=cfg.get('exit_bottleneck_ch', 128),
        input_shape=tuple(cfg.get('input_shape', [3, 321, 321])),
    )
