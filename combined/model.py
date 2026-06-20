"""
model.py
HybridSiam-CD+ (Combined) — Model Architecture

Components:
  - HybridChangeDetector  : original HybridSiam-CD base architecture
                             (frozen DINOv3 ViT-L semantic encoder +
                              Siamese ResNet34 spatial branch returning
                              6 feature scales s0..s5 + decoder)
  - ChannelAttention / SpatialAttention / CBAM : attention modules
                             (Woo et al., 2018)
  - LearnableTemporalFusion : replaces fixed absolute-difference fusion
                              with a learnable linear projection
  - HybridSiamPlus        : HybridChangeDetector + CBAM + learnable
                             temporal fusion (the "+" architecture)

This version matches the combined multi-domain training notebook, in
which the ResNet34 spatial branch returns 6 feature scales (s0..s5) and
the decoder bottleneck fuses (vit, upsampled s5, s4).
"""

import types

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


# ════════════════════════════════════════════════════════════════════
#  Base architecture — HybridSiam-CD (original)
# ════════════════════════════════════════════════════════════════════

class ViTEncoder(nn.Module):
    """Frozen DINOv3-sat-93M ViT-L semantic encoder with absolute-difference
    temporal fusion (replaced by LearnableTemporalFusion in HybridSiamPlus)."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
        )
        self.out_dim = self.backbone.embed_dim
        self.proj = nn.Identity()

    def forward(self, before, after):
        feat_before = self.backbone(before)
        feat_after = self.backbone(after)
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        feat_before = feat_before[:, num_prefix:]
        feat_after = feat_after[:, num_prefix:]
        feat_before = self.proj(feat_before)
        feat_after = self.proj(feat_after)
        # Original fusion: absolute difference
        return torch.abs(feat_before - feat_after)


class ResNetSpatialBranch(nn.Module):
    """Siamese ResNet34 spatial branch — extracts 6 multi-scale feature
    maps (stem output, then 4 residual stages, plus an extra deep stage)
    from each temporal image independently, fused by absolute difference
    at each scale. Returns (s0, s1, s2, s3, s4, s5)."""

    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        # Extra deep projection stage to produce s5 (used by the decoder
        # bottleneck alongside the ViT semantic features)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def _forward_single(self, x):
        s0 = self.stem(x)                 # 64ch,  H/2
        s1 = self.layer1(self.pool(s0))   # 64ch,  H/4
        s2 = self.layer2(s1)              # 128ch, H/8
        s3 = self.layer3(s2)              # 256ch, H/16
        s4 = self.layer4(s3)              # 512ch, H/32
        s5 = self.layer5(s4)              # 512ch, H/32 (deep projection)
        return s0, s1, s2, s3, s4, s5

    def forward(self, before, after):
        b0, b1, b2, b3, b4, b5 = self._forward_single(before)
        a0, a1, a2, a3, a4, a5 = self._forward_single(after)
        d0 = torch.abs(b0 - a0)
        d1 = torch.abs(b1 - a1)
        d2 = torch.abs(b2 - a2)
        d3 = torch.abs(b3 - a3)
        d4 = torch.abs(b4 - a4)
        d5 = torch.abs(b5 - a5)
        return d0, d1, d2, d3, d4, d5


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class Decoder(nn.Module):
    """Fuses ViT semantic features with ResNet spatial features (s0..s5)
    at multiple scales, progressively upsampling to full resolution.

    Bottleneck input: concat(vit, upsampled s5, s4)
    """

    def __init__(self, vit_dim=1024):
        super().__init__()
        self.bottleneck = ConvBlock(vit_dim + 512 + 512, 256)
        self.up1 = UpBlock(256, 128)
        self.fuse1 = ConvBlock(128 + 256, 128)
        self.up2 = UpBlock(128, 64)
        self.fuse2 = ConvBlock(64 + 128, 64)
        self.up3 = UpBlock(64, 64)
        self.fuse3 = ConvBlock(64 + 64, 64)
        self.up4 = UpBlock(64, 32)
        self.fuse4 = ConvBlock(32 + 64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, vit_features, resnet_features, output_size):
        B, N, D = vit_features.shape
        h = w = int(N ** 0.5)
        vit = vit_features.permute(0, 2, 1).view(B, D, h, w)

        s0, s1, s2, s3, s4, s5 = resnet_features
        s5_up = F.interpolate(s5, size=(h, w), mode="nearest")

        x = self.bottleneck(torch.cat([vit, s5_up, s4], dim=1))
        x = self.up1(x)
        x = self.fuse1(torch.cat([x, s3], dim=1))
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, s2], dim=1))
        x = self.up3(x)
        x = self.fuse3(torch.cat([x, s1], dim=1))
        x = self.up4(x)
        x = self.fuse4(torch.cat([x, s0], dim=1))
        x = self.head(x)

        if x.shape[2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="nearest")
        return x


class HybridChangeDetector(nn.Module):
    """Original HybridSiam-CD architecture: frozen DINOv3 ViT-L semantic
    encoder + Siamese ResNet34 spatial branch (6 scales) + fusion decoder."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ViTEncoder(pretrained=pretrained)
        self.spatial_branch = ResNetSpatialBranch(pretrained=pretrained)
        self.decoder = Decoder(vit_dim=self.encoder.out_dim)

    def forward(self, before, after):
        output_size = before.shape[2:]
        vit_features = self.encoder(before, after)
        resnet_features = self.spatial_branch(before, after)
        return self.decoder(vit_features, resnet_features, output_size)


# ════════════════════════════════════════════════════════════════════
#  Attention modules — CBAM (Woo et al., 2018)
# ════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module — sequential channel then
    spatial attention (Woo et al., 2018)."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


# ════════════════════════════════════════════════════════════════════
#  Learnable temporal fusion (replaces fixed absolute difference)
# ════════════════════════════════════════════════════════════════════

class LearnableTemporalFusion(nn.Module):
    """Learnable linear projection fusing before/after ViT token
    embeddings, replacing the fixed |before - after| operation used in
    the original HybridSiam-CD (Zhang et al., 2020; Chen et al., 2021)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, feat_before, feat_after):
        return self.conv(torch.cat([feat_before, feat_after], dim=-1))


# ════════════════════════════════════════════════════════════════════
#  HybridSiam-CD+ : base architecture + CBAM + learnable temporal fusion
# ════════════════════════════════════════════════════════════════════

class HybridSiamPlus(nn.Module):
    """HybridSiam-CD+ — adds five CBAM attention modules to the decoder
    fusion stages and replaces the fixed absolute-difference temporal
    fusion with a learnable linear projection.

    Loads pretrained HybridSiam-CD weights directly in __init__ (matching
    combined-training notebook behaviour), then optionally freezes the
    ViT backbone before the new CBAM/fusion modules are constructed.
    """

    def __init__(self, pretrained_path, freeze_vit=True):
        super().__init__()
        self.base = HybridChangeDetector(pretrained=False)
        self._load_pretrained(pretrained_path)

        if freeze_vit:
            for p in self.base.encoder.backbone.parameters():
                p.requires_grad = False
            print("[VRAM] ViT backbone frozen.")

        self.cbam_bottleneck = CBAM(256)
        self.cbam_fuse1 = CBAM(128)
        self.cbam_fuse2 = CBAM(64)
        self.cbam_fuse3 = CBAM(64)
        self.cbam_fuse4 = CBAM(32)

        enc_dim = self.base.encoder.out_dim
        self.temporal_fusion = LearnableTemporalFusion(enc_dim, enc_dim)

        self._patch_encoder()
        self._patch_decoder()

    def _load_pretrained(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model", "state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
        base_state = self.base.state_dict()
        compatible = {
            k: v
            for k, v in state.items()
            if k in base_state and base_state[k].shape == v.shape
        }
        self.base.load_state_dict(compatible, strict=False)
        print(f"[Weights] {len(compatible)}/{len(base_state)} keys loaded.")

    # ── monkey-patch the encoder's forward to use learnable fusion ──
    def _patch_encoder(self):
        fusion = self.temporal_fusion

        def _forward(self_enc, before, after):
            feat_before = self_enc.backbone(before)
            feat_after = self_enc.backbone(after)
            num_prefix = getattr(self_enc.backbone, "num_prefix_tokens", 1)
            feat_before = feat_before[:, num_prefix:]
            feat_after = feat_after[:, num_prefix:]
            feat_before = self_enc.proj(feat_before)
            feat_after = self_enc.proj(feat_after)
            return fusion(feat_before, feat_after)

        self.base.encoder.forward = types.MethodType(_forward, self.base.encoder)

    # ── monkey-patch the decoder's forward to insert CBAM at each stage ──
    def _patch_decoder(self):
        decoder = self.base.decoder
        cbam_bn = self.cbam_bottleneck
        cbam_f1 = self.cbam_fuse1
        cbam_f2 = self.cbam_fuse2
        cbam_f3 = self.cbam_fuse3
        cbam_f4 = self.cbam_fuse4

        def _forward(self_dec, vit_features, resnet_features, output_size):
            B, N, D = vit_features.shape
            h = w = int(N ** 0.5)
            vit = vit_features.permute(0, 2, 1).view(B, D, h, w)

            s0, s1, s2, s3, s4, s5 = resnet_features
            s5_up = F.interpolate(s5, size=(h, w), mode="nearest")

            x = self_dec.bottleneck(torch.cat([vit, s5_up, s4], dim=1))
            x = cbam_bn(x)
            x = self_dec.up1(x)
            x = self_dec.fuse1(torch.cat([x, s3], dim=1))
            x = cbam_f1(x)
            x = self_dec.up2(x)
            x = self_dec.fuse2(torch.cat([x, s2], dim=1))
            x = cbam_f2(x)
            x = self_dec.up3(x)
            x = self_dec.fuse3(torch.cat([x, s1], dim=1))
            x = cbam_f3(x)
            x = self_dec.up4(x)
            x = self_dec.fuse4(torch.cat([x, s0], dim=1))
            x = cbam_f4(x)
            x = self_dec.head(x)

            if x.shape[2:] != output_size:
                x = F.interpolate(x, size=output_size, mode="nearest")
            return x

        decoder.forward = types.MethodType(_forward, decoder)

    def forward(self, before, after):
        return self.base(before, after)

    # ── helper: build the 3 LR parameter groups used by the optimiser ──
    def get_param_groups(self, lr_decoder, lr_cbam, lr_fusion):
        vit_ids = set(id(p) for p in self.base.encoder.backbone.parameters())
        cbam_ids = (
            set(id(p) for p in self.cbam_bottleneck.parameters())
            | set(id(p) for p in self.cbam_fuse1.parameters())
            | set(id(p) for p in self.cbam_fuse2.parameters())
            | set(id(p) for p in self.cbam_fuse3.parameters())
            | set(id(p) for p in self.cbam_fuse4.parameters())
        )
        fusion_ids = set(id(p) for p in self.temporal_fusion.parameters())

        cbam_params = [p for p in self.parameters() if id(p) in cbam_ids and p.requires_grad]
        fusion_params = [p for p in self.parameters() if id(p) in fusion_ids and p.requires_grad]
        other_params = [
            p
            for p in self.parameters()
            if id(p) not in vit_ids
            and id(p) not in cbam_ids
            and id(p) not in fusion_ids
            and p.requires_grad
        ]

        return [
            {"params": other_params, "lr": lr_decoder, "name": "decoder"},
            {"params": cbam_params, "lr": lr_cbam, "name": "cbam"},
            {"params": fusion_params, "lr": lr_fusion, "name": "fusion"},
        ]
