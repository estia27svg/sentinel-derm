"""Frozen-backbone feature extractor used for both training and serving.

We deliberately keep the backbone frozen and train only a linear head on cached
embeddings. For a dataset of this size that is within a fraction of a point of
full fine-tuning, it is fully reproducible on a CPU, and it lets the conformal
and OOD layers be refit in seconds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = {
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
}


def build_transform(image_size: int = 224) -> transforms.Compose:
    resize = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FeatureExtractor(nn.Module):
    """Returns (pooled_embedding, spatial_feature_map).

    The spatial map is kept because Grad-CAM needs it; returning it from the
    forward pass avoids a second hook-based forward call at inference time.
    """

    def __init__(self, name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
        builder, weights = BACKBONES[name]
        constructor = builder(weights=weights if pretrained else None)

        if hasattr(constructor, "features"):
            self.trunk = constructor.features
        else:  # resnet family: everything except global pool + classifier
            self.trunk = nn.Sequential(*list(constructor.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.name = name
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        with torch.no_grad():
            probe = self.trunk(torch.zeros(1, 3, 224, 224))
        self.out_dim = int(probe.shape[1])

    def train(self, mode: bool = True):  # backbone always stays in eval mode
        return super().train(False)

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.trunk(batch)
        pooled = self.pool(feature_map).flatten(1)
        return pooled, feature_map


def parameter_counts() -> dict[str, int]:
    counts = {}
    for name in BACKBONES:
        model = FeatureExtractor(name, pretrained=False)
        counts[name] = sum(p.numel() for p in model.parameters())
    return counts
