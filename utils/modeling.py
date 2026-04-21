from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from .config import DEFAULT_IMAGE_SIZE


def build_model(model_name: str = "mobilenet_v2", num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )
        backbone = model
    else:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )
        backbone = model

    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        if model_name == "resnet50":
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        else:
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True

    return model


def save_checkpoint(path: Path, model, class_names, image_size: int, model_name: str) -> None:
    checkpoint = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": image_size,
        "num_classes": len(class_names),
    }
    torch.save(checkpoint, path)
