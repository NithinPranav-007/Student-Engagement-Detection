from __future__ import annotations

from pathlib import Path
from typing import Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, FER2013_EMOTION_TO_CLASS


class FER2013Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int, train: bool) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = build_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        pixels = np.asarray(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        label = int(FER2013_EMOTION_TO_CLASS[int(row["emotion"])] )
        image = Image.fromarray(pixels)
        image = image.convert("RGB")
        image = self.transform(image)
        return image, label


def build_transforms(image_size: int = DEFAULT_IMAGE_SIZE, train: bool = True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
                transforms.RandomAutocontrast(p=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _build_folder_datasets(data_dir: Path, image_size: int, val_split: float):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if train_dir.exists() and val_dir.exists():
        train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(image_size=image_size, train=True))
        val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms(image_size=image_size, train=False))
        return train_dataset, val_dataset, train_dataset.classes

    full_dataset = datasets.ImageFolder(data_dir, transform=build_transforms(image_size=image_size, train=True))
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=42,
        stratify=getattr(full_dataset, "targets", None),
    )

    train_dataset = deepcopy(full_dataset)
    train_dataset.transform = build_transforms(image_size=image_size, train=True)
    val_dataset = deepcopy(full_dataset)
    val_dataset.transform = build_transforms(image_size=image_size, train=False)

    return torch.utils.data.Subset(train_dataset, train_indices), torch.utils.data.Subset(val_dataset, val_indices), full_dataset.classes


def _build_fer2013_datasets(csv_path: Path, image_size: int, val_split: float):
    dataframe = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(dataframe, test_size=val_split, random_state=42, stratify=dataframe["emotion"])
    train_dataset = FER2013Dataset(train_df, image_size=image_size, train=True)
    val_dataset = FER2013Dataset(val_df, image_size=image_size, train=False)
    return train_dataset, val_dataset, CLASS_NAMES


def build_datasets(data_dir: str, image_size: int = DEFAULT_IMAGE_SIZE, val_split: float = 0.2):
    data_path = Path(data_dir)
    csv_path = data_path / "fer2013.csv"
    if csv_path.exists():
        return _build_fer2013_datasets(csv_path, image_size=image_size, val_split=val_split)
    return _build_folder_datasets(data_path, image_size=image_size, val_split=val_split)
