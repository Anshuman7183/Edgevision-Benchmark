"""CIFAR-10 dataset and DataLoader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms() -> transforms.Compose:
    """Return deterministic tensor transforms for CIFAR-10."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_cifar10_datasets(
    data_dir: str | Path = "data",
    download: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 train and test datasets through torchvision."""
    transform = get_cifar10_transforms()
    root = Path(data_dir)

    train_dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.CIFAR10(
        root=str(root),
        train=False,
        transform=transform,
        download=download,
    )
    return train_dataset, test_dataset


def get_cifar10_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: bool = False,
    seed: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create CPU-friendly CIFAR-10 train and test DataLoaders."""
    train_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        download=download,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
