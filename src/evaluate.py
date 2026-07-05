"""Evaluate the trained PyTorch CIFAR-10 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data_loader import CIFAR10_CLASSES, get_cifar10_loaders
from model import build_model
from utils import get_device, save_json, set_seed


DEFAULT_MODEL_PATH = Path("models") / "cnn_cifar10.pth"
DEFAULT_OUTPUT_PATH = Path("results") / "evaluation_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing or receiving the CIFAR-10 dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Use 0 for best Windows compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_checkpoint(model_path: Path, device: torch.device) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}. "
            "Run training first with python src/train.py --epochs 1 --batch-size 64."
        )
    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {model_path}")
    return checkpoint


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    checkpoint = load_checkpoint(model_path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = build_model(num_classes=len(CIFAR10_CLASSES))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, int, int]:
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0.0
    return accuracy, correct, total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")
    model = load_model(args.model_path, device)
    _, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    accuracy, correct, total = evaluate(model, test_loader, device)
    result = {
        "model_path": str(args.model_path),
        "dataset": "CIFAR-10",
        "num_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
    }
    output_path = save_json(result, DEFAULT_OUTPUT_PATH)

    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
