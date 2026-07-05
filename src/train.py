"""Train the lightweight CIFAR-10 CNN on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data_loader import CIFAR10_CLASSES, get_cifar10_loaders
from model import build_model
from utils import ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 CNN.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("models") / "cnn_cifar10.pth",
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Use 0 for best Windows compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 100,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        if batch_idx % log_interval == 0:
            current_loss = running_loss / total
            current_accuracy = correct / total
            print(
                f"  batch {batch_idx}/{len(train_loader)} "
                f"| loss: {current_loss:.4f} "
                f"| train_acc: {current_accuracy * 100:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def save_checkpoint(
    model: nn.Module,
    save_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    ensure_dir(save_path.parent)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "LightweightCNN",
        "dataset": "CIFAR-10",
        "classes": CIFAR10_CLASSES,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    torch.save(checkpoint, save_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")
    train_loader, _ = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(num_classes=len(CIFAR10_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss, accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| loss: {loss:.4f} "
            f"| train_acc: {accuracy * 100:.2f}%"
        )

    save_checkpoint(
        model=model,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
