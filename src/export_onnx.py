"""Export the trained CIFAR-10 PyTorch model to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn

from model import build_model
from utils import ensure_dir, get_device, get_file_size_mb


DEFAULT_CHECKPOINT_PATH = Path("models") / "cnn_cifar10.pth"
DEFAULT_ONNX_PATH = Path("models") / "cnn_cifar10.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 CNN to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help="Output path for the exported ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to use for export.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run training first with python src/train.py --epochs 1 --batch-size 64."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state_dict': {checkpoint_path}")
    return checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    device: torch.device,
    opset: int,
) -> None:
    ensure_dir(onnx_path.parent)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def validate_onnx_model(onnx_path: Path) -> None:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


def main() -> None:
    args = parse_args()
    device = get_device()

    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)
    export_to_onnx(
        model=model,
        onnx_path=args.onnx_path,
        device=device,
        opset=args.opset,
    )
    validate_onnx_model(args.onnx_path)

    size_mb = get_file_size_mb(args.onnx_path)
    print("ONNX validation passed.")
    print(f"Exported ONNX model to {args.onnx_path}")
    print(f"ONNX file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
