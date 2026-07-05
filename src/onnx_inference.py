"""Validate ONNX Runtime inference against the PyTorch checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from data_loader import CIFAR10_CLASSES, get_cifar10_loaders
from model import build_model
from utils import get_device, set_seed


DEFAULT_CHECKPOINT_PATH = Path("models") / "cnn_cifar10.pth"
DEFAULT_ONNX_PATH = Path("models") / "cnn_cifar10.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX Runtime predictions."
    )
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
        help="Path to the exported ONNX FP32 model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing or receiving the CIFAR-10 dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Use 0 for best Windows compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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


def load_pytorch_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(num_classes=len(CIFAR10_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. "
            "Run export first with python src/export_onnx.py."
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def get_test_batch(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return next(iter(test_loader))


def run_pytorch_inference(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(inputs.to(device))
    return logits.cpu().numpy()


def run_onnx_inference(
    session: ort.InferenceSession,
    inputs: torch.Tensor,
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: inputs.cpu().numpy()})
    return outputs[0]


def compare_logits(
    pytorch_logits: np.ndarray,
    onnx_logits: np.ndarray,
) -> dict[str, Any]:
    pytorch_predictions = np.argmax(pytorch_logits, axis=1)
    onnx_predictions = np.argmax(onnx_logits, axis=1)
    abs_diff = np.abs(pytorch_logits - onnx_logits)

    matching_predictions = int(np.sum(pytorch_predictions == onnx_predictions))
    total_predictions = int(pytorch_predictions.shape[0])

    return {
        "pytorch_predictions": pytorch_predictions.tolist(),
        "onnx_predictions": onnx_predictions.tolist(),
        "matching_predictions": matching_predictions,
        "total_predictions": total_predictions,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")
    pytorch_model = load_pytorch_model(args.checkpoint, device)
    onnx_session = load_onnx_session(args.onnx_path)
    images, labels = get_test_batch(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    pytorch_logits = run_pytorch_inference(pytorch_model, images, device)
    onnx_logits = run_onnx_inference(onnx_session, images)
    comparison = compare_logits(pytorch_logits, onnx_logits)

    print(f"Batch labels: {labels.tolist()}")
    print(f"PyTorch predictions: {comparison['pytorch_predictions']}")
    print(f"ONNX predictions: {comparison['onnx_predictions']}")
    print(
        "Matching predictions: "
        f"{comparison['matching_predictions']}/{comparison['total_predictions']}"
    )
    print(f"Max absolute logits difference: {comparison['max_abs_diff']:.8f}")
    print(f"Mean absolute logits difference: {comparison['mean_abs_diff']:.8f}")

    if comparison["matching_predictions"] == comparison["total_predictions"]:
        print("Prediction consistency: passed")
    else:
        print("Prediction consistency: failed")


if __name__ == "__main__":
    main()
