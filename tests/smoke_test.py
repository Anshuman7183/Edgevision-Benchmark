"""Lightweight CI smoke test for committed model artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import build_model  # noqa: E402


CHECKPOINT_PATH = REPO_ROOT / "models" / "cnn_cifar10.pth"
ONNX_FP32_PATH = REPO_ROOT / "models" / "cnn_cifar10.onnx"
ONNX_INT8_PATH = REPO_ROOT / "models" / "cnn_cifar10_int8.onnx"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact is missing: {path}. "
            "Run the local pipeline to generate it and commit/include the artifact "
            "before expecting CI smoke tests to pass."
        )


def torch_load(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state_dict': {path}")
    return checkpoint


def check_shape(name: str, output: Any) -> None:
    shape = tuple(output.shape)
    if shape != (1, 10):
        raise AssertionError(f"{name} output shape should be (1, 10), got {shape}")


def run_pytorch_smoke(dummy_tensor: torch.Tensor) -> None:
    require_file(CHECKPOINT_PATH)
    checkpoint = torch_load(CHECKPOINT_PATH)

    model = build_model(num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        output = model(dummy_tensor)
    check_shape("PyTorch FP32", output)
    print("PyTorch FP32 smoke test passed.")


def run_onnx_smoke(name: str, model_path: Path, dummy_array: np.ndarray) -> None:
    require_file(model_path)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: dummy_array})[0]
    check_shape(name, output)
    print(f"{name} smoke test passed.")


def main() -> None:
    torch.manual_seed(42)
    dummy_tensor = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    dummy_array = dummy_tensor.numpy()

    run_pytorch_smoke(dummy_tensor)
    run_onnx_smoke("ONNX FP32", ONNX_FP32_PATH, dummy_array)
    run_onnx_smoke("ONNX INT8", ONNX_INT8_PATH, dummy_array)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
