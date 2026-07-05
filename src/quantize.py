"""Apply ONNX Runtime dynamic quantization to the exported CNN model."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from data_loader import get_cifar10_loaders
from utils import ensure_dir, get_file_size_mb, set_seed


DEFAULT_INPUT_PATH = Path("models") / "cnn_cifar10.onnx"
DEFAULT_OUTPUT_PATH = Path("models") / "cnn_cifar10_int8.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the FP32 ONNX model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the INT8 ONNX model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing or receiving the CIFAR-10 dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Validation batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Use 0 for best Windows compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def quantize_model(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {input_path}. "
            "Run export first with python src/export_onnx.py."
        )

    ensure_dir(output_path.parent)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )


def load_onnx_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"Quantized ONNX model not found: {model_path}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def run_validation_inference(
    session: ort.InferenceSession,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[list[int], tuple[int, ...]]:
    _, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    images, _ = next(iter(test_loader))

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: images.numpy()})
    logits = outputs[0]
    predictions = logits.argmax(axis=1).tolist()
    return predictions, tuple(logits.shape)


def calculate_size_reduction(fp32_size_mb: float, int8_size_mb: float) -> float:
    if fp32_size_mb == 0:
        return 0.0
    return ((fp32_size_mb - int8_size_mb) / fp32_size_mb) * 100


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    quantize_model(args.input, args.output)

    fp32_size_mb = get_file_size_mb(args.input)
    int8_size_mb = get_file_size_mb(args.output)
    size_reduction = calculate_size_reduction(fp32_size_mb, int8_size_mb)

    session = load_onnx_session(args.output)
    predictions, logits_shape = run_validation_inference(
        session=session,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"FP32 ONNX model: {args.input}")
    print(f"INT8 ONNX model: {args.output}")
    print(f"FP32 ONNX size: {fp32_size_mb:.2f} MB")
    print(f"INT8 ONNX size: {int8_size_mb:.2f} MB")
    print(f"Size reduction: {size_reduction:.2f}%")
    print("ONNX Runtime loaded the quantized model successfully.")
    print(f"INT8 validation logits shape: {logits_shape}")
    print(f"INT8 validation predictions: {predictions}")
    print("INT8 inference execution: passed")
    print(
        "Note: dynamic quantization primarily targets supported weight-bearing "
        "operators such as Linear/Gemm in this model."
    )


if __name__ == "__main__":
    main()
