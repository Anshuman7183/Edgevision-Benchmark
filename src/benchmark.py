"""Benchmark PyTorch FP32, ONNX FP32, and ONNX INT8 inference."""

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from data_loader import CIFAR10_CLASSES, get_cifar10_datasets
from model import build_model
from utils import ensure_dir, get_device, get_file_size_mb, save_json, set_seed


DEFAULT_CHECKPOINT_PATH = Path("models") / "cnn_cifar10.pth"
DEFAULT_ONNX_PATH = Path("models") / "cnn_cifar10.onnx"
DEFAULT_INT8_PATH = Path("models") / "cnn_cifar10_int8.onnx"
DEFAULT_CSV_PATH = Path("results") / "benchmark_results.csv"
DEFAULT_JSON_PATH = Path("results") / "benchmark_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch, ONNX FP32, and ONNX INT8 inference."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of CIFAR-10 test samples to preload and evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for accuracy and latency measurement.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=20,
        help="Untimed inference runs before measurement.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=100,
        help="Timed inference runs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help="Path to the ONNX FP32 model.",
    )
    parser.add_argument(
        "--int8",
        type=Path,
        default=DEFAULT_INT8_PATH,
        help="Path to the ONNX INT8 model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing or receiving the CIFAR-10 dataset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Reserved for CLI consistency. Preloading is done in-process.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Append benchmark rows to existing CSV/JSON results.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.measured_runs <= 0:
        raise ValueError("--measured-runs must be positive.")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run training first with python src/train.py."
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


def load_onnx_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def preload_test_subset(
    data_dir: Path,
    num_samples: int,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[np.ndarray], torch.Tensor]:
    _, test_dataset = get_cifar10_datasets(data_dir=data_dir, download=True)
    if num_samples > len(test_dataset):
        raise ValueError(
            f"Requested {num_samples} samples, but CIFAR-10 test set has "
            f"{len(test_dataset)} samples."
        )

    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in range(num_samples):
        image, label = test_dataset[index]
        images.append(image)
        labels.append(int(label))

    all_images = torch.stack(images)
    all_labels = torch.tensor(labels, dtype=torch.long)

    torch_batches = [
        all_images[start : start + batch_size]
        for start in range(0, num_samples, batch_size)
    ]
    numpy_batches = [batch.numpy() for batch in torch_batches]
    return torch_batches, numpy_batches, all_labels


def predict_pytorch(
    model: nn.Module,
    batches: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def predict_onnx(
    session: ort.InferenceSession,
    batches: list[np.ndarray],
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions: list[np.ndarray] = []

    for batch in batches:
        logits = session.run([output_name], {input_name: batch})[0]
        predictions.append(np.argmax(logits, axis=1))
    return np.concatenate(predictions)


def calculate_accuracy(predictions: np.ndarray, labels: torch.Tensor) -> float:
    labels_np = labels.numpy()
    return float(np.mean(predictions == labels_np))


def benchmark_latency(
    inference_fn: Callable[[Any], None],
    batches: list[Any],
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, float]:
    for batch in take_cycled(batches, warmup_runs):
        inference_fn(batch)

    latencies_ms: list[float] = []
    measured_samples = 0
    for batch in take_cycled(batches, measured_runs):
        batch_size = len(batch)
        start_time = time.perf_counter()
        inference_fn(batch)
        elapsed_seconds = time.perf_counter() - start_time
        latencies_ms.append(elapsed_seconds * 1000)
        measured_samples += batch_size

    latencies = np.array(latencies_ms, dtype=np.float64)
    total_seconds = float(np.sum(latencies) / 1000)
    throughput = measured_samples / total_seconds if total_seconds > 0 else 0.0

    return {
        "avg_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_samples_per_sec": float(throughput),
    }


def take_cycled(items: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    iterator = cycle(items)
    return [next(iterator) for _ in range(count)]


def build_pytorch_inference_fn(
    model: nn.Module,
    device: torch.device,
) -> Callable[[torch.Tensor], None]:
    def infer(batch: torch.Tensor) -> None:
        with torch.inference_mode():
            model(batch.to(device))

    return infer


def build_onnx_inference_fn(
    session: ort.InferenceSession,
) -> Callable[[np.ndarray], None]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def infer(batch: np.ndarray) -> None:
        session.run([output_name], {input_name: batch})

    return infer


def format_row(result: dict[str, Any]) -> str:
    return (
        f"{result['model_format']:<14} "
        f"{result['batch_size']:>5} "
        f"{result['num_samples']:>7} "
        f"{result['accuracy'] * 100:>8.2f}% "
        f"{result['avg_latency_ms']:>12.3f} "
        f"{result['median_latency_ms']:>12.3f} "
        f"{result['p95_latency_ms']:>10.3f} "
        f"{result['throughput_samples_per_sec']:>14.2f} "
        f"{result['model_size_mb']:>10.2f}"
    )


def load_existing_json(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of benchmark rows in {output_path}")
    return data


def merge_results(
    new_results: list[dict[str, Any]],
    json_path: Path,
    append_results: bool,
) -> list[dict[str, Any]]:
    if not append_results:
        return new_results
    return load_existing_json(json_path) + new_results


def save_csv(results: list[dict[str, Any]], output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return output_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")
    print("Loading models...")
    pytorch_model = load_pytorch_model(args.checkpoint, device)
    onnx_fp32_session = load_onnx_session(args.onnx)
    onnx_int8_session = load_onnx_session(args.int8)

    print("Preloading CIFAR-10 test subset into memory...")
    torch_batches, numpy_batches, labels = preload_test_subset(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    print(
        f"Preloaded {args.num_samples} samples as {len(torch_batches)} batches "
        f"of up to {args.batch_size}."
    )

    benchmark_specs = [
        {
            "model_format": "PyTorch FP32",
            "model_path": args.checkpoint,
            "accuracy_predictions": predict_pytorch(
                pytorch_model, torch_batches, device
            ),
            "inference_fn": build_pytorch_inference_fn(pytorch_model, device),
            "latency_batches": torch_batches,
        },
        {
            "model_format": "ONNX FP32",
            "model_path": args.onnx,
            "accuracy_predictions": predict_onnx(onnx_fp32_session, numpy_batches),
            "inference_fn": build_onnx_inference_fn(onnx_fp32_session),
            "latency_batches": numpy_batches,
        },
        {
            "model_format": "ONNX INT8",
            "model_path": args.int8,
            "accuracy_predictions": predict_onnx(onnx_int8_session, numpy_batches),
            "inference_fn": build_onnx_inference_fn(onnx_int8_session),
            "latency_batches": numpy_batches,
        },
    ]

    results: list[dict[str, Any]] = []
    print("Measuring inference latency...")
    for spec in benchmark_specs:
        latency_metrics = benchmark_latency(
            inference_fn=spec["inference_fn"],
            batches=spec["latency_batches"],
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        )
        accuracy = calculate_accuracy(spec["accuracy_predictions"], labels)
        result = {
            "model_format": spec["model_format"],
            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100,
            "avg_latency_ms": latency_metrics["avg_latency_ms"],
            "median_latency_ms": latency_metrics["median_latency_ms"],
            "p95_latency_ms": latency_metrics["p95_latency_ms"],
            "throughput_samples_per_sec": latency_metrics[
                "throughput_samples_per_sec"
            ],
            "model_size_mb": get_file_size_mb(spec["model_path"]),
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
        }
        results.append(result)

    all_results = merge_results(
        new_results=results,
        json_path=DEFAULT_JSON_PATH,
        append_results=args.append_results,
    )
    csv_path = save_csv(all_results, DEFAULT_CSV_PATH)
    json_path = save_json(all_results, DEFAULT_JSON_PATH)

    print()
    print(
        "Model Format   Batch Samples Accuracy   Avg ms/batch   Median ms   "
        "P95 ms    Samples/sec    Size MB"
    )
    print("-" * 106)
    for result in all_results:
        print(format_row(result))

    print()
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved JSON results to {json_path}")


if __name__ == "__main__":
    main()
