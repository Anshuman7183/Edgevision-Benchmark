"""Benchmark local ONNX Runtime FP32 against Triton-served ONNX FP32."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from data_loader import get_cifar10_datasets
from utils import ensure_dir, get_file_size_mb, save_json, set_seed


DEFAULT_TRITON_URL = "localhost:8000"
DEFAULT_MODEL_NAME = "edgevision_cnn"
DEFAULT_ONNX_PATH = Path("models") / "cnn_cifar10.onnx"
DEFAULT_CSV_PATH = Path("results") / "triton_benchmark_results.csv"
DEFAULT_JSON_PATH = Path("results") / "triton_benchmark_results.json"
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
CSV_FIELDNAMES = [
    "model_format",
    "accuracy",
    "accuracy_percent",
    "avg_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "throughput_samples_per_sec",
    "request_success_rate",
    "request_success_percent",
    "successful_requests",
    "failed_requests",
    "total_requests",
    "model_size_mb",
    "num_samples",
    "batch_size",
    "warmup_runs",
    "measured_runs",
    "url",
    "model_name",
    "onnx_path",
    "seed",
]


@dataclass
class RequestStats:
    successful_requests: int = 0
    failed_requests: int = 0

    @property
    def total_requests(self) -> int:
        return self.successful_requests + self.failed_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def record_success(self) -> None:
        self.successful_requests += 1

    def record_failure(self) -> None:
        self.failed_requests += 1


@dataclass(frozen=True)
class TritonRequest:
    inputs: list[httpclient.InferInput]
    outputs: list[httpclient.InferRequestedOutput]
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local ONNX Runtime FP32 and Triton-served ONNX FP32."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_TRITON_URL,
        help=f"Triton HTTP endpoint host:port. Default: {DEFAULT_TRITON_URL}",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Triton model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help=f"Path to local ONNX FP32 model. Default: {DEFAULT_ONNX_PATH}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing or receiving the CIFAR-10 dataset. Default: data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of CIFAR-10 test samples in the fixed subset. Default: 500",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for accuracy and latency measurement. Default: 1",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=20,
        help="Untimed inference runs before measurement. Default: 20",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=100,
        help="Timed inference runs. Default: 100",
    )
    parser.add_argument("--seed", type=int, default=42, help="Subset seed. Default: 42")
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Append benchmark rows to existing CSV/JSON results.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.batch_size <= 0 or args.batch_size > 8:
        raise ValueError("--batch-size must be between 1 and 8 for this Triton model.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.measured_runs <= 0:
        raise ValueError("--measured-runs must be positive.")
    if not args.onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")


def load_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_names = {item.name for item in session.get_inputs()}
    output_names = {item.name for item in session.get_outputs()}
    if INPUT_NAME not in input_names:
        raise ValueError(f"ONNX model does not expose input '{INPUT_NAME}'.")
    if OUTPUT_NAME not in output_names:
        raise ValueError(f"ONNX model does not expose output '{OUTPUT_NAME}'.")
    return session


def preload_test_subset(
    data_dir: Path,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    _, test_dataset = get_cifar10_datasets(data_dir=data_dir, download=True)
    if num_samples > len(test_dataset):
        raise ValueError(
            f"Requested {num_samples} samples, but CIFAR-10 test set has "
            f"{len(test_dataset)} samples."
        )

    rng = np.random.default_rng(seed)
    selected_indices = np.sort(
        rng.choice(len(test_dataset), size=num_samples, replace=False)
    )

    images: list[np.ndarray] = []
    labels: list[int] = []
    for index in selected_indices:
        image, label = test_dataset[int(index)]
        images.append(image.cpu().numpy())
        labels.append(int(label))

    all_images = np.ascontiguousarray(np.stack(images).astype(np.float32, copy=False))
    all_labels = np.array(labels, dtype=np.int64)
    batches = [
        all_images[start : start + batch_size]
        for start in range(0, num_samples, batch_size)
    ]
    return batches, all_labels


def build_onnx_feeds(batches: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
    return [{INPUT_NAME: batch} for batch in batches]


def build_triton_requests(batches: list[np.ndarray]) -> list[TritonRequest]:
    requests: list[TritonRequest] = []
    for batch in batches:
        infer_input = httpclient.InferInput(INPUT_NAME, batch.shape, "FP32")
        infer_input.set_data_from_numpy(batch)
        requested_output = httpclient.InferRequestedOutput(OUTPUT_NAME)
        requests.append(
            TritonRequest(
                inputs=[infer_input],
                outputs=[requested_output],
                batch_size=int(batch.shape[0]),
            )
        )
    return requests


def take_cycled(items: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    iterator = cycle(items)
    return [next(iterator) for _ in range(count)]


def predict_onnx(
    session: ort.InferenceSession,
    feeds: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, RequestStats]:
    stats = RequestStats()
    predictions: list[np.ndarray] = []
    for feed in feeds:
        logits = session.run([OUTPUT_NAME], feed)[0]
        predictions.append(np.argmax(logits, axis=1))
        stats.record_success()
    return np.concatenate(predictions), stats


def predict_triton(
    client: httpclient.InferenceServerClient,
    model_name: str,
    requests: list[TritonRequest],
) -> tuple[np.ndarray, RequestStats]:
    stats = RequestStats()
    predictions: list[np.ndarray] = []
    for request in requests:
        try:
            response = client.infer(
                model_name=model_name,
                inputs=request.inputs,
                outputs=request.outputs,
            )
            logits = response.as_numpy(OUTPUT_NAME)
            if logits is None:
                raise RuntimeError(f"Triton response missing output '{OUTPUT_NAME}'.")
            predictions.append(np.argmax(logits, axis=1))
            stats.record_success()
        except (InferenceServerException, Exception) as exc:
            print(f"Triton accuracy request failed: {exc}")
            predictions.append(np.full(request.batch_size, -1, dtype=np.int64))
            stats.record_failure()
    return np.concatenate(predictions), stats


def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions == labels))


def summarize_latencies(
    latencies_ms: list[float],
    successful_samples: int,
    total_measured_seconds: float,
) -> dict[str, float | None]:
    if not latencies_ms:
        return {
            "avg_latency_ms": None,
            "median_latency_ms": None,
            "p95_latency_ms": None,
            "throughput_samples_per_sec": 0.0,
        }

    latencies = np.array(latencies_ms, dtype=np.float64)
    throughput = (
        successful_samples / total_measured_seconds
        if total_measured_seconds > 0
        else 0.0
    )
    return {
        "avg_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_samples_per_sec": float(throughput),
    }


def benchmark_onnx_latency(
    session: ort.InferenceSession,
    feeds: list[dict[str, np.ndarray]],
    warmup_runs: int,
    measured_runs: int,
) -> tuple[dict[str, float | None], RequestStats]:
    stats = RequestStats()
    for feed in take_cycled(feeds, warmup_runs):
        session.run([OUTPUT_NAME], feed)
        stats.record_success()

    latencies_ms: list[float] = []
    successful_samples = 0
    total_measured_seconds = 0.0
    for feed in take_cycled(feeds, measured_runs):
        batch_size = int(feed[INPUT_NAME].shape[0])
        start_time = time.perf_counter()
        session.run([OUTPUT_NAME], feed)
        elapsed_seconds = time.perf_counter() - start_time
        total_measured_seconds += elapsed_seconds
        latencies_ms.append(elapsed_seconds * 1000)
        successful_samples += batch_size
        stats.record_success()

    metrics = summarize_latencies(
        latencies_ms=latencies_ms,
        successful_samples=successful_samples,
        total_measured_seconds=total_measured_seconds,
    )
    return metrics, stats


def benchmark_triton_latency(
    client: httpclient.InferenceServerClient,
    model_name: str,
    requests: list[TritonRequest],
    warmup_runs: int,
    measured_runs: int,
) -> tuple[dict[str, float | None], RequestStats]:
    stats = RequestStats()
    for request in take_cycled(requests, warmup_runs):
        try:
            client.infer(
                model_name=model_name,
                inputs=request.inputs,
                outputs=request.outputs,
            )
            stats.record_success()
        except (InferenceServerException, Exception) as exc:
            print(f"Triton warm-up request failed: {exc}")
            stats.record_failure()

    latencies_ms: list[float] = []
    successful_samples = 0
    total_measured_seconds = 0.0
    for request in take_cycled(requests, measured_runs):
        start_time = time.perf_counter()
        try:
            client.infer(
                model_name=model_name,
                inputs=request.inputs,
                outputs=request.outputs,
            )
            elapsed_seconds = time.perf_counter() - start_time
            total_measured_seconds += elapsed_seconds
            latencies_ms.append(elapsed_seconds * 1000)
            successful_samples += request.batch_size
            stats.record_success()
        except (InferenceServerException, Exception) as exc:
            elapsed_seconds = time.perf_counter() - start_time
            total_measured_seconds += elapsed_seconds
            print(f"Triton measured request failed: {exc}")
            stats.record_failure()

    metrics = summarize_latencies(
        latencies_ms=latencies_ms,
        successful_samples=successful_samples,
        total_measured_seconds=total_measured_seconds,
    )
    return metrics, stats


def combine_stats(*stats_items: RequestStats) -> RequestStats:
    combined = RequestStats()
    for stats in stats_items:
        combined.successful_requests += stats.successful_requests
        combined.failed_requests += stats.failed_requests
    return combined


def build_result_row(
    model_format: str,
    accuracy: float,
    latency_metrics: dict[str, float | None],
    request_stats: RequestStats,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_format": model_format,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "avg_latency_ms": latency_metrics["avg_latency_ms"],
        "median_latency_ms": latency_metrics["median_latency_ms"],
        "p95_latency_ms": latency_metrics["p95_latency_ms"],
        "throughput_samples_per_sec": latency_metrics["throughput_samples_per_sec"],
        "request_success_rate": request_stats.success_rate,
        "request_success_percent": request_stats.success_rate * 100,
        "successful_requests": request_stats.successful_requests,
        "failed_requests": request_stats.failed_requests,
        "total_requests": request_stats.total_requests,
        "model_size_mb": get_file_size_mb(args.onnx_path),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "url": args.url,
        "model_name": args.model_name,
        "onnx_path": str(args.onnx_path),
        "seed": args.seed,
    }


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
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    return output_path


def format_metric(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def print_results(results: list[dict[str, Any]]) -> None:
    print()
    print(
        "Model Format              Batch Samples Accuracy   Avg ms/batch   "
        "Median ms   P95 ms   Samples/sec  Success"
    )
    print("-" * 112)
    for result in results:
        print(
            f"{result['model_format']:<24} "
            f"{result['batch_size']:>5} "
            f"{result['num_samples']:>7} "
            f"{result['accuracy_percent']:>8.2f}% "
            f"{format_metric(result['avg_latency_ms']):>14} "
            f"{format_metric(result['median_latency_ms']):>11} "
            f"{format_metric(result['p95_latency_ms']):>8} "
            f"{result['throughput_samples_per_sec']:>13.2f} "
            f"{result['request_success_percent']:>7.2f}%"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    print("Loading local ONNX Runtime session...")
    onnx_session = load_onnx_session(args.onnx_path)
    triton_client = httpclient.InferenceServerClient(url=args.url)

    print("Preloading CIFAR-10 test subset and prebuilding batches...")
    numpy_batches, labels = preload_test_subset(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    onnx_feeds = build_onnx_feeds(numpy_batches)
    triton_requests = build_triton_requests(numpy_batches)
    print(
        f"Prepared {args.num_samples} samples as {len(numpy_batches)} batches "
        f"of up to {args.batch_size}."
    )

    print("Computing local ONNX Runtime accuracy...")
    onnx_predictions, onnx_accuracy_stats = predict_onnx(onnx_session, onnx_feeds)
    onnx_accuracy = calculate_accuracy(onnx_predictions, labels)

    print("Computing Triton-served ONNX accuracy...")
    triton_predictions, triton_accuracy_stats = predict_triton(
        client=triton_client,
        model_name=args.model_name,
        requests=triton_requests,
    )
    triton_accuracy = calculate_accuracy(triton_predictions, labels)

    print("Measuring local ONNX Runtime latency...")
    onnx_latency_metrics, onnx_latency_stats = benchmark_onnx_latency(
        session=onnx_session,
        feeds=onnx_feeds,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )

    print("Measuring Triton-served ONNX latency...")
    triton_latency_metrics, triton_latency_stats = benchmark_triton_latency(
        client=triton_client,
        model_name=args.model_name,
        requests=triton_requests,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )

    results = [
        build_result_row(
            model_format="Local ONNX Runtime FP32",
            accuracy=onnx_accuracy,
            latency_metrics=onnx_latency_metrics,
            request_stats=combine_stats(onnx_accuracy_stats, onnx_latency_stats),
            args=args,
        ),
        build_result_row(
            model_format="Triton-served ONNX FP32",
            accuracy=triton_accuracy,
            latency_metrics=triton_latency_metrics,
            request_stats=combine_stats(triton_accuracy_stats, triton_latency_stats),
            args=args,
        ),
    ]

    all_results = merge_results(
        new_results=results,
        json_path=DEFAULT_JSON_PATH,
        append_results=args.append_results,
    )
    csv_path = save_csv(all_results, DEFAULT_CSV_PATH)
    json_path = save_json(all_results, DEFAULT_JSON_PATH)

    print_results(all_results)
    print()
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved JSON results to {json_path}")


if __name__ == "__main__":
    main()
