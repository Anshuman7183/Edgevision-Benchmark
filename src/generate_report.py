"""Generate the Markdown benchmark report from saved result files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from utils import ensure_dir


DEFAULT_BENCHMARK_JSON = Path("results") / "benchmark_results.json"
DEFAULT_BENCHMARK_CSV = Path("results") / "benchmark_results.csv"
DEFAULT_EVALUATION_JSON = Path("results") / "evaluation_results.json"
DEFAULT_REPORT_PATH = Path("reports") / "benchmark_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark report.")
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=DEFAULT_BENCHMARK_JSON,
        help="Path to benchmark JSON results.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=DEFAULT_BENCHMARK_CSV,
        help="Path to benchmark CSV results.",
    )
    parser.add_argument(
        "--evaluation-json",
        type=Path,
        default=DEFAULT_EVALUATION_JSON,
        help="Path to full PyTorch evaluation JSON results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def load_benchmark_results(json_path: Path, csv_path: Path) -> list[dict[str, Any]]:
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {json_path}")
        return data

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return [normalize_csv_row(row) for row in reader]

    raise FileNotFoundError(
        f"No benchmark results found at {json_path} or {csv_path}"
    )


def normalize_csv_row(row: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = dict(row)
    float_fields = (
        "accuracy",
        "accuracy_percent",
        "avg_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "throughput_samples_per_sec",
        "model_size_mb",
    )
    int_fields = ("num_samples", "batch_size", "warmup_runs", "measured_runs")

    for field in float_fields:
        normalized[field] = float(normalized[field])
    for field in int_fields:
        normalized[field] = int(normalized[field])
    return normalized


def load_evaluation_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return data


def fmt_percent(value: float) -> str:
    return f"{value:.2f}%"


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def benchmark_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Model Format | Batch | Samples | Accuracy | Avg Latency ms/batch | "
        "Median ms | P95 ms | Throughput samples/sec | Size MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| {model_format} | {batch_size} | {num_samples} | {accuracy} | "
            "{avg} | {median} | {p95} | {throughput} | {size} |".format(
                model_format=row["model_format"],
                batch_size=row["batch_size"],
                num_samples=row["num_samples"],
                accuracy=fmt_percent(row["accuracy_percent"]),
                avg=fmt_float(row["avg_latency_ms"]),
                median=fmt_float(row["median_latency_ms"]),
                p95=fmt_float(row["p95_latency_ms"]),
                throughput=fmt_float(row["throughput_samples_per_sec"], 2),
                size=fmt_float(row["model_size_mb"], 2),
            )
        )
    return "\n".join(lines)


def grouped_by_batch(results: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(int(row["batch_size"]), []).append(row)
    return dict(sorted(grouped.items()))


def best_by_metric(
    rows: list[dict[str, Any]],
    metric: str,
    reverse: bool,
) -> dict[str, Any]:
    return sorted(rows, key=lambda row: row[metric], reverse=reverse)[0]


def build_comparison_section(
    title: str,
    results: list[dict[str, Any]],
    metric: str,
    unit: str,
    reverse: bool,
    description: str,
) -> str:
    lines = [f"## {title}", "", description, ""]
    for batch_size, rows in grouped_by_batch(results).items():
        best = best_by_metric(rows, metric, reverse=reverse)
        value = best[metric]
        if metric == "accuracy_percent":
            value_text = fmt_percent(value)
        elif unit:
            value_text = f"{fmt_float(value)} {unit}"
        else:
            value_text = fmt_float(value)
        lines.append(
            f"- Batch size {batch_size}: {best['model_format']} led with {value_text}."
        )
    return "\n".join(lines)


def model_size_observation(results: list[dict[str, Any]]) -> str:
    sizes: dict[str, float] = {}
    for row in results:
        sizes.setdefault(row["model_format"], float(row["model_size_mb"]))

    fp32_size = sizes.get("ONNX FP32")
    int8_size = sizes.get("ONNX INT8")
    if fp32_size is None or int8_size is None or fp32_size == 0:
        return "- Model size comparison is unavailable because expected ONNX rows are missing."

    reduction = ((fp32_size - int8_size) / fp32_size) * 100
    return (
        f"- ONNX INT8 reduced model size from {fp32_size:.2f} MB to "
        f"{int8_size:.2f} MB, a {reduction:.2f}% reduction versus ONNX FP32."
    )


def full_evaluation_section(evaluation: dict[str, Any] | None) -> str:
    lines = ["## Full Test-Set Evaluation", ""]
    if evaluation is None:
        lines.append(
            "Full PyTorch evaluation result was not found at "
            "`results/evaluation_results.json`."
        )
        return "\n".join(lines)

    lines.extend(
        [
            "This result is separate from the benchmark subset accuracy below.",
            "",
            f"- Model path: `{evaluation['model_path']}`",
            f"- Dataset: {evaluation['dataset']}",
            f"- Samples: {evaluation['num_samples']}",
            f"- Correct predictions: {evaluation['correct']}",
            f"- Accuracy: {fmt_percent(float(evaluation['accuracy_percent']))}",
            f"- Evaluation batch size: {evaluation['batch_size']}",
            f"- Device: {evaluation['device']}",
            f"- Seed: {evaluation['seed']}",
        ]
    )
    return "\n".join(lines)


def build_report(results: list[dict[str, Any]], evaluation: dict[str, Any] | None) -> str:
    batch_sizes = ", ".join(str(size) for size in grouped_by_batch(results))
    sample_counts = ", ".join(
        str(count) for count in sorted({int(row["num_samples"]) for row in results})
    )

    sections = [
        "# EdgeVision Benchmark Report",
        "",
        "## Project Summary",
        "",
        (
            "EdgeVision Benchmark is a CPU-friendly ML systems pipeline that trains "
            "a lightweight CIFAR-10 CNN in PyTorch, exports it to ONNX, applies "
            "ONNX Runtime dynamic INT8 quantization, and compares runtime behavior "
            "across PyTorch FP32, ONNX FP32, and ONNX INT8 formats."
        ),
        "",
        "## Hardware and Environment Note",
        "",
        (
            "Environment details should be recorded before publishing final numbers, "
            "for example CPU model, RAM, operating system, Python version, PyTorch "
            "version, ONNX Runtime version, and whether other heavy background "
            "processes were running."
        ),
        "",
        full_evaluation_section(evaluation),
        "",
        "## Benchmark Methodology",
        "",
        (
            f"The benchmark used saved results for CIFAR-10 test subsets with "
            f"{sample_counts} samples and batch size(s) {batch_sizes}. The benchmark "
            "preloads selected tensors and labels into memory before timing, performs "
            "warm-up inference runs, and measures only model forward/session execution "
            "with `time.perf_counter()`."
        ),
        "",
        (
            "Model loading, dataset loading, preprocessing, and result serialization "
            "are excluded from measured inference latency. Latency is reported per "
            "batch, while throughput is reported in samples per second."
        ),
        "",
        "## Benchmark Results",
        "",
        benchmark_table(results),
        "",
        build_comparison_section(
            title="Accuracy Comparison",
            results=results,
            metric="accuracy_percent",
            unit="",
            reverse=True,
            description=(
                "Benchmark subset accuracy is computed only on the selected subset, "
                "so it should not be confused with the full 10,000-image evaluation."
            ),
        ),
        "",
        build_comparison_section(
            title="Latency Comparison",
            results=results,
            metric="avg_latency_ms",
            unit="ms/batch",
            reverse=False,
            description=(
                "Lower average batch latency is better. In the saved CPU benchmark, "
                "ONNX FP32 was the fastest measured format."
            ),
        ),
        "",
        build_comparison_section(
            title="Throughput Comparison",
            results=results,
            metric="throughput_samples_per_sec",
            unit="samples/sec",
            reverse=True,
            description=(
                "Higher throughput is better. Throughput is calculated from measured "
                "inference time and the number of samples processed during timed runs."
            ),
        ),
        "",
        "## Model Size Comparison",
        "",
        model_size_observation(results),
        "",
        "## Key Observations",
        "",
        "- ONNX FP32 was fastest in the measured CPU benchmark.",
        (
            "- ONNX INT8 significantly reduced model size, but it was slower than "
            "ONNX FP32 in these CPU measurements."
        ),
        (
            "- PyTorch FP32 and ONNX FP32 matched benchmark subset accuracy in the "
            "saved runs."
        ),
        (
            "- ONNX INT8 showed a small accuracy drop compared with FP32 on the "
            "benchmark subset."
        ),
        "",
        "## Quantization Trade-Offs",
        "",
        (
            "Dynamic quantization made the ONNX model much smaller, which is useful "
            "for storage and deployment packaging. However, the measured CPU latency "
            "did not improve in this run. This is a useful reminder that quantization "
            "benefits depend on model structure, operator coverage, runtime kernels, "
            "and target hardware."
        ),
        "",
        (
            "This project uses ONNX Runtime dynamic quantization, which primarily "
            "affects supported weight-bearing operators such as Linear/Gemm. The "
            "results should not be described as full convolution quantization."
        ),
        "",
        "## Limitations",
        "",
        "- Benchmarks were run on a local CPU environment, not on Qualcomm NPU or DSP hardware.",
        "- CIFAR-10 is useful for a compact deployment pipeline demo, but it is not a production edge-vision workload.",
        "- Latency can vary with CPU load, power mode, ONNX Runtime settings, and background processes.",
        "- The INT8 model was produced with dynamic quantization rather than calibrated static quantization.",
        "- Benchmark subset accuracy can differ from full test-set accuracy.",
        "",
        "## Resume Relevance",
        "",
        (
            "This project demonstrates an end-to-end model deployment workflow: "
            "PyTorch training, independent evaluation, ONNX export, ONNX Runtime "
            "inference validation, INT8 quantization, latency benchmarking, throughput "
            "measurement, and model-size comparison. It is directly relevant to ML "
            "systems, model optimization, and edge AI software roles."
        ),
        "",
    ]
    return "\n".join(sections)


def write_report(report: str, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    benchmark_results = load_benchmark_results(
        json_path=args.benchmark_json,
        csv_path=args.benchmark_csv,
    )
    evaluation_result = load_evaluation_result(args.evaluation_json)
    report = build_report(benchmark_results, evaluation_result)
    output_path = write_report(report, args.output)
    print(f"Generated report at {output_path}")


if __name__ == "__main__":
    main()
