# EdgeVision Benchmark Report

## Project Summary

EdgeVision Benchmark is a CPU-friendly ML systems pipeline that trains a lightweight CIFAR-10 CNN in PyTorch, exports it to ONNX, applies ONNX Runtime dynamic INT8 quantization, and compares runtime behavior across PyTorch FP32, ONNX FP32, and ONNX INT8 formats.

## Hardware and Environment Note

Environment details should be recorded before publishing final numbers, for example CPU model, RAM, operating system, Python version, PyTorch version, ONNX Runtime version, and whether other heavy background processes were running.

## Full Test-Set Evaluation

This result is separate from the benchmark subset accuracy below.

- Model path: `models\cnn_cifar10.pth`
- Dataset: CIFAR-10
- Samples: 10000
- Correct predictions: 7384
- Accuracy: 73.84%
- Evaluation batch size: 64
- Device: cpu
- Seed: 42

## Benchmark Methodology

The benchmark used saved results for CIFAR-10 test subsets with 1000 samples and batch size(s) 1, 32. The benchmark preloads selected tensors and labels into memory before timing, performs warm-up inference runs, and measures only model forward/session execution with `time.perf_counter()`.

Model loading, dataset loading, preprocessing, and result serialization are excluded from measured inference latency. Latency is reported per batch, while throughput is reported in samples per second.

## Benchmark Results

| Model Format | Batch | Samples | Accuracy | Avg Latency ms/batch | Median ms | P95 ms | Throughput samples/sec | Size MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 1 | 1000 | 75.90% | 0.546 | 0.456 | 0.880 | 1831.22 | 0.61 |
| ONNX FP32 | 1 | 1000 | 75.90% | 0.073 | 0.067 | 0.080 | 13663.82 | 0.60 |
| ONNX INT8 | 1 | 1000 | 75.60% | 0.595 | 0.202 | 0.355 | 1680.70 | 0.16 |
| PyTorch FP32 | 32 | 1000 | 75.90% | 4.593 | 4.357 | 6.329 | 6809.90 | 0.61 |
| ONNX FP32 | 32 | 1000 | 75.90% | 1.767 | 0.683 | 5.140 | 17699.50 | 0.60 |
| ONNX INT8 | 32 | 1000 | 75.20% | 9.197 | 7.884 | 18.275 | 3401.26 | 0.16 |

## Accuracy Comparison

Benchmark subset accuracy is computed only on the selected subset, so it should not be confused with the full 10,000-image evaluation.

- Batch size 1: PyTorch FP32 led with 75.90%.
- Batch size 32: PyTorch FP32 led with 75.90%.

## Latency Comparison

Lower average batch latency is better. In the saved CPU benchmark, ONNX FP32 was the fastest measured format.

- Batch size 1: ONNX FP32 led with 0.073 ms/batch.
- Batch size 32: ONNX FP32 led with 1.767 ms/batch.

## Throughput Comparison

Higher throughput is better. Throughput is calculated from measured inference time and the number of samples processed during timed runs.

- Batch size 1: ONNX FP32 led with 13663.815 samples/sec.
- Batch size 32: ONNX FP32 led with 17699.496 samples/sec.

## Model Size Comparison

- ONNX INT8 reduced model size from 0.60 MB to 0.16 MB, a 73.37% reduction versus ONNX FP32.

## Key Observations

- ONNX FP32 was fastest in the measured CPU benchmark.
- ONNX INT8 significantly reduced model size, but it was slower than ONNX FP32 in these CPU measurements.
- PyTorch FP32 and ONNX FP32 matched benchmark subset accuracy in the saved runs.
- ONNX INT8 showed a small accuracy drop compared with FP32 on the benchmark subset.

## Quantization Trade-Offs

Dynamic quantization made the ONNX model much smaller, which is useful for storage and deployment packaging. However, the measured CPU latency did not improve in this run. This is a useful reminder that quantization benefits depend on model structure, operator coverage, runtime kernels, and target hardware.

This project uses ONNX Runtime dynamic quantization, which primarily affects supported weight-bearing operators such as Linear/Gemm. The results should not be described as full convolution quantization.

## Limitations

- Benchmarks were run on a local CPU environment, not on Qualcomm NPU or DSP hardware.
- CIFAR-10 is useful for a compact deployment pipeline demo, but it is not a production edge-vision workload.
- Latency can vary with CPU load, power mode, ONNX Runtime settings, and background processes.
- The INT8 model was produced with dynamic quantization rather than calibrated static quantization.
- Benchmark subset accuracy can differ from full test-set accuracy.

## Resume Relevance

This project demonstrates an end-to-end model deployment workflow: PyTorch training, independent evaluation, ONNX export, ONNX Runtime inference validation, INT8 quantization, latency benchmarking, throughput measurement, and model-size comparison. It is directly relevant to ML systems, model optimization, and edge AI software roles.
