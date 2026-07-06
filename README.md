# EdgeVision Benchmark: PyTorch to ONNX Quantized Inference Pipeline

This project trains a lightweight CNN on CIFAR-10, exports it to ONNX, applies ONNX Runtime dynamic INT8 quantization, and benchmarks PyTorch FP32 vs ONNX FP32 vs ONNX INT8 on CPU.

## Why I built this

I built this to understand what happens after a model is trained. Training is only one part of the workflow; the model also has to be exported, validated in another runtime, quantized, benchmarked, and compared honestly.

The goal was to keep the model small and the pipeline reproducible, then measure the trade-offs instead of assuming quantization would automatically make everything faster.

## What this project does

- Trains a lightweight PyTorch CNN on CIFAR-10
- Evaluates PyTorch accuracy
- Exports the model to ONNX
- Validates ONNX output against PyTorch output
- Applies ONNX Runtime dynamic INT8 quantization
- Benchmarks PyTorch FP32, ONNX FP32, and ONNX INT8
- Saves CSV, JSON, and Markdown reports

## Pipeline

```text
CIFAR-10 -> PyTorch CNN -> .pth checkpoint -> ONNX FP32 -> ONNX INT8 -> Benchmark -> Report
```

## Tech stack

- Python
- PyTorch
- torchvision
- ONNX
- ONNX Runtime
- NumPy
- Pandas
- scikit-learn
- tqdm

The current scripts mainly use PyTorch, torchvision, ONNX, ONNX Runtime, NumPy, CSV, and JSON. Pandas, scikit-learn, and tqdm are included in `requirements.txt` for analysis/reporting utilities and likely follow-up work.

## Folder structure

```text
Edgevision-Benchmark/
|   .gitignore
|   PRD.md
|   PROJECT_CONTEXT.md
|   README.md
|   requirements.txt
|
+---data/
|   |   cifar-10-python.tar.gz
|   |
|   \---cifar-10-batches-py/
|           batches.meta
|           data_batch_1
|           data_batch_2
|           data_batch_3
|           data_batch_4
|           data_batch_5
|           readme.html
|           test_batch
|
+---models/
|       .gitkeep
|       cnn_cifar10.onnx
|       cnn_cifar10.pth
|       cnn_cifar10_int8.onnx
|
+---notebooks/
+---reports/
|       .gitkeep
|       benchmark_report.md
|
+---results/
|       .gitkeep
|       benchmark_results.csv
|       benchmark_results.json
|       evaluation_results.json
|
\---src/
        benchmark.py
        data_loader.py
        evaluate.py
        export_onnx.py
        generate_report.py
        model.py
        onnx_inference.py
        quantize.py
        train.py
        utils.py
```

`data/`, `models/*.pth`, `models/*.onnx`, and generated result/report files are produced by running the pipeline.

## Setup on Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If CIFAR-10 download is slow, manually place `cifar-10-python.tar.gz` inside the `data/` folder.

## Commands

Train the PyTorch model:

```powershell
python src/train.py --epochs 5 --batch-size 64
```

Evaluate the PyTorch checkpoint:

```powershell
python src/evaluate.py
```

Export to ONNX:

```powershell
python src/export_onnx.py
```

Validate ONNX Runtime output against PyTorch:

```powershell
python src/onnx_inference.py --batch-size 8
```

Quantize the ONNX model with dynamic INT8 quantization:

```powershell
python src/quantize.py
```

Run benchmark for single-image inference:

```powershell
python src/benchmark.py --num-samples 1000 --batch-size 1 --warmup-runs 20 --measured-runs 100
```

Run benchmark for batched inference and append to the same result files:

```powershell
python src/benchmark.py --num-samples 1000 --batch-size 32 --warmup-runs 20 --measured-runs 100 --append-results
```

Generate the Markdown report:

```powershell
python src/generate_report.py
```

## EdgeVision TritonServe

This repo also includes a practical Triton serving path for the exported FP32 ONNX model. Triton adds a Docker server, versioned model repository, `config.pbtxt`, HTTP health check, HTTP client inference, and a benchmark script that compares server-based inference against local ONNX Runtime.

```text
triton_model_repo/
\---edgevision_cnn/
    |   config.pbtxt
    \---1/
            model.onnx
```

Start NVIDIA Triton Inference Server from Windows PowerShell:

```powershell
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v ${PWD}\triton_model_repo:/models `
  nvcr.io/nvidia/tritonserver:24.10-py3 `
  tritonserver --model-repository=/models
```

Check readiness and run one sample:

```powershell
python src/triton_health_check.py --url localhost:8000 --model-name edgevision_cnn
python src/triton_client.py --url localhost:8000 --model-name edgevision_cnn --sample-index 0 --data-dir data
```

Run the Triton comparison benchmark:

```powershell
python src/triton_benchmark.py --url localhost:8000 --model-name edgevision_cnn --onnx-path models/cnn_cifar10.onnx --data-dir data --num-samples 500 --batch-size 1 --warmup-runs 20 --measured-runs 100 --seed 42
python src/triton_benchmark.py --url localhost:8000 --model-name edgevision_cnn --onnx-path models/cnn_cifar10.onnx --data-dir data --num-samples 500 --batch-size 8 --warmup-runs 20 --measured-runs 100 --seed 42 --append-results
```

Saved Triton results show the ONNX model was served successfully with 100% request success. Local ONNX Runtime was faster than Triton-served ONNX in this local CPU benchmark. Triton is useful here as a serving layer, not as a claimed speedup.

| Runtime | Batch | Samples | Accuracy | Avg ms/batch | P95 ms | Throughput samples/sec | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local ONNX Runtime FP32 | 1 | 500 | 77.60% | 0.491 | 0.152 | 2038.62 | 100.00% |
| Triton-served ONNX FP32 | 1 | 500 | 77.60% | 1.352 | 5.696 | 739.44 | 100.00% |
| Local ONNX Runtime FP32 | 8 | 500 | 77.60% | 0.670 | 1.602 | 11875.46 | 100.00% |
| Triton-served ONNX FP32 | 8 | 500 | 77.60% | 25.356 | 54.619 | 313.92 | 100.00% |

The Triton benchmark excludes server startup and dataset loading time. This was CPU-based Triton serving because no NVIDIA driver/GPU was detected locally. Full notes are in `reports/triton_serving_report.md`.

## Evaluation result

Full PyTorch test-set evaluation from `results/evaluation_results.json`:

- Samples: 10,000 CIFAR-10 test images
- Correct predictions: 7,384
- Accuracy: 73.84%
- Device: CPU

This is the full test-set result. The benchmark table below uses a 1,000-sample subset, so those accuracies are separate measurements.

## Benchmark methodology

- The same CIFAR-10 test subset is used for PyTorch FP32, ONNX FP32, and ONNX INT8.
- Selected tensors and labels are preloaded before timing.
- Model loading and dataset loading are excluded from latency.
- Warm-up runs are executed before measured runs.
- `time.perf_counter()` is used for timing.
- Latency is measured only around actual PyTorch forward calls or ONNX Runtime session calls.
- Batch size 1 represents single-image inference.
- Batch size 32 represents batched throughput.

## Benchmark results

Saved benchmark results from `results/benchmark_results.csv` and `results/benchmark_results.json`:

| Model Format | Batch Size | Samples | Accuracy | Avg Latency ms/batch | Median ms | P95 ms | Throughput samples/sec | Size MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 1 | 1000 | 75.90% | 0.546 | 0.456 | 0.880 | 1831.22 | 0.61 |
| ONNX FP32 | 1 | 1000 | 75.90% | 0.073 | 0.067 | 0.080 | 13663.82 | 0.60 |
| ONNX INT8 | 1 | 1000 | 75.60% | 0.595 | 0.202 | 0.355 | 1680.70 | 0.16 |
| PyTorch FP32 | 32 | 1000 | 75.90% | 4.593 | 4.357 | 6.329 | 6809.90 | 0.61 |
| ONNX FP32 | 32 | 1000 | 75.90% | 1.767 | 0.683 | 5.140 | 17699.50 | 0.60 |
| ONNX INT8 | 32 | 1000 | 75.20% | 9.197 | 7.884 | 18.275 | 3401.26 | 0.16 |

## Key observations

- ONNX FP32 matched PyTorch FP32 accuracy on the 1,000-sample benchmark subset.
- ONNX FP32 was the fastest runtime for both batch size 1 and batch size 32 in this CPU benchmark.
- ONNX INT8 reduced model size from about 0.60 MB to 0.16 MB, which is about a 73% reduction.
- ONNX INT8 had a small accuracy drop compared with FP32 on the benchmark subset.
- ONNX INT8 was slower than ONNX FP32 on this machine, so quantization mainly helped model size here, not latency.

## Quantization note

This project uses ONNX Runtime dynamic quantization. Dynamic quantization mainly targets supported weight-bearing operators such as `Linear`/`Gemm`.

This CNN is small and convolution-heavy, so INT8 did not improve latency in this CPU test. That is an important deployment trade-off rather than a project failure: quantization results depend on model structure, runtime kernels, operator coverage, and the hardware being measured.

## Limitations

- CPU-only benchmark
- No Qualcomm NPU/DSP testing
- CIFAR-10 is small and used for reproducibility
- Dynamic quantization only, not static calibration-based quantization
- Latency varies by hardware, OS, power mode, and background processes

## Future improvements

- Static quantization with calibration data
- Per-layer profiling
- Larger model comparison
- Hardware-specific benchmarking
- Charts for latency and model size
- GitHub Actions smoke tests

## Summary

I built EdgeVision Benchmark to practice the part of machine learning that happens after training. I trained a small CNN on CIFAR-10 in PyTorch, saved the checkpoint, exported the model to ONNX, validated that ONNX Runtime produced matching predictions, and then created a dynamically quantized INT8 ONNX version.

For benchmarking, I used the same preloaded CIFAR-10 test subset across all three formats and measured only the actual inference calls after warm-up. In the CPU benchmark, ONNX FP32 was the fastest runtime. INT8 reduced the model size by about 73%, but it was slower than ONNX FP32 and had a small accuracy drop. The main lesson was that quantization is a trade-off to measure, not something to assume will always improve latency.

## Author
Anshuman Anand Nayak
