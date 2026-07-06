# EdgeVision Benchmark: PyTorch to ONNX Quantized Inference Pipeline

EdgeVision Benchmark is a small ML systems project around one CIFAR-10 CNN: train it in PyTorch, export it to ONNX, quantize it, benchmark the local runtimes, then serve the FP32 ONNX model through NVIDIA Triton for comparison.

I kept the model and dataset intentionally small so the whole workflow is easy to inspect, rerun, and review without waiting on a large training job.

## What is included

### Local inference pipeline

- Train and evaluate a lightweight PyTorch CNN on CIFAR-10.
- Export the checkpoint to ONNX FP32.
- Validate ONNX Runtime output against PyTorch output.
- Apply ONNX Runtime dynamic INT8 quantization.
- Benchmark PyTorch FP32, ONNX FP32, and ONNX INT8 on CPU.

### Triton serving path

- Package the FP32 ONNX model in a Triton model repository.
- Serve it with NVIDIA Triton Inference Server in Docker.
- Check server liveness, server readiness, and model readiness.
- Send a single CIFAR-10 HTTP inference request with `tritonclient.http`.
- Benchmark Triton-served ONNX FP32 against local ONNX Runtime FP32.

### Reproducibility

- Keep small model artifacts committed for smoke tests and review.
- Save benchmark outputs as CSV/JSON.
- Keep Markdown reports beside the measured artifacts.
- Run a lightweight GitHub Actions smoke test without training or benchmarking.

## Pipeline

```text
CIFAR-10 -> PyTorch CNN -> ONNX FP32 -> ONNX INT8 -> Local Benchmark
                              |
                              +-> Triton Model Repo -> HTTP Inference -> Triton Benchmark
```

## Tech stack

- Python
- PyTorch
- torchvision
- ONNX
- ONNX Runtime
- NVIDIA Triton Inference Server
- Docker
- NumPy
- Pandas
- scikit-learn
- tqdm

The current scripts mainly use PyTorch, torchvision, ONNX, ONNX Runtime, Triton HTTP client, NumPy, CSV, and JSON. Pandas, scikit-learn, and tqdm are also pinned in `requirements.txt`.

## Folder structure

```text
Edgevision-Benchmark/
|   .gitignore
|   docker-compose.yml
|   README.md
|   requirements.txt
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
|       triton_serving_report.md
|
+---results/
|       .gitkeep
|       benchmark_results.csv
|       benchmark_results.json
|       evaluation_results.json
|       triton_benchmark_results.csv
|       triton_benchmark_results.json
|
+---src/
|       benchmark.py
|       data_loader.py
|       evaluate.py
|       export_onnx.py
|       generate_report.py
|       model.py
|       onnx_inference.py
|       quantize.py
|       train.py
|       triton_benchmark.py
|       triton_client.py
|       triton_health_check.py
|       utils.py
|
+---tests/
|       smoke_test.py
|
\---triton_model_repo/
    \---edgevision_cnn/
        |   config.pbtxt
        \---1/
                model.onnx
```

Artifact policy:

- `data/` is local and ignored. The scripts can download CIFAR-10 when needed, or you can place the archive there manually.
- Small model artifacts in `models/` and `triton_model_repo/` are committed so smoke tests and review can run without retraining.
- Benchmark results and Markdown reports are committed as evidence for the measured local runs.

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

## CI smoke test

GitHub Actions runs `tests/smoke_test.py` through `.github/workflows/ci.yml`. The smoke test is intentionally lightweight: it checks core files and artifacts without training, downloading CIFAR-10, starting Triton, or running benchmarks.

## EdgeVision TritonServe

The Triton path serves the exported FP32 ONNX model from a standard model repository. It uses Docker for the server, `config.pbtxt` for the model contract, HTTP checks for readiness, and a small Python client for inference.

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

The saved Triton runs completed with 100% request success. Local ONNX Runtime was still faster in this local CPU setup, which is expected for a small model and single-client HTTP requests.

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

This CNN is small and convolution-heavy, so INT8 did not improve latency in this CPU test. Quantization results depend on model structure, runtime kernels, operator coverage, and the hardware being measured.

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

## What I learned

The useful work started after the model checkpoint existed: exporting, validating, quantizing, serving, and measuring each runtime with the same inputs.

ONNX FP32 was the fastest local runtime in these CPU results. Dynamic INT8 made the model much smaller, but it was slower on this machine and had a small accuracy drop.

Triton successfully served the ONNX model and gave a clean deployment shape: model repository, Docker server, health check, HTTP client, and benchmark script. In this CPU-only setup, Triton was slower than local ONNX Runtime, but it made the serving boundary explicit and testable.

## Author
Anshuman Anand Nayak
