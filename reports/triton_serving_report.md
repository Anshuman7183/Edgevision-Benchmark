# EdgeVision TritonServe Report

## Project Goal

EdgeVision TritonServe extends the local EdgeVision ONNX inference pipeline by serving the exported CIFAR-10 CNN through NVIDIA Triton Inference Server. The goal is not to prove Triton is faster on a single local CPU run. The goal is to show the model can be packaged in a Triton model repository, loaded by a Dockerized server, queried through an HTTP client, checked for readiness, and benchmarked against local ONNX Runtime.

In this benchmark, local ONNX Runtime was faster than Triton-served ONNX. Triton still successfully served the ONNX model with a 100% request success rate in the saved runs.

## What Triton Adds

Triton adds a serving layer around the existing ONNX model:

- Docker-based inference server
- Versioned model repository
- `config.pbtxt` model contract
- HTTP health and readiness checks
- HTTP client inference with `tritonclient.http`
- Server-based benchmark script compared against local ONNX Runtime

This run used CPU-based Triton serving because no NVIDIA driver/GPU was detected locally. `nvidia-smi` was not available in the environment.

## Model Repository

```text
triton_model_repo/
\---edgevision_cnn/
    |   config.pbtxt
    \---1/
            model.onnx
```

The Triton config uses the ONNX Runtime backend:

```text
name: "edgevision_cnn"
platform: "onnxruntime_onnx"
max_batch_size: 8
input:  name "input",  dims [3, 32, 32]
output: name "logits", dims [10]
```

## Commands

Start Triton from Windows PowerShell:

```powershell
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v ${PWD}\triton_model_repo:/models `
  nvcr.io/nvidia/tritonserver:24.10-py3 `
  tritonserver --model-repository=/models
```

Check server and model readiness:

```powershell
python src/triton_health_check.py --url localhost:8000 --model-name edgevision_cnn
```

Run one HTTP inference request:

```powershell
python src/triton_client.py --url localhost:8000 --model-name edgevision_cnn --sample-index 0 --data-dir data
```

Benchmark batch size 1:

```powershell
python src/triton_benchmark.py --url localhost:8000 --model-name edgevision_cnn --onnx-path models/cnn_cifar10.onnx --data-dir data --num-samples 500 --batch-size 1 --warmup-runs 20 --measured-runs 100 --seed 42
```

Benchmark batch size 8 and append to the same results files:

```powershell
python src/triton_benchmark.py --url localhost:8000 --model-name edgevision_cnn --onnx-path models/cnn_cifar10.onnx --data-dir data --num-samples 500 --batch-size 8 --warmup-runs 20 --measured-runs 100 --seed 42 --append-results
```

## Benchmark Methodology

The benchmark compares local ONNX Runtime FP32 against Triton-served ONNX FP32 using the same fixed CIFAR-10 test subset. The script preloads samples and labels before timing and prebuilds batches before timing.

Latency is measured only around `onnxruntime.InferenceSession.run()` or `tritonclient.http.InferenceServerClient.infer()` calls using `time.perf_counter()`. The benchmark excludes Triton server startup time, dataset download/loading time, model repository setup, preprocessing outside the preloaded subset, and result serialization. Warm-up requests run before measured requests.

Throughput is computed as samples processed divided by measured inference time. Triton failures are counted and reported through request success rate.

## Results

Saved results from `results/triton_benchmark_results.csv` and `results/triton_benchmark_results.json`:

| Runtime | Batch | Samples | Accuracy | Avg ms/batch | Median ms | P95 ms | Throughput samples/sec | Request success | Model MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Local ONNX Runtime FP32 | 1 | 500 | 77.60% | 0.491 | 0.060 | 0.152 | 2038.62 | 100.00% | 0.60 |
| Triton-served ONNX FP32 | 1 | 500 | 77.60% | 1.352 | 0.652 | 5.696 | 739.44 | 100.00% | 0.60 |
| Local ONNX Runtime FP32 | 8 | 500 | 77.60% | 0.670 | 0.161 | 1.602 | 11875.46 | 100.00% | 0.60 |
| Triton-served ONNX FP32 | 8 | 500 | 77.60% | 25.356 | 8.537 | 54.619 | 313.92 | 100.00% | 0.60 |

## Observations

- Triton successfully served the ONNX model for both tested batch sizes with 100% request success.
- Local ONNX Runtime was faster in this local CPU benchmark for both batch size 1 and batch size 8.
- Accuracy matched between local ONNX Runtime and Triton because both executed the same FP32 ONNX model on the same fixed subset.
- Triton added HTTP/server overhead in this setup. That overhead is expected in a local single-client CPU benchmark and should not be described as a model-speed improvement.
- The value of this extension is deployment structure: model repository, server lifecycle, health checks, client API, and benchmark comparison.

## Limitations

- CPU-based Triton serving only; no NVIDIA driver/GPU was detected locally.
- The benchmark uses a small CIFAR-10 CNN and a 500-sample subset.
- Results are local-machine measurements and can vary with CPU load, Docker Desktop behavior, power mode, and background processes.
- The benchmark uses one client process and does not test concurrent clients, dynamic batching, gRPC, or GPU acceleration.
- Only the FP32 ONNX model is served through Triton in this report.

## Troubleshooting

- If the health check fails, confirm Docker is running and Triton is still serving on `localhost:8000`.
- If the model is not ready, check that `triton_model_repo/edgevision_cnn/1/model.onnx` exists and that `config.pbtxt` uses input `input` and output `logits`.
- If the Docker command fails on Windows PowerShell, run it from the repository root so `${PWD}\triton_model_repo:/models` points at the model repository.
- If CIFAR-10 loading fails, confirm the `data/` directory contains the CIFAR-10 files or allow the existing data loader to download them before benchmarking.
- If ports are already in use, stop the previous Triton container or change the exposed ports consistently in the commands.

## Future Improvements

- Run the same Triton benchmark on an NVIDIA GPU-enabled machine.
- Add gRPC benchmarking and compare it with HTTP.
- Test concurrent clients and Triton dynamic batching.
- Add Triton `perf_analyzer` runs for a server-side view of throughput.
- Try serving the INT8 ONNX model only after validating backend compatibility.
- Record CPU, RAM, Docker, Triton, ONNX Runtime, and OS versions beside each benchmark run.
