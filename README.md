# Birdly v2

**Bird species classification with ConvNeXt-Tiny, optimized inference, and cloud and edge deployment.**

[**Live Application**](https://www.birdlyus.com/)

Birdly is a bird species classifier that I originally built in 2022. In 2026, I rebuilt it with a new data pipeline, a ConvNeXt-Tiny model, and a complete inference service.

The model identifies **236 bird species found in New York State** and achieves **86.59% top-1 accuracy** on a test set of 4,712 images.

<img width="1802" height="1090" alt="image" src="https://github.com/user-attachments/assets/5824cba4-1ba7-4a7e-8acc-4d18fc0d9a5f" />
<img width="1828" height="766" alt="image" src="https://github.com/user-attachments/assets/fd75d4be-45a0-4015-bfd5-3937cc112d1f" />

## Architecture

<img width="1310" height="960" alt="image" src="https://github.com/user-attachments/assets/c35b6eac-17f4-4294-b00a-f1aacac546e6" />

The system consists of:

- **Data:** iNaturalist Open Data, Polars, Parquet, and DVC.
- **Training:** PyTorch, ConvNeXt-Tiny, and MLflow.
- **Inference:** PyTorch, ONNX Runtime, TensorRT, FP16 and INT8 experiments.
- **Serving:** FastAPI on Google Cloud Run, Next.js on Vercel, and a separate Raspberry Pi 5 deployment.

The data pipeline filters observations by taxonomy, geographic region, image quality, and license. It also applies per-species sampling limits and validates the final dataset.

## Results

### Model quality

| Metric | Result |
| --- | ---: |
| Top-1 accuracy | **86.59%** |
| Top-5 accuracy | **96.75%** |
| Macro F1 | **0.8656** |
| MCC | 0.8653 |
| Test images | 4,712 |

### Inference performance

I compared multiple CPU and GPU inference backends, measuring model-only and end-to-end performance separately.

| Experiment | Result |
| --- | --- |
| ONNX Runtime CPU | **63.87 ms** p50 at batch 1, 4 threads |
| CPU improvement over PyTorch | **~32% lower latency** |
| TensorRT FP16 | **3.62 ms** model-only p50 at batch 1 |
| TensorRT FP16 throughput | **730.66 images/s** at batch 32 |
| TensorRT end-to-end throughput | **117.81 images/s** at batch 32 |

TensorRT provided the best GPU performance. However, end-to-end throughput was limited by preprocessing and other non-model operations. Static INT8 quantization reduced accuracy without providing a useful CPU speedup.

I selected **ONNX Runtime FP32** for the deployed CPU service.

## Deployment

### Google Cloud Run

The FastAPI service runs on Cloud Run with 2 vCPUs, 2 GiB RAM, and scale-to-zero enabled. It sustained approximately **11.5 requests/s** in load testing.

The API includes image validation, rate limiting, bounded concurrency, request logging, and reliability tests.

### Raspberry Pi 5

I also deployed the same FP32 model on a **Raspberry Pi 5 with 1 GB RAM**, using ONNX Runtime on ARM64 and a systemd-managed FastAPI service.

The Pi sustained **3.65 successful requests/s** with two concurrent clients. Single-client p50 latency was **284.53 ms**. The service rejects requests above its configured admission limit rather than allowing unbounded concurrent inference.

## Repository

| Directory | Contents |
| --- | --- |
| `src/birdly/data/` | Data pipeline and validation |
| `src/birdly/training/` | Training and evaluation |
| `src/birdly/inference/` | Inference backends and benchmarks |
| `src/birdly/api/` | FastAPI service |
| `configs/` | Experiment and service configurations |
| `deploy/` | Cloud and Raspberry Pi deployment |
| `artifacts/` | Evaluation and benchmark results |

## Getting started

```bash
git clone https://github.com/ArjunBasandrai/birdly.git
cd birdly
uv sync
uv run birdly --help
```

Reproducing the full pipeline requires the iNaturalist source data, local configuration, and model artifacts. The DVC pipeline is defined in `dvc.yaml`.

For Raspberry Pi installation, see [`deploy/pi5/README.md`](deploy/pi5/README.md).

## License

Licensed under [GPL-3.0](LICENSE). Dataset images retain their original licenses.
