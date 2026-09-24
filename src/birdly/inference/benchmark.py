from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import psutil
import torch

from birdly.inference.predict import (
    InferenceBackendName,
    Predictor,
    load_predictor,
    predict_batch,
    preprocess_image,
)

DEFAULT_CUDA_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
DEFAULT_CPU_BATCH_SIZES = (1, 2, 4, 8, 16)
DEFAULT_CPU_THREAD_COUNTS = (1, 2, 4)
DEFAULT_MANIFEST_PATH = Path("data/processed/validated_manifest.parquet")
DEFAULT_IMAGE_ROOT = Path("B:/images")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/benchmark")


@dataclass
class BenchmarkResult:
    mode: str
    batch_size: int
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    throughput_images_per_second: float | None
    warmup_total_ms: float | None
    peak_gpu_memory_mib: float | None
    process_ram_mib: float | None
    precision: str
    device: str
    cpu_threads: int | None
    model_name: str
    backend: str
    status: str = "ok"
    error: str = ""


@dataclass(frozen=True)
class InferenceBackend:
    predictor: Predictor
    cpu_threads: int | None = None
    gpu_memory_baseline_mib: float | None = None

    @property
    def name(self) -> str:
        return self.predictor.backend

    @property
    def model_name(self) -> str:
        return self.predictor.model_name

    @property
    def precision(self) -> str:
        return self.predictor.precision

    def model_forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self.predictor.model(image_batch)

    def predict_paths(self, image_paths: Sequence[Path]) -> Any:
        return predict_batch(self.predictor, image_paths)


def load_benchmark_inputs(
    count: int,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    image_root: Path = DEFAULT_IMAGE_ROOT,
) -> list[Path]:
    rows = (
        pl.read_parquet(
            manifest_path,
            columns=["taxon_id", "split", "image_path"],
        )
        .filter(pl.col("split") == "test")
        .sort(["taxon_id", "image_path"])
        .unique("taxon_id", keep="first", maintain_order=True)
        .head(count)
    )
    paths = [image_root / value for value in rows["image_path"].to_list()]
    if len(paths) < count:
        raise ValueError(f"Only {len(paths)} representative images are available")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Benchmark image is missing: {missing[0]}")
    return paths


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "gpu_memory_used_mib": None,
        "gpu_memory_total_mib": None,
        "gpu_temperature_c": None,
        "gpu_utilization_percent": None,
        "warnings": [],
    }
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.splitlines()[0]
        name, used, total, temperature, utilization = (
            value.strip() for value in output.split(",")
        )
        environment.update(
            {
                "gpu_name": name,
                "gpu_memory_used_mib": int(used),
                "gpu_memory_total_mib": int(total),
                "gpu_temperature_c": int(temperature),
                "gpu_utilization_percent": int(utilization),
            }
        )
        if int(used) / int(total) >= 0.25 or int(utilization) >= 50:
            environment["warnings"].append(
                "GPU is substantially occupied; results can include unrelated load."
            )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        environment["warnings"].append("nvidia-smi telemetry is unavailable.")
    return environment


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_timer(call: Callable[[], Any], device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1000.0


def _end_to_end_timer(call: Callable[[], Any], device: torch.device) -> float:
    _sync(device)
    start = time.perf_counter()
    call()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _device_gpu_memory_used_mib(device: torch.device) -> float:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (total_bytes - free_bytes) / (1024**2)


def _peak_memory(backend: InferenceBackend) -> float | None:
    device = backend.predictor.device
    if device.type != "cuda":
        return None
    if backend.gpu_memory_baseline_mib is not None:
        return max(
            0.0,
            _device_gpu_memory_used_mib(device) - backend.gpu_memory_baseline_mib,
        )
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def _process_ram(device: torch.device) -> float | None:
    if device.type != "cpu":
        return None
    return psutil.Process().memory_info().rss / (1024**2)


def _summarize(
    mode: str,
    batch_size: int,
    latencies: Sequence[float],
    warmup_total_ms: float,
    backend: InferenceBackend,
) -> BenchmarkResult:
    mean = sum(latencies) / len(latencies)
    return BenchmarkResult(
        mode=mode,
        batch_size=batch_size,
        mean_latency_ms=mean,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        p99_latency_ms=_percentile(latencies, 0.99),
        throughput_images_per_second=batch_size * 1000.0 / mean,
        warmup_total_ms=warmup_total_ms,
        peak_gpu_memory_mib=_peak_memory(backend),
        process_ram_mib=_process_ram(backend.predictor.device),
        precision=backend.precision,
        device=str(backend.predictor.device),
        cpu_threads=backend.cpu_threads,
        model_name=backend.model_name,
        backend=backend.name,
    )


def _oom_result(
    mode: str,
    batch_size: int,
    backend: InferenceBackend,
    error: BaseException,
) -> BenchmarkResult:
    return BenchmarkResult(
        mode=mode,
        batch_size=batch_size,
        mean_latency_ms=None,
        p50_latency_ms=None,
        p95_latency_ms=None,
        p99_latency_ms=None,
        throughput_images_per_second=None,
        warmup_total_ms=None,
        peak_gpu_memory_mib=_peak_memory(backend),
        process_ram_mib=_process_ram(backend.predictor.device),
        precision=backend.precision,
        device=str(backend.predictor.device),
        cpu_threads=backend.cpu_threads,
        model_name=backend.model_name,
        backend=backend.name,
        status="oom",
        error=str(error).splitlines()[0],
    )


def _benchmark_batches(
    *,
    mode: str,
    batch_sizes: Sequence[int],
    warmup_iterations: int,
    iterations: int,
    backend: InferenceBackend,
    prepare_call: Callable[[int], Callable[[], Any]],
    timer: Callable[[Callable[[], Any], torch.device], float],
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    device = backend.predictor.device
    for batch_size in batch_sizes:
        try:
            call = prepare_call(batch_size)
            with torch.inference_mode():
                _sync(device)
                warmup_start = time.perf_counter()
                for _ in range(warmup_iterations):
                    call()
                _sync(device)
                warmup_total_ms = (time.perf_counter() - warmup_start) * 1000.0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                latencies = [timer(call, device) for _ in range(iterations)]
            results.append(
                _summarize(
                    mode,
                    batch_size,
                    latencies,
                    warmup_total_ms,
                    backend,
                )
            )
        except (RuntimeError, MemoryError) as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and (
                "out of memory" not in str(error).lower()
            ):
                raise
            results.append(_oom_result(mode, batch_size, backend, error))
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
    return results


def _print_table(results: Sequence[BenchmarkResult]) -> None:
    print(
        f"{'backend':<25} {'precision':<7} {'device':<6} {'threads':>7} "
        f"{'mode':<12} {'batch':>5} {'warmup':>9} {'mean ms':>9} {'p50':>9} "
        f"{'p95':>9} {'p99':>9} {'img/s':>9} {'GPU MiB':>9} {'RAM MiB':>9} "
        f"{'status':>8}"
    )
    for result in results:
        raw_values = (
            result.mean_latency_ms,
            result.p50_latency_ms,
            result.p95_latency_ms,
            result.p99_latency_ms,
            result.throughput_images_per_second,
        )
        values = ["-" if value is None else f"{value:.2f}" for value in raw_values]
        gpu_memory = (
            "-"
            if result.peak_gpu_memory_mib is None
            else f"{result.peak_gpu_memory_mib:.1f}"
        )
        process_ram = (
            "-" if result.process_ram_mib is None else f"{result.process_ram_mib:.1f}"
        )
        warmup_total = (
            "-" if result.warmup_total_ms is None else f"{result.warmup_total_ms:.2f}"
        )
        cpu_threads = "-" if result.cpu_threads is None else str(result.cpu_threads)
        print(
            f"{result.backend:<25} {result.precision:<7} {result.device:<6} "
            f"{cpu_threads:>7} {result.mode:<12} {result.batch_size:>5} "
            f"{warmup_total:>9} {values[0]:>9} "
            f"{values[1]:>9} {values[2]:>9} {values[3]:>9} {values[4]:>9} "
            f"{gpu_memory:>9} {process_ram:>9} {result.status:>8}"
        )


def _save_results(
    output_directory: Path,
    payload: dict[str, Any],
    results: Sequence[BenchmarkResult],
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_directory / f"benchmark-{timestamp}.json"
    csv_path = output_directory / f"benchmark-{timestamp}.csv"
    rows = [asdict(result) for result in results]
    stored_payload = {"timestamp_utc": timestamp, **payload}
    json_path.write_text(json.dumps(stored_payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _run_device_benchmark(
    *,
    device: str,
    image_paths: Sequence[Path],
    batch_sizes: Sequence[int],
    warmup_iterations: int,
    iterations: int,
    cpu_threads: int | None = None,
    compile_model: bool = False,
    inference_backend: InferenceBackendName = "pytorch",
) -> tuple[dict[str, Any], list[BenchmarkResult]]:
    if compile_model:
        torch.compiler.reset()
    if device == "cpu":
        if cpu_threads is None:
            raise ValueError("CPU thread count is required for a CPU benchmark")
        torch.set_num_threads(cpu_threads)
        if torch.get_num_threads() != cpu_threads:
            raise RuntimeError("PyTorch did not apply the requested CPU thread count")
    elif cpu_threads is not None:
        raise ValueError("CPU thread count must be null for a CUDA benchmark")

    gpu_memory_baseline_mib = (
        _device_gpu_memory_used_mib(torch.device(device))
        if device == "cuda" and inference_backend == "onnxruntime-cuda"
        else None
    )
    load_start = time.perf_counter()
    predictor = load_predictor(
        device=device,
        compile_model=compile_model,
        backend=inference_backend,
        cpu_threads=cpu_threads or 1,
    )
    _sync(predictor.device)
    model_load_ms = (time.perf_counter() - load_start) * 1000.0
    backend = InferenceBackend(
        predictor,
        cpu_threads=cpu_threads,
        gpu_memory_baseline_mib=gpu_memory_baseline_mib,
    )
    cpu_tensors = [preprocess_image(predictor, path) for path in image_paths]

    cold_batch = cpu_tensors[0].unsqueeze(0).to(predictor.device)
    _sync(predictor.device)
    cold_timer = _end_to_end_timer if compile_model else _model_timer
    with torch.inference_mode():
        cold_inference_ms = cold_timer(
            lambda: backend.model_forward(cold_batch), predictor.device
        )

    def prepare_model_call(batch_size: int) -> Callable[[], Any]:
        image_batch = torch.stack(cpu_tensors[:batch_size]).to(predictor.device)
        _sync(predictor.device)
        return lambda: backend.model_forward(image_batch)

    def prepare_end_to_end_call(batch_size: int) -> Callable[[], Any]:
        paths = image_paths[:batch_size]
        return lambda: backend.predict_paths(paths)

    model_results = _benchmark_batches(
        mode="model-only",
        batch_sizes=batch_sizes,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        backend=backend,
        prepare_call=prepare_model_call,
        timer=_model_timer,
    )
    end_to_end_results = _benchmark_batches(
        mode="end-to-end",
        batch_sizes=batch_sizes,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        backend=backend,
        prepare_call=prepare_end_to_end_call,
        timer=_end_to_end_timer,
    )
    warmed_latency_ms = next(
        (
            result.mean_latency_ms
            for result in model_results
            if result.batch_size == 1 and result.status == "ok"
        ),
        None,
    )
    startup = {
        "device": str(predictor.device),
        "cpu_threads": cpu_threads,
        "backend": predictor.backend,
        "compile_mode": predictor.compile_mode,
        "model_load_ms": model_load_ms,
        "cold_model_inference_ms": cold_inference_ms,
        "cold_inference_includes_compilation": compile_model,
        "compile_and_first_inference_ms": cold_inference_ms if compile_model else None,
        "warmed_model_inference_mean_ms": warmed_latency_ms,
    }
    return startup, model_results + end_to_end_results


def run_benchmark(
    *,
    cuda_batch_sizes: Sequence[int] = DEFAULT_CUDA_BATCH_SIZES,
    cpu_batch_sizes: Sequence[int] = DEFAULT_CPU_BATCH_SIZES,
    cpu_thread_counts: Sequence[int] = DEFAULT_CPU_THREAD_COUNTS,
    warmup_iterations: int = 10,
    iterations: int = 50,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    compile_model: bool = False,
    inference_backend: InferenceBackendName = "pytorch",
) -> dict[str, Any]:
    if inference_backend != "pytorch" and compile_model:
        raise ValueError("torch.compile cannot be combined with ONNX Runtime")
    if inference_backend == "onnxruntime-cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if compile_model and warmup_iterations < 1:
        raise ValueError("Compiled benchmarks require at least one warmup iteration")
    environment = collect_environment()
    for warning in environment["warnings"]:
        print(f"WARNING: {warning}")
    print(json.dumps(environment, indent=2))

    device_configs: list[tuple[str, Sequence[int], int | None]] = []
    if (
        inference_backend in ("pytorch", "onnxruntime-cuda")
        and torch.cuda.is_available()
    ):
        device_configs.append(("cuda", cuda_batch_sizes, None))
    elif inference_backend == "pytorch":
        print("WARNING: CUDA is unavailable; the CUDA benchmark is skipped.")
    if inference_backend != "onnxruntime-cuda":
        device_configs.extend(
            ("cpu", cpu_batch_sizes, cpu_threads) for cpu_threads in cpu_thread_counts
        )

    image_paths = load_benchmark_inputs(
        max(max(batch_sizes) for _, batch_sizes, _ in device_configs)
    )
    startup: dict[str, Any] = {}
    results: list[BenchmarkResult] = []
    for device, batch_sizes, cpu_threads in device_configs:
        label = device if cpu_threads is None else f"{device}, threads: {cpu_threads}"
        print(f"Benchmark device: {label}")
        device_startup, device_results = _run_device_benchmark(
            device=device,
            image_paths=image_paths[: max(batch_sizes)],
            batch_sizes=batch_sizes,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            cpu_threads=cpu_threads,
            compile_model=compile_model,
            inference_backend=inference_backend,
        )
        if device == "cuda":
            startup[device] = device_startup
        else:
            startup.setdefault(device, []).append(device_startup)
        results.extend(device_results)
        if device == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "environment": environment,
        "config": {
            "cpu_batch_sizes": list(cpu_batch_sizes),
            "cuda_batch_sizes": list(cuda_batch_sizes),
            "cpu_thread_counts": list(cpu_thread_counts),
            "warmup_iterations": warmup_iterations,
            "iterations": iterations,
            "output_directory": str(output_directory),
            "compile_model": compile_model,
            "inference_backend": inference_backend,
        },
        "startup": startup,
        "input_paths": [str(path) for path in image_paths],
        "results": [asdict(result) for result in results],
    }
    _print_table(results)
    print(json.dumps(startup, indent=2))
    json_path, csv_path = _save_results(output_directory, payload, results)
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return {**payload, "json_path": json_path, "csv_path": csv_path}


def _parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Benchmark Birdly inference.")
    parser.add_argument(
        "--cuda-batch-sizes",
        "--batch-sizes",
        dest="cuda_batch_sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_CUDA_BATCH_SIZES),
    )
    parser.add_argument(
        "--cpu-batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_CPU_BATCH_SIZES),
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--backend",
        choices=(
            "pytorch",
            "onnxruntime",
            "onnxruntime-cuda",
            "onnxruntime-int8",
        ),
        default="pytorch",
    )
    parser.add_argument(
        "--cpu-thread-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_CPU_THREAD_COUNTS),
    )
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    args = parser.parse_args()
    cuda_batch_sizes = tuple(sorted(set(args.cuda_batch_sizes)))
    cpu_batch_sizes = tuple(sorted(set(args.cpu_batch_sizes)))
    cpu_thread_counts = tuple(sorted(set(args.cpu_thread_counts)))
    if (
        not cuda_batch_sizes
        or cuda_batch_sizes[0] < 1
        or not cpu_batch_sizes
        or cpu_batch_sizes[0] < 1
        or not cpu_thread_counts
        or cpu_thread_counts[0] < 1
    ):
        parser.error("batch sizes must be positive")
    if args.warmup_iterations < 0 or args.iterations < 1:
        parser.error("warmup must not be negative and iterations must be positive")
    if args.compile_model and args.warmup_iterations < 1:
        parser.error("compiled benchmarks require at least one warmup iteration")
    if args.backend != "pytorch" and args.compile_model:
        parser.error("--compile cannot be combined with --backend onnxruntime")
    return {
        "cuda_batch_sizes": cuda_batch_sizes,
        "cpu_batch_sizes": cpu_batch_sizes,
        "cpu_thread_counts": cpu_thread_counts,
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "output_directory": args.output_dir,
        "compile_model": args.compile_model,
        "inference_backend": args.backend,
    }


def main() -> None:
    run_benchmark(**_parse_args())


if __name__ == "__main__":
    main()
