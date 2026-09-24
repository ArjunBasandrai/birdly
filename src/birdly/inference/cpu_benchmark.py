from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnxruntime as ort
import psutil
import torch

from birdly.inference.predict import (
    Predictor,
    inference_context,
    load_predictor,
    predict_image,
    preprocess_image,
)

DEFAULT_BACKENDS = (
    "pytorch-eager",
    "pytorch-compile",
    "onnxruntime-fp32",
    "onnxruntime-int8",
)
DEFAULT_THREAD_COUNTS = (1, 2, 4)
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_ITERATIONS = 20
DEFAULT_OUTPUT_ROOT = Path("/results")
DEFAULT_MODEL_DIRECTORY = Path("/app/model")
DEFAULT_INPUT_IMAGE = Path("/app/input/representative-image")
DEFAULT_MANIFEST_PATH = Path("/app/build-manifest.json")


@dataclass(frozen=True)
class BackendSpec:
    predictor_backend: str
    compile_model: bool
    artifact_name: str


@dataclass
class Measurement:
    backend: str
    runtime_backend: str
    mode: str
    threads: int
    batch_size: int
    model_load_ms: float
    cold_inference_ms: float
    compile_first_inference_ms: float | None
    warmup_total_ms: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_images_per_second: float
    peak_process_rss_mib: float
    model_artifact_size_bytes: int
    model_artifact_size_mib: float
    runtime_version: str
    model_name: str
    git_commit: str
    status: str = "ok"
    error: str = ""


BACKEND_SPECS = {
    "pytorch-eager": BackendSpec("pytorch", False, "best.pt"),
    "pytorch-compile": BackendSpec("pytorch", True, "best.pt"),
    "onnxruntime-fp32": BackendSpec("onnxruntime", False, "onnx/birdly-fp32.onnx"),
    "onnxruntime-int8": BackendSpec("onnxruntime-int8", False, "onnx/birdly-int8.onnx"),
}


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _measure(call: Callable[[], Any]) -> float:
    started = time.perf_counter()
    call()
    return (time.perf_counter() - started) * 1000.0


def _peak_rss_mib() -> float:
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except ImportError:
        return psutil.Process().memory_info().rss / (1024**2)


def _summarize(
    *,
    backend_name: str,
    predictor: Predictor,
    mode: str,
    threads: int,
    model_load_ms: float,
    cold_inference_ms: float,
    compile_first_inference_ms: float | None,
    warmup_total_ms: float,
    latencies: Sequence[float],
    artifact_size_bytes: int,
    runtime_version: str,
    git_commit: str,
) -> Measurement:
    mean = sum(latencies) / len(latencies)
    return Measurement(
        backend=backend_name,
        runtime_backend=predictor.backend,
        mode=mode,
        threads=threads,
        batch_size=1,
        model_load_ms=model_load_ms,
        cold_inference_ms=cold_inference_ms,
        compile_first_inference_ms=compile_first_inference_ms,
        warmup_total_ms=warmup_total_ms,
        mean_latency_ms=mean,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        p99_latency_ms=_percentile(latencies, 0.99),
        throughput_images_per_second=1000.0 / mean,
        peak_process_rss_mib=_peak_rss_mib(),
        model_artifact_size_bytes=artifact_size_bytes,
        model_artifact_size_mib=artifact_size_bytes / (1024**2),
        runtime_version=runtime_version,
        model_name=predictor.model_name,
        git_commit=git_commit,
    )


def _benchmark_call(
    call: Callable[[], Any],
    *,
    warmup_iterations: int,
    iterations: int,
) -> tuple[float, float, list[float]]:
    cold_latency_ms = _measure(call)
    warmup_started = time.perf_counter()
    for _ in range(warmup_iterations):
        call()
    warmup_total_ms = (time.perf_counter() - warmup_started) * 1000.0
    latencies = [_measure(call) for _ in range(iterations)]
    return cold_latency_ms, warmup_total_ms, latencies


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("The build manifest must be a JSON object")
    return payload


def _load_worker_predictor(
    backend_name: str,
    model_directory: Path,
    threads: int,
) -> tuple[Predictor, float]:
    spec = BACKEND_SPECS[backend_name]
    if spec.compile_model:
        torch.compiler.reset()
    started = time.perf_counter()
    predictor = load_predictor(
        checkpoint_path=model_directory / "best.pt",
        class_mapping_path=model_directory / "class_mapping.json",
        preprocessing_path=model_directory / "preprocessing.json",
        onnx_model_path=model_directory / "onnx" / "birdly-fp32.onnx",
        onnx_metadata_path=model_directory / "onnx" / "metadata.json",
        onnx_int8_model_path=model_directory / "onnx" / "birdly-int8.onnx",
        onnx_int8_metadata_path=model_directory / "onnx" / "metadata-int8.json",
        device="cpu",
        backend=spec.predictor_backend,
        compile_model=spec.compile_model,
        cpu_threads=threads,
    )
    return predictor, (time.perf_counter() - started) * 1000.0


def _run_worker(args: argparse.Namespace) -> int:
    threads = args.threads[0]
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    manifest = _load_manifest(args.manifest)
    spec = BACKEND_SPECS[args.backend]
    artifact_path = args.model_directory / spec.artifact_name
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {artifact_path}")
    predictor, model_load_ms = _load_worker_predictor(
        args.backend,
        args.model_directory,
        threads,
    )
    runtime_version = (
        torch.__version__ if spec.predictor_backend == "pytorch" else ort.__version__
    )
    artifact_size_bytes = artifact_path.stat().st_size
    input_tensor = preprocess_image(predictor, args.input_image).unsqueeze(0)

    def model_call() -> Any:
        with torch.inference_mode(), inference_context(predictor):
            return predictor.model(input_tensor)

    model_cold_ms, model_warmup_ms, model_latencies = _benchmark_call(
        model_call,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    compile_first_inference_ms = model_cold_ms if spec.compile_model else None

    def end_to_end_call() -> Any:
        return predict_image(predictor, args.input_image, top_k=5)

    end_to_end_cold_ms, end_to_end_warmup_ms, end_to_end_latencies = _benchmark_call(
        end_to_end_call,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    git_commit = str(manifest["git_commit"])
    measurements = [
        _summarize(
            backend_name=args.backend,
            predictor=predictor,
            mode="model-only",
            threads=threads,
            model_load_ms=model_load_ms,
            cold_inference_ms=model_cold_ms,
            compile_first_inference_ms=compile_first_inference_ms,
            warmup_total_ms=model_warmup_ms,
            latencies=model_latencies,
            artifact_size_bytes=artifact_size_bytes,
            runtime_version=runtime_version,
            git_commit=git_commit,
        ),
        _summarize(
            backend_name=args.backend,
            predictor=predictor,
            mode="end-to-end",
            threads=threads,
            model_load_ms=model_load_ms,
            cold_inference_ms=end_to_end_cold_ms,
            compile_first_inference_ms=compile_first_inference_ms,
            warmup_total_ms=end_to_end_warmup_ms,
            latencies=end_to_end_latencies,
            artifact_size_bytes=artifact_size_bytes,
            runtime_version=runtime_version,
            git_commit=git_commit,
        ),
    ]
    payload = {
        "backend": args.backend,
        "compile_cache_directory": os.getenv("TORCHINDUCTOR_CACHE_DIR"),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "iterations": args.iterations,
        "measurements": [asdict(measurement) for measurement in measurements],
        "process_id": os.getpid(),
        "status": "ok",
        "threads": threads,
        "warmup_iterations": args.warmup_iterations,
    }
    _write_json(args.worker_result, payload)
    return 0


def _cpu_information() -> dict[str, Any]:
    cpu_model = None
    instruction_sets: list[str] = []
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        for line in cpuinfo_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("model name") and cpu_model is None:
                cpu_model = line.split(":", 1)[1].strip()
            if line.startswith("flags"):
                instruction_sets = line.split(":", 1)[1].split()
                break
    if cpu_model is None:
        cpu_model = platform.processor() or None
    process = psutil.Process()
    affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else None
    return {
        "cpu_affinity": affinity,
        "cpu_instruction_sets": instruction_sets,
        "cpu_model": cpu_model,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "total_ram_gib": psutil.virtual_memory().total / (1024**3),
    }


def _create_run_directory(output_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _failure_payload(
    *,
    backend: str,
    threads: int,
    detail: str,
    return_code: int | None,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "error": detail,
        "measurements": [],
        "return_code": return_code,
        "status": "failed",
        "threads": threads,
    }


def _run_configuration(
    *,
    backend: str,
    threads: int,
    args: argparse.Namespace,
    run_directory: Path,
) -> dict[str, Any]:
    result_path = run_directory / f"{backend}-threads-{threads}.json"
    command = [
        sys.executable,
        "-m",
        "birdly.inference.cpu_benchmark",
        "--worker",
        "--backend",
        backend,
        "--threads",
        str(threads),
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--iterations",
        str(args.iterations),
        "--model-directory",
        str(args.model_directory),
        "--input-image",
        str(args.input_image),
        "--manifest",
        str(args.manifest),
        "--worker-result",
        str(result_path),
    ]
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    cache_directory: Path | None = None
    if backend == "pytorch-compile":
        cache_directory = Path(tempfile.mkdtemp(prefix="birdly-compile-"))
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_directory / "inductor")
        environment["TORCH_EXTENSIONS_DIR"] = str(cache_directory / "extensions")
        environment["TRITON_CACHE_DIR"] = str(cache_directory / "triton")
    try:
        completed = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.worker_timeout_seconds,
            check=False,
        )
        if completed.returncode == 0 and result_path.is_file():
            return json.loads(result_path.read_text(encoding="utf-8"))
        detail = (
            completed.stderr or completed.stdout or "Worker produced no result"
        ).strip()
        payload = _failure_payload(
            backend=backend,
            threads=threads,
            detail=detail[-4000:],
            return_code=completed.returncode,
        )
    except subprocess.TimeoutExpired as error:
        payload = _failure_payload(
            backend=backend,
            threads=threads,
            detail=f"Worker timed out after {error.timeout} seconds",
            return_code=None,
        )
    finally:
        if cache_directory is not None:
            shutil.rmtree(cache_directory, ignore_errors=True)
    _write_json(result_path, payload)
    return payload


def _write_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: Sequence[dict[str, Any]]) -> None:
    print(
        f"{'backend':<20} {'threads':>7} {'mode':<11} {'p50 ms':>9} "
        f"{'p95 ms':>9} {'p99 ms':>9} {'img/s':>9} {'RAM MiB':>9} "
        f"{'size MiB':>9} {'status':>8}"
    )
    for row in rows:
        if row["status"] == "ok":
            values = [
                f"{row['p50_latency_ms']:.2f}",
                f"{row['p95_latency_ms']:.2f}",
                f"{row['p99_latency_ms']:.2f}",
                f"{row['throughput_images_per_second']:.2f}",
                f"{row['peak_process_rss_mib']:.1f}",
                f"{row['model_artifact_size_mib']:.1f}",
            ]
        else:
            values = ["-"] * 6
        print(
            f"{row['backend']:<20} {row['threads']:>7} {row['mode']:<11} "
            f"{values[0]:>9} {values[1]:>9} {values[2]:>9} {values[3]:>9} "
            f"{values[4]:>9} {values[5]:>9} {row['status']:>8}"
        )


def _validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    missing = [
        path
        for path in (args.model_directory, args.input_image, args.manifest)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Benchmark input is missing: {missing[0]}")
    manifest = _load_manifest(args.manifest)
    expected_files = manifest.get("model_files", {})
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("Build manifest has no model files")
    return manifest


def _run_orchestrator(args: argparse.Namespace) -> int:
    manifest = _validate_inputs(args)
    run_directory = _create_run_directory(args.output_root)
    environment = {
        **_cpu_information(),
        "container_id": platform.node(),
        "git_commit": manifest["git_commit"],
        "input_image_sha256": manifest["input_image"]["sha256"],
        "iterations": args.iterations,
        "onnxruntime_version": ort.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "run_id": run_directory.name,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "thread_counts": args.threads,
        "warmup_iterations": args.warmup_iterations,
    }
    _write_json(run_directory / "environment.json", environment)

    configuration_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for backend in args.backends:
        for threads in args.threads:
            print(f"Running {backend}, threads={threads} ...", flush=True)
            payload = _run_configuration(
                backend=backend,
                threads=threads,
                args=args,
                run_directory=run_directory,
            )
            configuration_results.append(payload)
            if payload["status"] == "ok":
                rows.extend(payload["measurements"])
                print("  PASS", flush=True)
            else:
                rows.append(
                    {
                        "backend": backend,
                        "threads": threads,
                        "mode": "-",
                        "status": "failed",
                        "error": payload["error"],
                    }
                )
                print(f"  FAIL: {payload['error'].splitlines()[-1]}", flush=True)

    _write_json(
        run_directory / "combined.json",
        {
            "configurations": configuration_results,
            "environment": environment,
            "measurements": rows,
        },
    )
    _write_combined_csv(run_directory / "combined.csv", rows)
    failures = [
        f"{result['backend']}-threads-{result['threads']}"
        for result in configuration_results
        if result["status"] != "ok"
    ]
    summary = {
        "artifact_directory": str(run_directory),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "failed_configurations": failures,
        "successful_configurations": len(configuration_results) - len(failures),
        "total_configurations": len(configuration_results),
    }
    _write_json(run_directory / "final-summary.json", summary)
    _print_table(rows)
    print(
        f"\nCompleted: {summary['successful_configurations']}/"
        f"{summary['total_configurations']} configurations"
    )
    print(f"Failed configurations: {', '.join(failures) if failures else 'none'}")
    print(f"Artifacts: {run_directory}")
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated Birdly CPU benchmarks.")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=DEFAULT_BACKENDS,
        default=list(DEFAULT_BACKENDS),
    )
    parser.add_argument(
        "--threads", nargs="+", type=int, default=list(DEFAULT_THREAD_COUNTS)
    )
    parser.add_argument(
        "--warmup-iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--worker-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--input-image", type=Path, default=DEFAULT_INPUT_IMAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=DEFAULT_BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.backends = list(dict.fromkeys(args.backends))
    args.threads = sorted(set(args.threads))
    if not args.threads or args.threads[0] < 1:
        parser.error("Thread counts must be positive")
    if args.warmup_iterations < 0 or args.iterations < 1:
        parser.error("Warmup must not be negative and iterations must be positive")
    if args.worker_timeout_seconds <= 0:
        parser.error("Worker timeout must be positive")
    if args.worker and (args.backend is None or args.worker_result is None):
        parser.error("Worker mode requires --backend and --worker-result")
    if args.worker and len(args.threads) != 1:
        parser.error("Worker mode requires one thread count")
    return args


def main() -> None:
    args = _parse_arguments()
    try:
        exit_code = _run_worker(args) if args.worker else _run_orchestrator(args)
    except Exception as error:  # noqa: BLE001
        print(
            f"Benchmark harness failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
