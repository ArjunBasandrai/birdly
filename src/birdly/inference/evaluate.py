from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import torch
from torch.nn import functional as F

from birdly.inference.predict import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CLASS_MAPPING_PATH,
    DEFAULT_ONNX_INT8_MODEL_PATH,
    DEFAULT_ONNX_MODEL_PATH,
    DEFAULT_TENSORRT_ENGINE_PATH,
    InferenceBackendName,
    Predictor,
    inference_context,
    load_predictor,
    predict_logits,
)
from birdly.training.metrics import ClassificationMetrics

DEFAULT_MANIFEST_PATH = Path("data/processed/validated_manifest.parquet")
DEFAULT_IMAGE_ROOT = Path("B:/images")
DEFAULT_OUTPUT_ROOT = Path("artifacts/evaluation")
DEFAULT_TRACKING_URI = "sqlite:///B:/birdly-training/birdly-research-v3/mlflow.db"
DEFAULT_EXPERIMENT_NAME = "birdly-local-research"
DEFAULT_BATCH_SIZE = 16
DEFAULT_CALIBRATION_BINS = 15


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_test_rows(
    manifest_path: Path,
    image_root: Path,
    predictor: Predictor,
) -> tuple[list[tuple[Path, int]], int]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Validated manifest is missing: {manifest_path}")
    manifest = pl.read_parquet(manifest_path)
    required_columns = {"split", "taxon_id", "image_path"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Validated manifest is missing columns: {sorted(missing_columns)}"
        )

    test_rows = manifest.filter(pl.col("split") == "test").select(
        "taxon_id", "image_path"
    )
    if test_rows.is_empty():
        raise ValueError("Validated manifest has no test rows")

    taxon_to_class = {
        taxon_id: class_index
        for class_index, taxon_id in predictor.class_to_taxon.items()
    }
    test_taxa = {int(value) for value in test_rows["taxon_id"].to_list()}
    unknown_taxa = sorted(test_taxa - set(taxon_to_class))
    if unknown_taxa:
        raise ValueError(
            f"Test manifest contains taxa outside the class mapping: {unknown_taxa[:5]}"
        )

    rows = [
        (image_root / str(image_path), taxon_to_class[int(taxon_id)])
        for taxon_id, image_path in test_rows.iter_rows()
    ]
    missing_images = [path for path, _ in rows if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Test split has {len(missing_images)} missing images; "
            f"first missing image: {missing_images[0]}"
        )
    return rows, test_rows.height


def _evaluate_batches(
    predictor: Predictor,
    rows: list[tuple[Path, int]],
    *,
    batch_size: int,
    calibration_bins: int,
) -> tuple[ClassificationMetrics, float, int]:
    metrics = ClassificationMetrics(
        len(predictor.class_to_taxon),
        calibration_bins=calibration_bins,
    )
    if predictor.device.type == "cuda":
        torch.cuda.synchronize(predictor.device)
    started = time.perf_counter()
    prediction_count = 0

    with torch.inference_mode(), inference_context(predictor):
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            image_paths = [path for path, _ in batch]
            targets = torch.tensor(
                [class_index for _, class_index in batch],
                dtype=torch.int64,
                device=predictor.device,
            )
            logits = predict_logits(predictor, image_paths)
            if logits.shape != (len(batch), len(predictor.class_to_taxon)):
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
            loss_sum = F.cross_entropy(logits, targets, reduction="sum")
            metrics.update(logits, targets, float(loss_sum.item()))
            prediction_count += logits.shape[0]

    if predictor.device.type == "cuda":
        torch.cuda.synchronize(predictor.device)
    elapsed_seconds = time.perf_counter() - started
    return metrics, elapsed_seconds, prediction_count


def _write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "class_index",
        "taxon_id",
        "species_name",
        "support",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _log_mlflow(
    summary: dict[str, Any],
    output_directory: Path,
    summary_path: Path,
) -> str:
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_registry_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
    run_prefix = "fp32" if summary["backend"] == "pytorch-eager" else summary["backend"]
    run_name = f"{run_prefix}-test-evaluation"
    if summary["is_subset"]:
        run_name += "-subset"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        summary["mlflow_run_id"] = run_id
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = summary["metrics"]
        mlflow.log_metrics(
            {
                "test_top1_accuracy": metrics["top1_accuracy"],
                "test_top5_accuracy": metrics["top5_accuracy"],
                "test_macro_precision": metrics["macro_precision"],
                "test_macro_recall": metrics["macro_recall"],
                "test_macro_f1": metrics["macro_f1"],
                "test_mcc": metrics["mcc"],
                "test_negative_log_likelihood": metrics["negative_log_likelihood"],
                "test_expected_calibration_error": metrics[
                    "expected_calibration_error"
                ],
                "test_runtime_seconds": summary["runtime_seconds"],
                "test_throughput_images_per_second": summary[
                    "throughput_images_per_second"
                ],
            }
        )
        mlflow.log_params(
            {
                "backend": summary["backend"],
                "batch_size": summary["batch_size"],
                "calibration_bins": summary["calibration_bins"],
                "checkpoint_path": summary["model_artifact_path"],
                "checkpoint_sha256": summary["model_artifact_sha256"],
                "class_count": summary["class_count"],
                "cpu_threads": summary["cpu_threads"]
                if summary["cpu_threads"] is not None
                else "-",
                "device": summary["device"],
                "evaluated_row_count": summary["evaluated_row_count"],
                "image_size": summary["image_size"],
                "manifest_path": summary["manifest_path"],
                "manifest_sha256": summary["manifest_sha256"],
                "precision": summary["precision"],
                "test_row_count": summary["test_row_count"],
            }
        )
        mlflow.set_tags(
            {
                "birdly.backend": summary["backend"],
                "birdly.git_commit": summary["git_commit"],
                "birdly.run_type": "test-evaluation",
                "birdly.split": "test",
                "birdly.subset": str(summary["is_subset"]).lower(),
            }
        )
        mlflow.log_artifacts(str(output_directory), artifact_path="evaluation")
    return run_id


def evaluate(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_images: int | None = None,
    backend: InferenceBackendName = "pytorch",
    cpu_threads: int = 1,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be positive")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")

    if backend in ("onnxruntime", "onnxruntime-int8"):
        device = torch.device("cpu")
    elif backend in ("pytorch-fp16", "onnxruntime-cuda", "tensorrt"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_load_started = time.perf_counter()
    predictor = load_predictor(
        device=device,
        backend=backend,
        cpu_threads=cpu_threads,
    )
    if predictor.device.type == "cuda":
        torch.cuda.synchronize(predictor.device)
    model_load_seconds = time.perf_counter() - model_load_started
    all_rows, test_row_count = _load_test_rows(
        manifest_path,
        image_root,
        predictor,
    )
    rows = all_rows if max_images is None else all_rows[:max_images]
    is_subset = len(rows) != test_row_count

    metric_state, runtime_seconds, prediction_count = _evaluate_batches(
        predictor,
        rows,
        batch_size=batch_size,
        calibration_bins=calibration_bins,
    )
    if prediction_count != len(rows):
        raise RuntimeError(
            f"Prediction count {prediction_count} does not match "
            f"evaluated row count {len(rows)}"
        )
    if not is_subset and prediction_count != test_row_count:
        raise RuntimeError(
            f"Prediction count {prediction_count} does not match "
            f"test row count {test_row_count}"
        )

    computed = metric_state.compute(predictor.class_to_taxon)
    loss = float(computed.pop("loss"))
    per_class = computed.pop("per_class")
    for row in per_class:
        row["species_name"] = predictor.taxon_to_name[int(row["taxon_id"])]
    metrics = {
        **computed,
        "cross_entropy": loss,
        "negative_log_likelihood": loss,
    }

    if predictor.backend == "onnxruntime-int8":
        model_artifact_path = DEFAULT_ONNX_INT8_MODEL_PATH
    elif predictor.backend == "tensorrt-fp16":
        model_artifact_path = DEFAULT_TENSORRT_ENGINE_PATH
    elif predictor.backend.startswith("onnxruntime"):
        model_artifact_path = DEFAULT_ONNX_MODEL_PATH
    else:
        model_artifact_path = DEFAULT_CHECKPOINT_PATH

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "-subset" if is_subset else ""
    output_prefix = (
        "fp32" if predictor.backend == "pytorch-eager" else predictor.backend
    )
    output_directory = output_root / f"{output_prefix}-test-{timestamp}{suffix}"
    output_directory.mkdir(parents=True, exist_ok=False)
    summary_path = output_directory / "summary.json"
    per_class_path = output_directory / "per_class_metrics.csv"
    confusion_path = output_directory / "confusion_matrix.npz"
    _write_per_class_csv(per_class_path, per_class)
    np.savez_compressed(
        confusion_path,
        confusion_matrix=metric_state.confusion.numpy(),
        class_indices=np.arange(len(predictor.class_to_taxon), dtype=np.int64),
        taxon_ids=np.array(
            [
                predictor.class_to_taxon[index]
                for index in range(len(predictor.class_to_taxon))
            ],
            dtype=np.int64,
        ),
    )

    summary: dict[str, Any] = {
        "artifacts": {
            "confusion_matrix": str(confusion_path.resolve()),
            "per_class_metrics": str(per_class_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
        "backend": predictor.backend,
        "batch_size": batch_size,
        "calibration_bins": calibration_bins,
        "class_count": len(predictor.class_to_taxon),
        "class_mapping_path": str(DEFAULT_CLASS_MAPPING_PATH),
        "class_mapping_sha256": _file_sha256(DEFAULT_CLASS_MAPPING_PATH),
        "cpu_threads": cpu_threads if predictor.device.type == "cpu" else None,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "device": str(predictor.device),
        "evaluated_row_count": len(rows),
        "git_commit": _git_commit(),
        "image_root": str(image_root.resolve()),
        "image_size": predictor.image_size,
        "is_subset": is_subset,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "metrics": metrics,
        "model_artifact_path": str(model_artifact_path),
        "model_artifact_sha256": _file_sha256(model_artifact_path),
        "model_load_seconds": model_load_seconds,
        "model_name": predictor.model_name,
        "precision": predictor.precision,
        "prediction_count": prediction_count,
        "preprocessing": predictor.preprocessing,
        "runtime_seconds": runtime_seconds,
        "split": "test",
        "test_row_count": test_row_count,
        "throughput_images_per_second": prediction_count / runtime_seconds,
    }
    run_id = _log_mlflow(summary, output_directory, summary_path)
    summary["mlflow_run_id"] = run_id
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Birdly on test data.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--calibration-bins", type=int, default=DEFAULT_CALIBRATION_BINS
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--backend",
        choices=(
            "pytorch",
            "pytorch-fp16",
            "onnxruntime",
            "onnxruntime-cuda",
            "onnxruntime-int8",
            "tensorrt",
        ),
        default="pytorch",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = evaluate(
        batch_size=args.batch_size,
        calibration_bins=args.calibration_bins,
        manifest_path=args.manifest,
        image_root=args.image_root,
        output_root=args.output_root,
        max_images=args.max_images,
        backend=args.backend,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
