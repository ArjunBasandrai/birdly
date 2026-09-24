from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import polars as pl
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from birdly.inference.export_onnx import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    INPUT_NAME,
    OUTPUT_NAME,
)
from birdly.inference.predict import (
    DEFAULT_CLASS_MAPPING_PATH,
    DEFAULT_ONNX_INT8_METADATA_PATH,
    DEFAULT_ONNX_INT8_MODEL_PATH,
    DEFAULT_ONNX_METADATA_PATH,
    DEFAULT_ONNX_MODEL_PATH,
    DEFAULT_PREPROCESSING_PATH,
    Predictor,
    load_predictor,
    predict_batch,
    preprocess_image,
)

DEFAULT_MANIFEST_PATH = Path("data/processed/validated_manifest.parquet")
DEFAULT_IMAGE_ROOT = Path("B:/images")
DEFAULT_CALIBRATION_PATH = DEFAULT_ONNX_INT8_MODEL_PATH.with_name(
    "calibration-int8.json"
)
DEFAULT_VERIFICATION_PATH = DEFAULT_ONNX_INT8_MODEL_PATH.with_name(
    "verification-int8.json"
)
DEFAULT_REGISTRATION_PATH = DEFAULT_ONNX_INT8_MODEL_PATH.with_name(
    "registration-int8.json"
)
DEFAULT_REGISTERED_MODEL_ALIAS = "cpu-int8"
DEFAULT_CALIBRATION_SAMPLE_COUNT = 64
DEFAULT_CALIBRATION_BATCH_SIZE = 8
DEFAULT_CALIBRATION_SEED = 2027
DEFAULT_VERIFICATION_IMAGE_COUNT = 4
QUANTIZED_OPERATOR_TYPES = ("Conv", "MatMul", "Gemm")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def select_calibration_samples(
    manifest_path: Path,
    image_root: Path,
    *,
    sample_count: int,
    seed: int,
) -> tuple[list[Path], dict[str, Any]]:
    if sample_count < 1:
        raise ValueError("Calibration sample count must be positive")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Validated manifest is missing: {manifest_path}")

    manifest = pl.read_parquet(
        manifest_path,
        columns=["split", "taxon_id", "image_path"],
    )
    training_rows = manifest.filter(pl.col("split") == "train").select(
        "taxon_id", "image_path"
    )
    if training_rows.height < sample_count:
        raise ValueError(
            f"Calibration requested {sample_count} images, but the training split "
            f"has {training_rows.height} rows"
        )

    paths_by_taxon: dict[int, list[str]] = {}
    for taxon_id, image_path in training_rows.iter_rows():
        paths_by_taxon.setdefault(int(taxon_id), []).append(str(image_path))

    generator = random.Random(seed)
    taxon_ids = sorted(paths_by_taxon)
    generator.shuffle(taxon_ids)
    for paths in paths_by_taxon.values():
        paths.sort()
        generator.shuffle(paths)

    selected: list[dict[str, Any]] = []
    positions = {taxon_id: 0 for taxon_id in taxon_ids}
    while len(selected) < sample_count:
        added = False
        for taxon_id in taxon_ids:
            position = positions[taxon_id]
            paths = paths_by_taxon[taxon_id]
            if position >= len(paths):
                continue
            selected.append(
                {
                    "image_path": paths[position],
                    "split": "train",
                    "taxon_id": taxon_id,
                }
            )
            positions[taxon_id] += 1
            added = True
            if len(selected) == sample_count:
                break
        if not added:
            raise RuntimeError("Could not select enough calibration images")

    image_paths = [image_root / row["image_path"] for row in selected]
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Calibration image is missing: {missing_images[0]}")

    selection_bytes = json.dumps(
        selected,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    report = {
        "batch_size": DEFAULT_CALIBRATION_BATCH_SIZE,
        "class_count_in_training_split": len(paths_by_taxon),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "image_root": str(image_root),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "sample_count": sample_count,
        "samples": selected,
        "seed": seed,
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "source_split": "train",
        "training_row_count": training_rows.height,
    }
    return image_paths, report


class BirdlyCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        image_paths: Sequence[Path],
        predictor: Predictor,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Calibration batch size must be positive")
        self.image_paths = list(image_paths)
        self.predictor = predictor
        self.batch_size = batch_size
        self.position = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.position >= len(self.image_paths):
            return None
        paths = self.image_paths[self.position : self.position + self.batch_size]
        self.position += len(paths)
        image_batch = torch.stack(
            [preprocess_image(self.predictor, path) for path in paths]
        )
        return {INPUT_NAME: image_batch.numpy()}

    def rewind(self) -> None:
        self.position = 0


def quantize_onnx_model(
    calibration_paths: Sequence[Path],
    calibration_report: dict[str, Any],
    *,
    calibration_batch_size: int,
    source_path: Path = DEFAULT_ONNX_MODEL_PATH,
    output_path: Path = DEFAULT_ONNX_INT8_MODEL_PATH,
    metadata_path: Path = DEFAULT_ONNX_INT8_METADATA_PATH,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(f"FP32 ONNX model is missing: {source_path}")
    source_metadata = json.loads(DEFAULT_ONNX_METADATA_PATH.read_text(encoding="utf-8"))
    source_hash = _file_sha256(source_path)
    if source_metadata["onnx_sha256"] != source_hash:
        raise ValueError("FP32 ONNX model hash does not match its metadata")

    preprocessing_predictor = load_predictor(
        backend="onnxruntime",
        device="cpu",
        cpu_threads=1,
    )
    calibration_reader = BirdlyCalibrationDataReader(
        calibration_paths,
        preprocessing_predictor,
        calibration_batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp.onnx")
    temporary_path.unlink(missing_ok=True)
    try:
        quantize_static(
            model_input=source_path,
            model_output=temporary_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=list(QUANTIZED_OPERATOR_TYPES),
            per_channel=True,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=["CPUExecutionProvider"],
            use_external_data_format=False,
        )
        onnx.checker.check_model(str(temporary_path))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    source_model = onnx.load(str(source_path), load_external_data=False)
    int8_model = onnx.load(str(output_path), load_external_data=False)
    source_operator_counts = Counter(node.op_type for node in source_model.graph.node)
    int8_operator_counts = Counter(node.op_type for node in int8_model.graph.node)
    intentionally_unquantized = sorted(
        set(source_operator_counts) - set(QUANTIZED_OPERATOR_TYPES)
    )
    metadata = {
        "artifact_size_bytes": output_path.stat().st_size,
        "backend": "onnxruntime-int8",
        "batch_dimension": "dynamic",
        "calibration": {
            "batch_size": calibration_batch_size,
            "manifest_path": calibration_report["manifest_path"],
            "manifest_sha256": calibration_report["manifest_sha256"],
            "sample_count": calibration_report["sample_count"],
            "seed": calibration_report["seed"],
            "selection_sha256": calibration_report["selection_sha256"],
            "source_split": calibration_report["source_split"],
        },
        "class_count": source_metadata["class_count"],
        "class_mapping_path": str(DEFAULT_CLASS_MAPPING_PATH),
        "class_mapping_sha256": source_metadata["class_mapping_sha256"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "image_size": source_metadata["image_size"],
        "input_name": INPUT_NAME,
        "int8_onnx_path": str(output_path),
        "int8_onnx_sha256": _file_sha256(output_path),
        "model_name": source_metadata["model_name"],
        "onnx_version": onnx.__version__,
        "onnxruntime_execution_provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "opset": source_metadata["opset"],
        "output_name": OUTPUT_NAME,
        "precision": "int8",
        "preprocessing": source_metadata["preprocessing"],
        "preprocessing_path": str(DEFAULT_PREPROCESSING_PATH),
        "preprocessing_sha256": source_metadata["preprocessing_sha256"],
        "quantization": {
            "activation_type": "QInt8",
            "calibration_method": "MinMax",
            "format": "QDQ",
            "int8_operator_counts": dict(sorted(int8_operator_counts.items())),
            "operator_types_intentionally_left_fp32": intentionally_unquantized,
            "operator_types_to_quantize": list(QUANTIZED_OPERATOR_TYPES),
            "per_channel": True,
            "reduce_range": False,
            "source_operator_counts": dict(sorted(source_operator_counts.items())),
            "weight_type": "QInt8",
        },
        "quantization_git_commit": _git_commit(),
        "source_checkpoint_path": source_metadata["source_checkpoint_path"],
        "source_checkpoint_sha256": source_metadata["source_checkpoint_sha256"],
        "source_fp32_onnx_path": str(source_path),
        "source_fp32_onnx_sha256": source_hash,
        "source_fp32_size_bytes": source_path.stat().st_size,
        "source_training_run_id": source_metadata["source_training_run_id"],
    }
    _write_json(metadata_path, metadata)
    return metadata


def verify_int8_inference(
    image_paths: Sequence[Path],
    *,
    report_path: Path = DEFAULT_VERIFICATION_PATH,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("At least one verification image is required")
    fp32_predictor = load_predictor(
        backend="onnxruntime",
        device="cpu",
        cpu_threads=1,
    )
    int8_predictor = load_predictor(
        backend="onnxruntime-int8",
        device="cpu",
        cpu_threads=1,
    )
    image_batch = torch.stack(
        [preprocess_image(fp32_predictor, path) for path in image_paths]
    )
    with torch.inference_mode():
        fp32_logits = fp32_predictor.model(image_batch)
        int8_logits = int8_predictor.model(image_batch)

    if fp32_logits.shape != int8_logits.shape:
        raise ValueError("INT8 output shape does not match FP32 ONNX")
    difference = (fp32_logits - int8_logits).abs()
    fp32_top5 = fp32_logits.topk(5, dim=1).indices
    int8_top5 = int8_logits.topk(5, dim=1).indices
    top1_matches = fp32_top5[:, 0].eq(int8_top5[:, 0])
    top5_set_matches = [
        set(fp32_row.tolist()) == set(int8_row.tolist())
        for fp32_row, int8_row in zip(fp32_top5, int8_top5)
    ]
    finite = bool(torch.isfinite(int8_logits).all().item())
    formatted_predictions = predict_batch(int8_predictor, image_paths[:1])
    prediction_format_valid = (
        len(formatted_predictions) == 1
        and len(formatted_predictions[0]) == 5
        and set(formatted_predictions[0][0]) == {"species_name", "taxon_id", "score"}
    )
    report = {
        "candidate_backend": int8_predictor.backend,
        "finite_logits": finite,
        "image_count": len(image_paths),
        "max_absolute_logit_difference": float(difference.max().item()),
        "mean_absolute_logit_difference": float(difference.mean().item()),
        "passed": finite and prediction_format_valid,
        "prediction_format_valid": prediction_format_valid,
        "reference_backend": fp32_predictor.backend,
        "top1_match_count": int(top1_matches.sum().item()),
        "top1_match_rate": float(top1_matches.float().mean().item()),
        "top5_set_match_count": sum(top5_set_matches),
        "top5_set_match_rate": sum(top5_set_matches) / len(top5_set_matches),
    }
    _write_json(report_path, report)
    if not report["passed"]:
        raise RuntimeError(f"INT8 inference verification failed: {report}")
    return report


def register_int8_model(
    metadata: dict[str, Any],
    calibration_report: dict[str, Any],
    verification_report: dict[str, Any],
    *,
    model_path: Path = DEFAULT_ONNX_INT8_MODEL_PATH,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_registry_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)

    onnx_model = onnx.load(str(model_path))
    sample_input = np.zeros(
        (1, 3, int(metadata["image_size"]), int(metadata["image_size"])),
        dtype=np.float32,
    )
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    sample_output = session.run([OUTPUT_NAME], {INPUT_NAME: sample_input})[0]
    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run(run_name="onnxruntime-int8-static-quantization") as run:
        mlflow.log_params(
            {
                "activation_type": "QInt8",
                "backend": "onnxruntime-int8",
                "calibration_method": "MinMax",
                "calibration_sample_count": metadata["calibration"]["sample_count"],
                "calibration_seed": metadata["calibration"]["seed"],
                "calibration_source_split": "train",
                "class_count": metadata["class_count"],
                "format": "QDQ",
                "int8_size_bytes": metadata["artifact_size_bytes"],
                "per_channel": True,
                "source_fp32_onnx_sha256": metadata["source_fp32_onnx_sha256"],
                "source_fp32_size_bytes": metadata["source_fp32_size_bytes"],
                "weight_type": "QInt8",
            }
        )
        mlflow.log_metric(
            "model_size_reduction_ratio",
            1.0 - metadata["artifact_size_bytes"] / metadata["source_fp32_size_bytes"],
        )
        mlflow.set_tags(
            {
                "birdly.backend": "onnxruntime-int8",
                "birdly.calibration_split": "train",
                "birdly.class_mapping_sha256": metadata["class_mapping_sha256"],
                "birdly.int8_onnx_sha256": metadata["int8_onnx_sha256"],
                "birdly.quantization_method": "static-QDQ-QInt8",
                "birdly.source_fp32_onnx_sha256": metadata["source_fp32_onnx_sha256"],
            }
        )
        mlflow.log_dict(metadata, "onnx_int8_metadata.json")
        mlflow.log_dict(calibration_report, "onnx_int8_calibration.json")
        mlflow.log_dict(verification_report, "onnx_int8_verification.json")
        model_info = mlflow.onnx.log_model(
            onnx_model,
            name="model",
            registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
            signature=signature,
            await_registration_for=300,
            onnx_execution_providers=["CPUExecutionProvider"],
            metadata=metadata,
            save_as_external_data=False,
            extra_files=[
                str(DEFAULT_ONNX_INT8_METADATA_PATH),
                str(DEFAULT_CALIBRATION_PATH),
                str(DEFAULT_VERIFICATION_PATH),
                str(DEFAULT_CLASS_MAPPING_PATH),
                str(DEFAULT_PREPROCESSING_PATH),
            ],
            pip_requirements=[
                f"numpy=={np.__version__}",
                f"onnx=={onnx.__version__}",
                f"onnxruntime=={ort.__version__}",
            ],
        )
        version = str(model_info.registered_model_version)
        client = MlflowClient()
        version_tags = {
            "birdly.backend": "onnxruntime-int8",
            "birdly.calibration_split": "train",
            "birdly.class_mapping_sha256": metadata["class_mapping_sha256"],
            "birdly.int8_onnx_sha256": metadata["int8_onnx_sha256"],
            "birdly.quantization_method": "static-QDQ-QInt8",
            "birdly.source_fp32_onnx_sha256": metadata["source_fp32_onnx_sha256"],
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                DEFAULT_REGISTERED_MODEL_NAME,
                version,
                key,
                str(value),
            )
        client.set_registered_model_alias(
            DEFAULT_REGISTERED_MODEL_NAME,
            DEFAULT_REGISTERED_MODEL_ALIAS,
            version,
        )
        result = {
            "alias": DEFAULT_REGISTERED_MODEL_ALIAS,
            "model_uri": model_info.model_uri,
            "registered_model": DEFAULT_REGISTERED_MODEL_NAME,
            "registered_model_uri": (
                f"models:/{DEFAULT_REGISTERED_MODEL_NAME}"
                f"@{DEFAULT_REGISTERED_MODEL_ALIAS}"
            ),
            "run_id": run.info.run_id,
            "version": version,
        }
        mlflow.log_dict(result, "onnx_int8_registration.json")
    _write_json(DEFAULT_REGISTRATION_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize Birdly ONNX for CPU.")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=DEFAULT_CALIBRATION_BATCH_SIZE,
    )
    parser.add_argument(
        "--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--verification-images",
        type=int,
        default=DEFAULT_VERIFICATION_IMAGE_COUNT,
    )
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.calibration_batch_size < 1:
        parser.error("calibration batch size must be positive")
    if args.verification_images < 1:
        parser.error("verification image count must be positive")

    calibration_paths, calibration_report = select_calibration_samples(
        args.manifest,
        args.image_root,
        sample_count=args.calibration_samples,
        seed=args.calibration_seed,
    )
    calibration_report["batch_size"] = args.calibration_batch_size
    _write_json(DEFAULT_CALIBRATION_PATH, calibration_report)
    metadata = quantize_onnx_model(
        calibration_paths,
        calibration_report,
        calibration_batch_size=args.calibration_batch_size,
    )
    verification_report = verify_int8_inference(
        calibration_paths[: args.verification_images]
    )
    metadata["inference_verification"] = verification_report
    _write_json(DEFAULT_ONNX_INT8_METADATA_PATH, metadata)

    result: dict[str, Any] = {
        "metadata": metadata,
        "verification": verification_report,
    }
    if args.register:
        result["registration"] = register_int8_model(
            metadata,
            calibration_report,
            verification_report,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
