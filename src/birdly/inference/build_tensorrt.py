from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import onnx
import tensorrt as trt
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from onnxruntime.transformers.float16 import convert_float_to_float16

from birdly.inference.export_onnx import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    INPUT_NAME,
    OUTPUT_NAME,
)
from birdly.inference.predict import (
    DEFAULT_CLASS_MAPPING_PATH,
    DEFAULT_ONNX_DIRECTORY,
    DEFAULT_ONNX_METADATA_PATH,
    DEFAULT_ONNX_MODEL_PATH,
    DEFAULT_PREPROCESSING_PATH,
    DEFAULT_TENSORRT_ENGINE_PATH,
    DEFAULT_TENSORRT_METADATA_PATH,
    load_predictor,
    predict_batch,
    predict_logits,
)

DEFAULT_PARITY_PATH = DEFAULT_ONNX_DIRECTORY / "parity-tensorrt-fp16.json"
DEFAULT_REGISTRATION_PATH = DEFAULT_ONNX_DIRECTORY / "registration-tensorrt-fp16.json"
DEFAULT_REGISTERED_MODEL_NAME = "birdly-bird-classifier-tensorrt"
DEFAULT_REGISTERED_MODEL_ALIAS = "gpu-fp16"
DEFAULT_MINIMUM_BATCH_SIZE = 1
DEFAULT_OPTIMUM_BATCH_SIZE = 8
DEFAULT_MAXIMUM_BATCH_SIZE = 32
DEFAULT_WORKSPACE_BYTES = 512 * 1024 * 1024


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


def build_tensorrt_engine(
    *,
    source_path: Path = DEFAULT_ONNX_MODEL_PATH,
    engine_path: Path = DEFAULT_TENSORRT_ENGINE_PATH,
    metadata_path: Path = DEFAULT_TENSORRT_METADATA_PATH,
    minimum_batch_size: int = DEFAULT_MINIMUM_BATCH_SIZE,
    optimum_batch_size: int = DEFAULT_OPTIMUM_BATCH_SIZE,
    maximum_batch_size: int = DEFAULT_MAXIMUM_BATCH_SIZE,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> dict[str, Any]:
    if not 1 <= minimum_batch_size <= optimum_batch_size <= maximum_batch_size:
        raise ValueError("TensorRT batch profile is invalid")
    if workspace_bytes < 1:
        raise ValueError("TensorRT workspace must be positive")
    if not source_path.is_file():
        raise FileNotFoundError(f"FP32 ONNX model is missing: {source_path}")

    source_metadata = json.loads(DEFAULT_ONNX_METADATA_PATH.read_text(encoding="utf-8"))
    source_hash = _file_sha256(source_path)
    if source_metadata["onnx_sha256"] != source_hash:
        raise ValueError("FP32 ONNX model hash does not match its metadata")

    source_model = onnx.load(str(source_path), load_external_data=False)
    fp16_model = convert_float_to_float16(
        source_model,
        keep_io_types=False,
        disable_shape_infer=False,
    )
    onnx.checker.check_model(fp16_model)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(fp16_model.SerializeToString()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parsing failed: {errors}")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError("Birdly TensorRT must have one input and one output")
    network_input = network.get_input(0)
    network_output = network.get_output(0)
    if network_input.name != INPUT_NAME or network_output.name != OUTPUT_NAME:
        raise ValueError("TensorRT input or output name is unexpected")
    if network_input.dtype != trt.float16 or network_output.dtype != trt.float16:
        raise ValueError("TensorRT input and output must use FP16")

    image_size = int(source_metadata["image_size"])
    minimum_shape = (minimum_batch_size, 3, image_size, image_size)
    optimum_shape = (optimum_batch_size, 3, image_size, image_size)
    maximum_shape = (maximum_batch_size, 3, image_size, image_size)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        minimum_shape,
        optimum_shape,
        maximum_shape,
    )
    if tuple(profile.get_shape(INPUT_NAME)) != (
        minimum_shape,
        optimum_shape,
        maximum_shape,
    ):
        raise RuntimeError("TensorRT rejected the optimization profile")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.add_optimization_profile(profile)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = engine_path.with_suffix(engine_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    build_started = time.perf_counter()
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build failed")
        temporary_path.write_bytes(bytes(serialized_engine))
        os.replace(temporary_path, engine_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    build_seconds = time.perf_counter() - build_started

    device = torch.device("cuda", torch.cuda.current_device())
    metadata = {
        "backend": "tensorrt-fp16",
        "build_git_commit": _git_commit(),
        "build_seconds": build_seconds,
        "class_count": source_metadata["class_count"],
        "class_mapping_path": str(DEFAULT_CLASS_MAPPING_PATH),
        "class_mapping_sha256": source_metadata["class_mapping_sha256"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "engine_path": str(engine_path),
        "engine_sha256": _file_sha256(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_name": torch.cuda.get_device_name(device),
        "image_size": image_size,
        "input_dtype": "float16",
        "input_name": INPUT_NAME,
        "input_profile": {
            "maximum": list(maximum_shape),
            "minimum": list(minimum_shape),
            "optimum": list(optimum_shape),
        },
        "model_name": source_metadata["model_name"],
        "onnx_fp16_conversion": (
            "onnxruntime.transformers.float16.convert_float_to_float16"
        ),
        "output_dtype": "float16",
        "output_name": OUTPUT_NAME,
        "precision": "float16",
        "preprocessing": source_metadata["preprocessing"],
        "preprocessing_path": str(DEFAULT_PREPROCESSING_PATH),
        "preprocessing_sha256": source_metadata["preprocessing_sha256"],
        "source_checkpoint_path": source_metadata["source_checkpoint_path"],
        "source_checkpoint_sha256": source_metadata["source_checkpoint_sha256"],
        "source_fp32_onnx_path": str(source_path),
        "source_fp32_onnx_sha256": source_hash,
        "strongly_typed_network": True,
        "tensorrt_version": trt.__version__,
        "workspace_bytes": workspace_bytes,
    }
    _write_json(metadata_path, metadata)
    return metadata


def verify_tensorrt_parity(
    image_paths: Sequence[Path],
    *,
    report_path: Path = DEFAULT_PARITY_PATH,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("At least one parity image is required")
    pytorch_predictor = load_predictor(backend="pytorch", device="cuda")
    tensorrt_predictor = load_predictor(backend="tensorrt", device="cuda")
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        pytorch_logits = predict_logits(pytorch_predictor, image_paths)
        tensorrt_logits = predict_logits(tensorrt_predictor, image_paths)
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32

    if pytorch_logits.shape != tensorrt_logits.shape:
        raise ValueError("TensorRT output shape does not match PyTorch")
    difference = (pytorch_logits - tensorrt_logits).abs()
    pytorch_top5 = pytorch_logits.topk(5, dim=1).indices
    tensorrt_top5 = tensorrt_logits.topk(5, dim=1).indices
    top1_match = bool(torch.equal(pytorch_top5[:, :1], tensorrt_top5[:, :1]))
    top5_match = bool(torch.equal(pytorch_top5, tensorrt_top5))
    finite = bool(torch.isfinite(tensorrt_logits).all().item())
    predictions = predict_batch(tensorrt_predictor, image_paths[:1])
    prediction_format_valid = (
        len(predictions) == 1
        and len(predictions[0]) <= 5
        and all(
            set(prediction) == {"species_name", "taxon_id", "score"}
            for prediction in predictions[0]
        )
    )
    report = {
        "candidate_backend": tensorrt_predictor.backend,
        "finite_logits": finite,
        "image_count": len(image_paths),
        "max_absolute_logit_difference": float(difference.max().item()),
        "mean_absolute_logit_difference": float(difference.mean().item()),
        "passed": finite and prediction_format_valid and top1_match and top5_match,
        "prediction_format_valid": prediction_format_valid,
        "reference_backend": pytorch_predictor.backend,
        "top1_match": top1_match,
        "top5_ordered_match": top5_match,
    }
    _write_json(report_path, report)
    if not report["passed"]:
        raise RuntimeError(f"TensorRT output parity failed: {report}")
    return report


def register_tensorrt_engine(
    metadata: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_registry_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
    client = MlflowClient()
    try:
        client.get_registered_model(DEFAULT_REGISTERED_MODEL_NAME)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        client.create_registered_model(DEFAULT_REGISTERED_MODEL_NAME)

    with mlflow.start_run(run_name="tensorrt-fp16-build") as run:
        mlflow.log_params(
            {
                "backend": metadata["backend"],
                "engine_size_bytes": metadata["engine_size_bytes"],
                "maximum_batch_size": metadata["input_profile"]["maximum"][0],
                "minimum_batch_size": metadata["input_profile"]["minimum"][0],
                "optimum_batch_size": metadata["input_profile"]["optimum"][0],
                "precision": metadata["precision"],
                "source_fp32_onnx_sha256": metadata["source_fp32_onnx_sha256"],
                "tensorrt_version": metadata["tensorrt_version"],
            }
        )
        mlflow.log_metric("engine_build_seconds", metadata["build_seconds"])
        mlflow.set_tags(
            {
                "birdly.backend": metadata["backend"],
                "birdly.engine_sha256": metadata["engine_sha256"],
                "birdly.model_format": "tensorrt-engine",
                "birdly.output_parity": "passed",
                "birdly.source_fp32_onnx_sha256": metadata["source_fp32_onnx_sha256"],
            }
        )
        mlflow.log_artifact(str(DEFAULT_TENSORRT_ENGINE_PATH), "tensorrt")
        mlflow.log_artifact(str(DEFAULT_TENSORRT_METADATA_PATH), "tensorrt")
        mlflow.log_artifact(str(DEFAULT_PARITY_PATH), "tensorrt")
        source = f"{run.info.artifact_uri}/tensorrt"
        version = client.create_model_version(
            name=DEFAULT_REGISTERED_MODEL_NAME,
            source=source,
            run_id=run.info.run_id,
            description="Birdly TensorRT FP16 engine and metadata",
        )
        for key, value in {
            "birdly.backend": metadata["backend"],
            "birdly.engine_sha256": metadata["engine_sha256"],
            "birdly.model_format": "tensorrt-engine",
            "birdly.output_parity": "passed",
            "birdly.source_fp32_onnx_sha256": metadata["source_fp32_onnx_sha256"],
            "birdly.tensorrt_version": metadata["tensorrt_version"],
        }.items():
            client.set_model_version_tag(
                DEFAULT_REGISTERED_MODEL_NAME,
                version.version,
                key,
                str(value),
            )
        client.set_registered_model_alias(
            DEFAULT_REGISTERED_MODEL_NAME,
            DEFAULT_REGISTERED_MODEL_ALIAS,
            version.version,
        )
        result = {
            "alias": DEFAULT_REGISTERED_MODEL_ALIAS,
            "artifact_uri": source,
            "registered_model": DEFAULT_REGISTERED_MODEL_NAME,
            "registered_model_uri": (
                f"models:/{DEFAULT_REGISTERED_MODEL_NAME}"
                f"@{DEFAULT_REGISTERED_MODEL_ALIAS}"
            ),
            "run_id": run.info.run_id,
            "version": str(version.version),
        }
        _write_json(DEFAULT_REGISTRATION_PATH, result)
        mlflow.log_artifact(str(DEFAULT_REGISTRATION_PATH), "tensorrt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Birdly TensorRT FP16.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    metadata = build_tensorrt_engine()
    parity = verify_tensorrt_parity(args.images)
    metadata["output_parity"] = parity
    _write_json(DEFAULT_TENSORRT_METADATA_PATH, metadata)
    result: dict[str, Any] = {"metadata": metadata, "parity": parity}
    if args.register:
        result["registration"] = register_tensorrt_engine(metadata, parity)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
