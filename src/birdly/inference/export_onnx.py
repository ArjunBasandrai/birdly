from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature

from birdly.inference.predict import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CLASS_MAPPING_PATH,
    DEFAULT_MODEL_DIRECTORY,
    DEFAULT_ONNX_METADATA_PATH,
    DEFAULT_ONNX_MODEL_PATH,
    DEFAULT_PREPROCESSING_PATH,
    load_predictor,
    preprocess_image,
)

DEFAULT_PARITY_PATH = DEFAULT_MODEL_DIRECTORY / "onnx" / "parity.json"
DEFAULT_REGISTRATION_PATH = DEFAULT_MODEL_DIRECTORY / "onnx" / "registration.json"
DEFAULT_TRACKING_URI = "sqlite:///B:/birdly-training/birdly-research-v3/mlflow.db"
DEFAULT_EXPERIMENT_NAME = "birdly-model-packages"
DEFAULT_REGISTERED_MODEL_NAME = "birdly-bird-classifier-onnx"
DEFAULT_REGISTERED_MODEL_ALIAS = "cpu-fp32"
ONNX_OPSET = 18
INPUT_NAME = "images"
OUTPUT_NAME = "logits"
MAX_DIFFERENCE_LIMIT = 1e-4
MEAN_DIFFERENCE_LIMIT = 1e-5


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


def export_onnx_model(
    output_path: Path = DEFAULT_ONNX_MODEL_PATH,
    metadata_path: Path = DEFAULT_ONNX_METADATA_PATH,
) -> dict[str, Any]:
    predictor = load_predictor(device="cpu")
    if not isinstance(predictor.model, torch.nn.Module):
        raise TypeError("ONNX export requires a PyTorch model")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    example_input = torch.zeros(
        (1, 3, predictor.image_size, predictor.image_size),
        dtype=torch.float32,
    )
    try:
        torch.onnx.export(
            predictor.model,
            (example_input,),
            temporary_path,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=ONNX_OPSET,
            dynamo=True,
            dynamic_shapes=({0: "batch"},),
            external_data=False,
            verbose=False,
        )
        onnx.checker.check_model(str(temporary_path))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    checkpoint = torch.load(
        DEFAULT_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    metadata = {
        "artifact_size_bytes": output_path.stat().st_size,
        "backend": "onnxruntime-fp32",
        "batch_dimension": "dynamic",
        "class_count": len(predictor.class_to_taxon),
        "class_mapping_path": str(DEFAULT_CLASS_MAPPING_PATH),
        "class_mapping_sha256": _file_sha256(DEFAULT_CLASS_MAPPING_PATH),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "export_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "image_size": predictor.image_size,
        "input_name": INPUT_NAME,
        "input_shape": ["batch", 3, predictor.image_size, predictor.image_size],
        "model_name": str(checkpoint["config"]["model"]["name"]),
        "onnx_path": str(output_path),
        "onnx_sha256": _file_sha256(output_path),
        "onnx_version": onnx.__version__,
        "onnxruntime_execution_provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "opset": ONNX_OPSET,
        "output_name": OUTPUT_NAME,
        "output_shape": ["batch", len(predictor.class_to_taxon)],
        "precision": "float32",
        "preprocessing": predictor.preprocessing,
        "preprocessing_path": str(DEFAULT_PREPROCESSING_PATH),
        "preprocessing_sha256": _file_sha256(DEFAULT_PREPROCESSING_PATH),
        "source_checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
        "source_checkpoint_sha256": _file_sha256(DEFAULT_CHECKPOINT_PATH),
        "source_training_run_id": "77f7cd76e2cf4ed0829c6ebbf8de8b65",
        "torch_version": str(torch.__version__),
    }
    _write_json(metadata_path, metadata)
    return metadata


def verify_onnx_parity(
    image_paths: Sequence[Path],
    *,
    report_path: Path = DEFAULT_PARITY_PATH,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("At least one parity image is required")
    pytorch_predictor = load_predictor(device="cpu")
    onnx_predictor = load_predictor(
        backend="onnxruntime",
        device="cpu",
        cpu_threads=1,
    )
    image_batch = torch.stack(
        [preprocess_image(pytorch_predictor, path) for path in image_paths]
    )
    with torch.inference_mode():
        pytorch_logits = pytorch_predictor.model(image_batch)
        onnx_logits = onnx_predictor.model(image_batch)

    if pytorch_logits.shape != onnx_logits.shape:
        raise ValueError("ONNX output shape does not match PyTorch")
    difference = (pytorch_logits - onnx_logits).abs()
    pytorch_top5 = pytorch_logits.topk(5, dim=1).indices
    onnx_top5 = onnx_logits.topk(5, dim=1).indices
    max_difference = float(difference.max().item())
    mean_difference = float(difference.mean().item())
    top1_match = bool(torch.equal(pytorch_top5[:, :1], onnx_top5[:, :1]))
    top5_match = bool(torch.equal(pytorch_top5, onnx_top5))
    passed = (
        max_difference <= MAX_DIFFERENCE_LIMIT
        and mean_difference <= MEAN_DIFFERENCE_LIMIT
        and top1_match
        and top5_match
    )
    report = {
        "image_count": len(image_paths),
        "max_difference": max_difference,
        "max_difference_limit": MAX_DIFFERENCE_LIMIT,
        "mean_difference": mean_difference,
        "mean_difference_limit": MEAN_DIFFERENCE_LIMIT,
        "passed": passed,
        "top1_match": top1_match,
        "top5_match": top5_match,
    }
    _write_json(report_path, report)
    if not passed:
        raise RuntimeError(f"ONNX output parity failed: {report}")
    return report


def register_onnx_model(
    metadata: dict[str, Any],
    parity: dict[str, Any],
    *,
    model_path: Path = DEFAULT_ONNX_MODEL_PATH,
    metadata_path: Path = DEFAULT_ONNX_METADATA_PATH,
    parity_path: Path = DEFAULT_PARITY_PATH,
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

    with mlflow.start_run(run_name="onnxruntime-fp32-export") as run:
        mlflow.log_params(
            {
                "backend": "onnxruntime",
                "class_count": metadata["class_count"],
                "dynamic_batch": True,
                "image_size": metadata["image_size"],
                "opset": metadata["opset"],
                "precision": "float32",
                "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
                "source_training_run_id": metadata["source_training_run_id"],
            }
        )
        mlflow.set_tags(
            {
                "birdly.backend": "onnxruntime-fp32",
                "birdly.class_mapping_sha256": metadata["class_mapping_sha256"],
                "birdly.onnx_sha256": metadata["onnx_sha256"],
                "birdly.output_parity": "passed",
                "birdly.source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
                "birdly.source_training_run_id": "77f7cd76e2cf4ed0829c6ebbf8de8b65",
            }
        )
        mlflow.log_dict(metadata, "onnx_export_metadata.json")
        mlflow.log_dict(parity, "onnx_output_parity.json")
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
                str(metadata_path),
                str(parity_path),
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
        for key, value in {
            "birdly.backend": "onnxruntime-fp32",
            "birdly.class_mapping_sha256": metadata["class_mapping_sha256"],
            "birdly.onnx_sha256": metadata["onnx_sha256"],
            "birdly.output_parity": "passed",
            "birdly.source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
            "birdly.source_training_run_id": metadata["source_training_run_id"],
        }.items():
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
        mlflow.log_dict(result, "onnx_registration.json")
    _write_json(DEFAULT_REGISTRATION_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and register Birdly ONNX.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    metadata = export_onnx_model()
    parity = verify_onnx_parity(args.images)
    metadata["output_parity"] = parity
    _write_json(DEFAULT_ONNX_METADATA_PATH, metadata)
    result: dict[str, Any] = {
        "metadata": metadata,
        "parity": parity,
    }
    if args.register:
        result["registration"] = register_onnx_model(metadata, parity)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
