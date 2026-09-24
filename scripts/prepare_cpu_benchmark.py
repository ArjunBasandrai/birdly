from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIRECTORY = REPOSITORY_ROOT / "deploy" / "cpu-benchmark"
BUILD_DIRECTORY = DEPLOYMENT_DIRECTORY / "build"
DEPLOYMENT_FILES = ("Dockerfile", "requirements.txt", ".dockerignore")
SOURCE_FILES = (
    "src/birdly/__init__.py",
    "src/birdly/inference/__init__.py",
    "src/birdly/inference/cpu_benchmark.py",
    "src/birdly/inference/onnx_backend.py",
    "src/birdly/inference/predict.py",
    "src/birdly/inference/taxon_names.json",
    "src/birdly/training/__init__.py",
    "src/birdly/training/model.py",
)
MODEL_FILES = (
    "best.pt",
    "class_mapping.json",
    "preprocessing.json",
    "onnx/birdly-fp32.onnx",
    "onnx/metadata.json",
    "onnx/birdly-int8.onnx",
    "onnx/metadata-int8.json",
)
PORTABLE_PATHS = {
    "class_mapping_path": "class_mapping.json",
    "int8_onnx_path": "onnx/birdly-int8.onnx",
    "onnx_path": "onnx/birdly-fp32.onnx",
    "preprocessing_path": "preprocessing.json",
    "source_checkpoint_path": "best.pt",
    "source_fp32_onnx_path": "onnx/birdly-fp32.onnx",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def _validate_model_bundle(model_source: Path) -> None:
    missing = [name for name in MODEL_FILES if not (model_source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Benchmark model bundle is missing: " + ", ".join(missing)
        )

    paths = {name: model_source / name for name in MODEL_FILES}
    fp32_metadata = _read_json(paths["onnx/metadata.json"])
    int8_metadata = _read_json(paths["onnx/metadata-int8.json"])
    class_mapping = _read_json(paths["class_mapping.json"])
    preprocessing = _read_json(paths["preprocessing.json"])
    actual_hashes = {
        "checkpoint": _sha256(paths["best.pt"]),
        "class_mapping": _sha256(paths["class_mapping.json"]),
        "preprocessing": _sha256(paths["preprocessing.json"]),
        "fp32": _sha256(paths["onnx/birdly-fp32.onnx"]),
        "int8": _sha256(paths["onnx/birdly-int8.onnx"]),
    }
    expected_values = (
        (fp32_metadata.get("backend"), "onnxruntime-fp32", "FP32 backend"),
        (int8_metadata.get("backend"), "onnxruntime-int8", "INT8 backend"),
        (
            fp32_metadata.get("source_checkpoint_sha256"),
            actual_hashes["checkpoint"],
            "FP32 source checkpoint hash",
        ),
        (
            int8_metadata.get("source_checkpoint_sha256"),
            actual_hashes["checkpoint"],
            "INT8 source checkpoint hash",
        ),
        (
            fp32_metadata.get("class_mapping_sha256"),
            actual_hashes["class_mapping"],
            "class mapping hash",
        ),
        (
            fp32_metadata.get("preprocessing_sha256"),
            actual_hashes["preprocessing"],
            "preprocessing hash",
        ),
        (
            fp32_metadata.get("onnx_sha256"),
            actual_hashes["fp32"],
            "FP32 ONNX hash",
        ),
        (
            int8_metadata.get("int8_onnx_sha256"),
            actual_hashes["int8"],
            "INT8 ONNX hash",
        ),
        (
            int8_metadata.get("source_fp32_onnx_sha256"),
            actual_hashes["fp32"],
            "INT8 source ONNX hash",
        ),
        (
            int(fp32_metadata["class_count"]),
            int(class_mapping["class_count"]),
            "class count",
        ),
        (
            preprocessing.get("source_checkpoint_sha256"),
            actual_hashes["checkpoint"],
            "preprocessing source checkpoint hash",
        ),
    )
    for actual, expected, label in expected_values:
        if actual != expected:
            raise ValueError(f"Invalid {label}")


def _portable_metadata(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _portable_metadata(item, item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    if key is not None and key.endswith("_path"):
        return PORTABLE_PATHS.get(key)
    return value


def _prepare_build(model_source: Path, input_image: Path) -> Path:
    model_source = model_source.expanduser().resolve()
    input_image = input_image.expanduser().resolve()
    _validate_model_bundle(model_source)
    if not input_image.is_file():
        raise FileNotFoundError(f"Representative image is missing: {input_image}")
    prepared_at = datetime.now(UTC).isoformat()
    commit = _git_commit()
    DEPLOYMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(prefix="build-", dir=DEPLOYMENT_DIRECTORY)
    )
    try:
        for name in DEPLOYMENT_FILES:
            _copy(DEPLOYMENT_DIRECTORY / name, staging_directory / name)
        for name in SOURCE_FILES:
            _copy(REPOSITORY_ROOT / name, staging_directory / name)
        for name in MODEL_FILES:
            source = model_source / name
            destination = staging_directory / "model" / name
            if name.endswith(("metadata.json", "metadata-int8.json")):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(
                        _portable_metadata(_read_json(source)), indent=2, sort_keys=True
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                _copy(source, destination)
        packaged_input = staging_directory / "input" / "representative-image"
        _copy(input_image, packaged_input)

        packaged_files = {}
        for name in MODEL_FILES:
            path = staging_directory / "model" / name
            packaged_files[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        manifest = {
            "backends": [
                "pytorch-eager",
                "pytorch-compile",
                "onnxruntime-fp32",
                "onnxruntime-int8",
            ],
            "git_commit": commit,
            "input_image": {
                "bytes": packaged_input.stat().st_size,
                "sha256": _sha256(packaged_input),
            },
            "model_files": packaged_files,
            "prepared_at_utc": prepared_at,
            "python_base_image": "python:3.12-slim",
        }
        (staging_directory / "build-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if BUILD_DIRECTORY.resolve().parent != DEPLOYMENT_DIRECTORY.resolve():
            raise RuntimeError("Invalid CPU benchmark build directory")
        if BUILD_DIRECTORY.exists():
            shutil.rmtree(BUILD_DIRECTORY)
        shutil.copytree(staging_directory, BUILD_DIRECTORY)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
    return BUILD_DIRECTORY


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the self-contained Birdly CPU benchmark image context."
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=os.getenv("BIRDLY_BENCHMARK_MODEL_SOURCE"),
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=os.getenv("BIRDLY_BENCHMARK_INPUT_IMAGE"),
    )
    args = parser.parse_args()
    if args.model_source is None or args.input_image is None:
        parser.error(
            "--model-source and --input-image are required unless their environment variables are set"
        )
    build_directory = _prepare_build(args.model_source, args.input_image)
    model_bytes = sum(
        path.stat().st_size
        for path in (build_directory / "model").rglob("*")
        if path.is_file()
    )
    print(f"CPU benchmark build context: {build_directory}")
    print(f"Packaged model bytes: {model_bytes}")


if __name__ == "__main__":
    main()
