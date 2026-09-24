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
DEPLOYMENT_DIRECTORY = REPOSITORY_ROOT / "deploy" / "cloud-run"
BUILD_DIRECTORY = DEPLOYMENT_DIRECTORY / "build"
DEPLOYMENT_FILES = (
    "Dockerfile",
    "requirements.txt",
    ".dockerignore",
    ".gcloudignore",
)
SOURCE_FILES = (
    "src/birdly/__init__.py",
    "src/birdly/api/__init__.py",
    "src/birdly/api/app.py",
    "src/birdly/api/config.py",
    "src/birdly/inference/__init__.py",
    "src/birdly/inference/onnx_backend.py",
    "src/birdly/inference/predict.py",
    "src/birdly/inference/species_names.json",
)
MODEL_FILES = (
    "class_mapping.json",
    "preprocessing.json",
    "onnx/birdly-fp32.onnx",
    "onnx/metadata.json",
)


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


def _validate_model_bundle(model_source: Path) -> dict[str, Any]:
    missing = [name for name in MODEL_FILES if not (model_source / name).is_file()]
    if missing:
        raise FileNotFoundError("Model bundle is missing: " + ", ".join(missing))

    class_mapping_path = model_source / "class_mapping.json"
    preprocessing_path = model_source / "preprocessing.json"
    onnx_path = model_source / "onnx" / "birdly-fp32.onnx"
    metadata_path = model_source / "onnx" / "metadata.json"
    class_mapping = _read_json(class_mapping_path)
    preprocessing = _read_json(preprocessing_path)
    metadata = _read_json(metadata_path)

    expected_hashes = {
        "class_mapping_sha256": _sha256(class_mapping_path),
        "preprocessing_sha256": _sha256(preprocessing_path),
        "onnx_sha256": _sha256(onnx_path),
    }
    for key, actual_hash in expected_hashes.items():
        if metadata.get(key) != actual_hash:
            raise ValueError(f"Model metadata has an invalid {key}")
    if metadata.get("backend") != "onnxruntime-fp32":
        raise ValueError("The production model must use ONNX Runtime FP32")
    if metadata.get("precision") != "float32":
        raise ValueError("The production model must use FP32")
    if int(metadata["class_count"]) != int(class_mapping["class_count"]):
        raise ValueError("The model and class mapping counts do not match")
    source_checkpoint_hash = metadata.get("source_checkpoint_sha256")
    if source_checkpoint_hash != preprocessing.get("source_checkpoint_sha256"):
        raise ValueError("The preprocessing and model source checkpoints do not match")
    return metadata


def _portable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    result["class_mapping_path"] = "class_mapping.json"
    result["preprocessing_path"] = "preprocessing.json"
    result["onnx_path"] = "onnx/birdly-fp32.onnx"
    result["source_checkpoint_path"] = None
    result["packaged_for"] = "google-cloud-run"
    return result


def _prepare_build(model_source: Path) -> Path:
    model_source = model_source.expanduser().resolve()
    metadata = _validate_model_bundle(model_source)
    prepared_at = datetime.now(UTC).isoformat()
    commit = _git_commit()
    DEPLOYMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(prefix="build-", dir=DEPLOYMENT_DIRECTORY)
    )
    try:
        for name in DEPLOYMENT_FILES:
            _copy(DEPLOYMENT_DIRECTORY / name, staging_directory / name)
        _copy(REPOSITORY_ROOT / "configs" / "api.yaml", staging_directory / "api.yaml")
        for name in SOURCE_FILES:
            _copy(REPOSITORY_ROOT / name, staging_directory / name)

        for name in MODEL_FILES:
            if name == "onnx/metadata.json":
                continue
            _copy(model_source / name, staging_directory / "model" / name)
        metadata_destination = staging_directory / "model" / "onnx" / "metadata.json"
        metadata_destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_destination.write_text(
            json.dumps(_portable_metadata(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        packaged_files = {}
        for name in MODEL_FILES:
            path = staging_directory / "model" / name
            packaged_files[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        manifest = {
            "backend": "onnxruntime-fp32",
            "git_commit": commit,
            "model_files": packaged_files,
            "prepared_at_utc": prepared_at,
            "python_base_image": "python:3.12-slim",
        }
        (staging_directory / "build-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if BUILD_DIRECTORY.resolve().parent != DEPLOYMENT_DIRECTORY.resolve():
            raise RuntimeError("Invalid Cloud Run build directory")
        if BUILD_DIRECTORY.exists():
            shutil.rmtree(BUILD_DIRECTORY)
        shutil.copytree(staging_directory, BUILD_DIRECTORY)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
    return BUILD_DIRECTORY


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the self-contained Birdly Cloud Run build context."
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=os.getenv("BIRDLY_MODEL_BUNDLE_SOURCE"),
        help="Directory with the production ONNX FP32 model bundle.",
    )
    args = parser.parse_args()
    if args.model_source is None:
        parser.error("--model-source or BIRDLY_MODEL_BUNDLE_SOURCE is required")
    build_directory = _prepare_build(args.model_source)
    model_bytes = sum(
        path.stat().st_size
        for path in (build_directory / "model").rglob("*")
        if path.is_file()
    )
    print(f"Cloud Run build context: {build_directory}")
    print(f"Packaged model bytes: {model_bytes}")


if __name__ == "__main__":
    main()
