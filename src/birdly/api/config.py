from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_CPU_BACKENDS = {"onnxruntime", "onnxruntime-int8"}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return payload


def _positive_integer(value: Any, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class ApiSettings:
    backend: str
    model_directory: Path
    rate_limit_per_minute: int
    maximum_upload_bytes: int
    maximum_image_width: int
    maximum_image_height: int
    request_timeout_seconds: float
    maximum_inference_concurrency: int
    cpu_threads: int
    allowed_cors_origins: tuple[str, ...]

    @property
    def class_mapping_path(self) -> Path:
        return self.model_directory / "class_mapping.json"

    @property
    def preprocessing_path(self) -> Path:
        return self.model_directory / "preprocessing.json"

    @property
    def onnx_model_path(self) -> Path:
        return self.model_directory / "onnx" / "birdly-fp32.onnx"

    @property
    def onnx_metadata_path(self) -> Path:
        return self.model_directory / "onnx" / "metadata.json"

    @property
    def onnx_int8_model_path(self) -> Path:
        return self.model_directory / "onnx" / "birdly-int8.onnx"

    @property
    def onnx_int8_metadata_path(self) -> Path:
        return self.model_directory / "onnx" / "metadata-int8.json"


def load_api_settings(
    config_path: str | Path | None = None,
    local_config_path: str | Path | None = None,
) -> ApiSettings:
    selected_config_path = Path(
        config_path or os.getenv("BIRDLY_API_CONFIG", "configs/api.yaml")
    )
    values = _read_yaml(selected_config_path)

    selected_local_path = Path(
        local_config_path or os.getenv("BIRDLY_LOCAL_CONFIG", ".birdly/local.yaml")
    )
    if selected_local_path.is_file():
        local_values = _read_yaml(selected_local_path).get("api", {})
        if not isinstance(local_values, dict):
            raise TypeError("The local API configuration must be a mapping")
        values.update(local_values)

    environment_values: dict[str, tuple[str, Any]] = {
        "BIRDLY_API_BACKEND": ("backend", str),
        "BIRDLY_MODEL_DIR": ("model_directory", Path),
        "BIRDLY_RATE_LIMIT_PER_MINUTE": ("rate_limit_per_minute", int),
        "BIRDLY_MAX_UPLOAD_BYTES": ("maximum_upload_bytes", int),
        "BIRDLY_MAX_IMAGE_WIDTH": ("maximum_image_width", int),
        "BIRDLY_MAX_IMAGE_HEIGHT": ("maximum_image_height", int),
        "BIRDLY_REQUEST_TIMEOUT_SECONDS": ("request_timeout_seconds", float),
        "BIRDLY_MAX_INFERENCE_CONCURRENCY": (
            "maximum_inference_concurrency",
            int,
        ),
        "BIRDLY_CPU_THREADS": ("cpu_threads", int),
    }
    for environment_name, (setting_name, converter) in environment_values.items():
        if environment_name in os.environ:
            values[setting_name] = converter(os.environ[environment_name])
    if "BIRDLY_CORS_ORIGINS" in os.environ:
        values["allowed_cors_origins"] = [
            value.strip()
            for value in os.environ["BIRDLY_CORS_ORIGINS"].split(",")
            if value.strip()
        ]

    required = {
        "backend",
        "model_directory",
        "rate_limit_per_minute",
        "maximum_upload_bytes",
        "maximum_image_width",
        "maximum_image_height",
        "request_timeout_seconds",
        "maximum_inference_concurrency",
        "cpu_threads",
        "allowed_cors_origins",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError("API configuration is missing: " + ", ".join(sorted(missing)))

    backend = str(values["backend"])
    if backend not in SUPPORTED_CPU_BACKENDS:
        raise ValueError(
            "API backend must be one of: " + ", ".join(sorted(SUPPORTED_CPU_BACKENDS))
        )
    origins = values["allowed_cors_origins"]
    if not isinstance(origins, list) or not all(
        isinstance(origin, str) and origin for origin in origins
    ):
        raise TypeError("allowed_cors_origins must be a list of origins")
    if "*" in origins:
        raise ValueError("CORS origins must be explicit")

    return ApiSettings(
        backend=backend,
        model_directory=Path(str(values["model_directory"])),
        rate_limit_per_minute=_nonnegative_integer(
            values["rate_limit_per_minute"], "rate_limit_per_minute"
        ),
        maximum_upload_bytes=_positive_integer(
            values["maximum_upload_bytes"], "maximum_upload_bytes"
        ),
        maximum_image_width=_positive_integer(
            values["maximum_image_width"], "maximum_image_width"
        ),
        maximum_image_height=_positive_integer(
            values["maximum_image_height"], "maximum_image_height"
        ),
        request_timeout_seconds=_positive_float(
            values["request_timeout_seconds"], "request_timeout_seconds"
        ),
        maximum_inference_concurrency=_positive_integer(
            values["maximum_inference_concurrency"],
            "maximum_inference_concurrency",
        ),
        cpu_threads=_positive_integer(values["cpu_threads"], "cpu_threads"),
        allowed_cors_origins=tuple(origins),
    )
