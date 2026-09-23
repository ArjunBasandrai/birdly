from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return payload


def load_training_config(path: str | Path = "configs/train.yaml") -> dict[str, Any]:
    config = load_yaml(path)
    required_sections = {
        "model",
        "image",
        "augmentation",
        "training",
        "evaluation",
        "safety",
    }
    missing_sections = required_sections.difference(config)
    if missing_sections:
        raise ValueError(
            "Training configuration is missing sections: "
            + ", ".join(sorted(missing_sections))
        )
    return config


def load_local_paths(path: str | Path = ".birdly/local.yaml") -> dict[str, Path]:
    config = load_yaml(path)
    required = {"image_data_root", "training_runs_root", "torch_cache_root"}
    missing = required.difference(config)
    if missing:
        raise ValueError(
            "Local configuration is missing paths: " + ", ".join(sorted(missing))
        )
    return {key: Path(str(config[key])) for key in required}
