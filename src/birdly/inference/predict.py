from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from torchvision import transforms
except ModuleNotFoundError:
    transforms = None

from birdly.inference.onnx_backend import load_onnx_runtime_model

DEFAULT_MODEL_DIRECTORY = Path(os.getenv("BIRDLY_MODEL_DIR", "artifacts/api-model"))
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODEL_DIRECTORY / "best.pt"
DEFAULT_CLASS_MAPPING_PATH = DEFAULT_MODEL_DIRECTORY / "class_mapping.json"
DEFAULT_TAXON_NAMES_PATH = Path(__file__).with_name("taxon_names.json")
DEFAULT_PREPROCESSING_PATH = DEFAULT_MODEL_DIRECTORY / "preprocessing.json"
DEFAULT_ONNX_DIRECTORY = DEFAULT_MODEL_DIRECTORY / "onnx"
DEFAULT_ONNX_MODEL_PATH = DEFAULT_ONNX_DIRECTORY / "birdly-fp32.onnx"
DEFAULT_ONNX_METADATA_PATH = DEFAULT_ONNX_DIRECTORY / "metadata.json"
DEFAULT_ONNX_INT8_MODEL_PATH = DEFAULT_ONNX_DIRECTORY / "birdly-int8.onnx"
DEFAULT_ONNX_INT8_METADATA_PATH = DEFAULT_ONNX_DIRECTORY / "metadata-int8.json"
DEFAULT_TENSORRT_ENGINE_PATH = DEFAULT_ONNX_DIRECTORY / "birdly-fp16.engine"
DEFAULT_TENSORRT_METADATA_PATH = DEFAULT_ONNX_DIRECTORY / "metadata-tensorrt-fp16.json"
DEFAULT_COMPILE_MODE = "default"
MINIMUM_TOP_CONFIDENCE = 0.50
CUMULATIVE_CONFIDENCE_TARGET = 0.90
MAXIMUM_RETURNED_PREDICTIONS = 5
InferenceBackendName = Literal[
    "pytorch",
    "pytorch-fp16",
    "onnxruntime",
    "onnxruntime-cuda",
    "onnxruntime-int8",
    "tensorrt",
]


class Prediction(TypedDict):
    taxon_id: int
    common_name: str
    scientific_name: str
    score: float


class TaxonName(TypedDict):
    common_name: str
    scientific_name: str


@dataclass(frozen=True)
class Predictor:
    model: Callable[[Any], Any]
    transform: Callable[[Image.Image], Any]
    class_to_taxon: dict[int, int]
    taxon_to_names: dict[int, TaxonName]
    device: Any
    model_name: str
    backend: str
    precision: str
    compile_mode: str | None
    autocast_dtype: Any | None
    image_size: int
    preprocessing: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("This inference backend requires PyTorch")
    return torch


def _numpy_transform(preprocessing: dict[str, Any]) -> Callable[[Image.Image], Any]:
    image_size = int(preprocessing["image_size"])
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(preprocessing["std"], dtype=np.float32)[:, None, None]

    def transform(image: Image.Image) -> np.ndarray:
        width, height = image.size
        if width <= height:
            resized_width = image_size
            resized_height = int(image_size * height / width)
        else:
            resized_height = image_size
            resized_width = int(image_size * width / height)
        resized = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
        left = round((resized_width - image_size) / 2.0)
        top = round((resized_height - image_size) / 2.0)
        cropped = resized.crop((left, top, left + image_size, top + image_size))
        image_array = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1)
        image_array = image_array / np.float32(255.0)
        return np.ascontiguousarray((image_array - mean) / std)

    return transform


def load_predictor(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING_PATH,
    taxon_names_path: str | Path = DEFAULT_TAXON_NAMES_PATH,
    device: Any = None,
    compile_model: bool = False,
    backend: InferenceBackendName = "pytorch",
    onnx_model_path: str | Path = DEFAULT_ONNX_MODEL_PATH,
    onnx_metadata_path: str | Path = DEFAULT_ONNX_METADATA_PATH,
    onnx_int8_model_path: str | Path = DEFAULT_ONNX_INT8_MODEL_PATH,
    onnx_int8_metadata_path: str | Path = DEFAULT_ONNX_INT8_METADATA_PATH,
    tensorrt_engine_path: str | Path = DEFAULT_TENSORRT_ENGINE_PATH,
    tensorrt_metadata_path: str | Path = DEFAULT_TENSORRT_METADATA_PATH,
    preprocessing_path: str | Path = DEFAULT_PREPROCESSING_PATH,
    cpu_threads: int = 1,
) -> Predictor:
    checkpoint_path = Path(checkpoint_path)
    class_mapping_path = Path(class_mapping_path)
    taxon_names_path = Path(taxon_names_path)
    if backend not in (
        "pytorch",
        "pytorch-fp16",
        "onnxruntime",
        "onnxruntime-cuda",
        "onnxruntime-int8",
        "tensorrt",
    ):
        raise ValueError(f"Unknown inference backend: {backend}")
    if backend in ("pytorch-fp16", "tensorrt"):
        torch_module = _require_torch()
        if compile_model:
            raise ValueError("torch.compile cannot be combined with this backend")
        selected_device = torch_module.device(device or "cuda")
        if selected_device.type != "cuda":
            raise ValueError(f"{backend} inference requires CUDA")
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    elif backend in ("onnxruntime", "onnxruntime-int8"):
        if compile_model:
            raise ValueError("torch.compile is not available for ONNX Runtime")
        if torch is None:
            if device is not None and str(device) != "cpu":
                raise ValueError("ONNX Runtime supports CPU inference only")
            selected_device = "cpu"
        else:
            if device is not None and torch.device(device).type != "cpu":
                raise ValueError("ONNX Runtime supports CPU inference only")
            selected_device = torch.device("cpu")
    elif backend == "onnxruntime-cuda":
        torch_module = _require_torch()
        if compile_model:
            raise ValueError("torch.compile is not available for ONNX Runtime")
        selected_device = torch_module.device(device or "cuda")
        if selected_device.type != "cuda":
            raise ValueError("ONNX Runtime CUDA requires a CUDA device")
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_index = selected_device.index
        if device_index is None:
            device_index = torch_module.cuda.current_device()
        selected_device = torch_module.device("cuda", device_index)
    else:
        torch_module = _require_torch()
        selected_device = torch_module.device(
            device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        )

    mapping_data = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    class_to_taxon = {
        int(index): int(taxon_id)
        for index, taxon_id in mapping_data["class_to_taxon"].items()
    }
    if int(mapping_data["class_count"]) != len(class_to_taxon):
        raise ValueError("Class mapping count does not match its entries")
    if set(class_to_taxon) != set(range(len(class_to_taxon))):
        raise ValueError("Class mapping indices are not contiguous")

    raw_taxon_names = json.loads(taxon_names_path.read_text(encoding="utf-8"))
    if not isinstance(raw_taxon_names, dict):
        raise TypeError("Taxon names must be a JSON object")
    taxon_to_names: dict[int, TaxonName] = {}
    for taxon_id, names in raw_taxon_names.items():
        if not isinstance(names, dict):
            raise TypeError(f"Taxon {taxon_id} has invalid name metadata")
        common_name = names.get("common_name")
        scientific_name = names.get("scientific_name")
        if not isinstance(common_name, str) or not common_name.strip():
            raise ValueError(f"Taxon {taxon_id} has no common name")
        if not isinstance(scientific_name, str) or not scientific_name.strip():
            raise ValueError(f"Taxon {taxon_id} has no scientific name")
        taxon_to_names[int(taxon_id)] = {
            "common_name": common_name.strip(),
            "scientific_name": scientific_name.strip(),
        }
    if set(taxon_to_names) != set(class_to_taxon.values()):
        raise ValueError("Taxon names do not match the class mapping")

    if backend == "tensorrt":
        from birdly.inference.tensorrt_backend import load_tensorrt_model

        metadata = json.loads(Path(tensorrt_metadata_path).read_text(encoding="utf-8"))
        preprocessing = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
        if (
            metadata["source_checkpoint_sha256"]
            != preprocessing["source_checkpoint_sha256"]
        ):
            raise ValueError("TensorRT source checkpoint does not match preprocessing")
        if metadata["class_mapping_sha256"] != _file_sha256(class_mapping_path):
            raise ValueError("TensorRT class mapping metadata does not match")
        if metadata["preprocessing_sha256"] != _file_sha256(Path(preprocessing_path)):
            raise ValueError("TensorRT preprocessing metadata does not match")
        if int(metadata["class_count"]) != len(class_to_taxon):
            raise ValueError("TensorRT class count does not match the class mapping")
        model = load_tensorrt_model(
            tensorrt_engine_path,
            device=selected_device,
        )
        model_name = str(metadata["model_name"])
        backend_label = "tensorrt-fp16"
        precision = "float16"
        compile_mode = None
        autocast_dtype = None
    elif backend in ("onnxruntime", "onnxruntime-cuda", "onnxruntime-int8"):
        selected_model_path = Path(
            onnx_int8_model_path if backend == "onnxruntime-int8" else onnx_model_path
        )
        selected_metadata_path = Path(
            onnx_int8_metadata_path
            if backend == "onnxruntime-int8"
            else onnx_metadata_path
        )
        metadata = json.loads(selected_metadata_path.read_text(encoding="utf-8"))
        preprocessing = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
        if (
            metadata["source_checkpoint_sha256"]
            != preprocessing["source_checkpoint_sha256"]
        ):
            raise ValueError("ONNX source checkpoint does not match preprocessing")
        if metadata["class_mapping_sha256"] != _file_sha256(class_mapping_path):
            raise ValueError("ONNX class mapping metadata does not match")
        if metadata["preprocessing_sha256"] != _file_sha256(Path(preprocessing_path)):
            raise ValueError("ONNX preprocessing metadata does not match")
        if int(metadata["class_count"]) != len(class_to_taxon):
            raise ValueError("ONNX class count does not match the class mapping")
        model = load_onnx_runtime_model(
            selected_model_path,
            cpu_threads=cpu_threads,
            device=selected_device,
        )
        model_name = str(metadata["model_name"])
        if backend == "onnxruntime-cuda":
            backend_label = "onnxruntime-fp32-cuda"
        elif backend == "onnxruntime-int8":
            backend_label = "onnxruntime-int8"
        else:
            backend_label = "onnxruntime-fp32"
        precision = "int8" if backend == "onnxruntime-int8" else "float32"
        compile_mode = None
        autocast_dtype = None
    else:
        torch_module = _require_torch()
        from birdly.training.model import build_model

        checkpoint = torch_module.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        checkpoint_mapping = {
            int(index): int(taxon_id)
            for index, taxon_id in checkpoint["class_to_taxon"].items()
        }
        if checkpoint_mapping != class_to_taxon:
            raise ValueError("Class mapping does not match the checkpoint")
        model = build_model(
            len(class_to_taxon),
            model_name=str(checkpoint["config"]["model"]["name"]),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(selected_device)
        model.eval()
        model_name = type(model).__name__
        if backend == "pytorch-fp16":
            compile_mode = None
            autocast_dtype = torch_module.float16
            backend_label = "pytorch-fp16"
            precision = "float16"
        else:
            compile_mode = DEFAULT_COMPILE_MODE if compile_model else None
            if compile_mode is not None:
                model = torch_module.compile(model, mode=compile_mode)
            backend_label = (
                "pytorch-compile-inductor" if compile_model else "pytorch-eager"
            )
            precision = str(next(model.parameters()).dtype).removeprefix("torch.")
            autocast_dtype = None
        preprocessing = checkpoint["preprocessing"]

    image_size = int(preprocessing["image_size"])
    if transforms is None:
        if backend not in ("onnxruntime", "onnxruntime-int8"):
            raise RuntimeError("This inference backend requires TorchVision")
        transform = _numpy_transform(preprocessing)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=preprocessing["mean"],
                    std=preprocessing["std"],
                ),
            ]
        )
    return Predictor(
        model=model,
        transform=transform,
        class_to_taxon=class_to_taxon,
        taxon_to_names=taxon_to_names,
        device=selected_device,
        model_name=model_name,
        backend=backend_label,
        precision=precision,
        compile_mode=compile_mode,
        autocast_dtype=autocast_dtype,
        image_size=image_size,
        preprocessing=preprocessing,
    )


def preprocess_image(
    predictor: Predictor,
    image_path: str | Path | Image.Image,
) -> Any:
    if isinstance(image_path, Image.Image):
        return predictor.transform(image_path.convert("RGB"))
    with Image.open(image_path) as image:
        return predictor.transform(image.convert("RGB"))


def inference_context(predictor: Predictor):
    if torch is None or predictor.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(
        device_type=predictor.device.type,
        dtype=predictor.autocast_dtype,
    )


def predict_logits(
    predictor: Predictor,
    image_paths: Sequence[str | Path | Image.Image],
) -> Any:
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    transformed_images = [
        preprocess_image(predictor, image_path) for image_path in image_paths
    ]
    if torch is not None and isinstance(transformed_images[0], torch.Tensor):
        image_batch = torch.stack(transformed_images).to(predictor.device)
        with torch.inference_mode(), inference_context(predictor):
            return predictor.model(image_batch).float()
    image_batch = np.stack(transformed_images).astype(np.float32, copy=False)
    return np.asarray(predictor.model(image_batch), dtype=np.float32)


def filter_predictions_by_confidence(
    predictions: Sequence[Prediction],
) -> list[Prediction]:
    if not predictions or predictions[0]["score"] <= MINIMUM_TOP_CONFIDENCE:
        return []

    selected: list[Prediction] = []
    cumulative_confidence = 0.0
    for prediction in predictions[:MAXIMUM_RETURNED_PREDICTIONS]:
        selected.append(prediction)
        cumulative_confidence += prediction["score"]
        if cumulative_confidence >= CUMULATIVE_CONFIDENCE_TARGET or math.isclose(
            cumulative_confidence,
            CUMULATIVE_CONFIDENCE_TARGET,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            break
    return selected


def _format_predictions(
    predictor: Predictor,
    class_indices: Sequence[int],
    scores: Sequence[float],
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for index, score in zip(class_indices, scores):
        taxon_id = predictor.class_to_taxon[int(index)]
        names = predictor.taxon_to_names[taxon_id]
        predictions.append(
            {
                "taxon_id": taxon_id,
                "common_name": names["common_name"],
                "scientific_name": names["scientific_name"],
                "score": float(score),
            }
        )
    return filter_predictions_by_confidence(predictions)


def predict_batch(
    predictor: Predictor,
    image_paths: Sequence[str | Path | Image.Image],
    top_k: int = 5,
) -> list[list[Prediction]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    inference_mode = torch.inference_mode() if torch is not None else nullcontext()
    with inference_mode:
        logits = predict_logits(predictor, image_paths)
        if torch is not None and isinstance(logits, torch.Tensor):
            probabilities = torch.softmax(logits, dim=1)
            scores, indices = probabilities.topk(
                min(top_k, len(predictor.class_to_taxon)), dim=1
            )
            score_rows = scores.tolist()
            index_rows = indices.tolist()
        else:
            logits = np.asarray(logits, dtype=np.float32)
            shifted_logits = logits - logits.max(axis=1, keepdims=True)
            exponentials = np.exp(shifted_logits)
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
            selected_count = min(top_k, len(predictor.class_to_taxon))
            indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :selected_count]
            scores = np.take_along_axis(probabilities, indices, axis=1)
            score_rows = scores.tolist()
            index_rows = indices.tolist()

    batch_predictions: list[list[Prediction]] = []
    for row_indices, row_scores in zip(index_rows, score_rows):
        batch_predictions.append(
            _format_predictions(predictor, row_indices, row_scores)
        )
    return batch_predictions


def predict_image(
    predictor: Predictor,
    image_path: str | Path | Image.Image,
    top_k: int = 5,
) -> list[Prediction]:
    return predict_batch(predictor, [image_path], top_k)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one bird image.")
    parser.add_argument("image", type=Path)
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
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    args = parser.parse_args()

    predictor = load_predictor(
        compile_model=args.compile_model,
        backend=args.backend,
        cpu_threads=args.cpu_threads,
    )
    for rank, prediction in enumerate(predict_image(predictor, args.image), start=1):
        print(
            f"{rank}. "
            f"taxon_id={prediction['taxon_id']} "
            f"common_name={prediction['common_name']} "
            f"scientific_name={prediction['scientific_name']} "
            f"score={prediction['score']:.6f}"
        )


if __name__ == "__main__":
    main()
