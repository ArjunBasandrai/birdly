"""MLflow inference wrapper for a frozen Birdly checkpoint."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from birdly.training.model import build_model


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BirdlyRegistryModel(mlflow.pyfunc.PythonModel):
    """Predict a class and taxon ID from a base64-encoded image."""

    def __init__(self, checkpoint_sha256: str) -> None:
        self.checkpoint_sha256 = checkpoint_sha256
        self.model: torch.nn.Module | None = None
        self.transform: Any = None
        self.class_to_taxon: dict[int, int] = {}

    def load_context(self, context: Any) -> None:
        checkpoint_path = Path(context.artifacts["checkpoint"])
        if file_sha256(checkpoint_path) != self.checkpoint_sha256:
            raise ValueError("Registered checkpoint SHA-256 does not match")
        # The checkpoint is a trusted local training artifact. It contains
        # optimizer and RNG state as well as the model weights.
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        self.class_to_taxon = {
            int(key): int(value) for key, value in checkpoint["class_to_taxon"].items()
        }
        if set(self.class_to_taxon) != set(range(len(self.class_to_taxon))):
            raise ValueError("Checkpoint class indices are not contiguous")
        config = checkpoint["config"]
        self.model = build_model(
            len(self.class_to_taxon),
            model_name=str(config["model"]["name"]),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        preprocessing = checkpoint["preprocessing"]
        image_size = int(preprocessing["image_size"])
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=preprocessing["mean"], std=preprocessing["std"]
                ),
            ]
        )

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: Any = None
    ) -> pd.DataFrame:
        del context, params
        if self.model is None or self.transform is None:
            raise RuntimeError("Registered model has not loaded")
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if "image_base64" not in model_input.columns:
            raise ValueError("Input must have an image_base64 column")

        results: list[tuple[int, int, float]] = []
        image_values = model_input["image_base64"].tolist()
        for offset in range(0, len(image_values), 8):
            tensors = []
            for value in image_values[offset : offset + 8]:
                if not isinstance(value, str):
                    raise TypeError("image_base64 values must be strings")
                image_bytes = base64.b64decode(value, validate=True)
                with Image.open(BytesIO(image_bytes)) as image:
                    tensors.append(self.transform(image.convert("RGB")))
            with torch.inference_mode():
                logits = self.model(torch.stack(tensors))
                scores, indices = torch.softmax(logits, dim=1).max(dim=1)
            for index, score in zip(indices.tolist(), scores.tolist()):
                results.append(
                    (int(index), self.class_to_taxon[int(index)], float(score))
                )
        return pd.DataFrame(
            results,
            columns=["class_index", "taxon_id", "confidence"],
        ).astype({"class_index": "int64", "taxon_id": "int64", "confidence": "float64"})
