from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from birdly.training.model import build_model

DEFAULT_MODEL_DIRECTORY = Path(
    "B:/birdly-training/birdly-research-v3/runs/trial-01-lr-low-seed-2027"
)
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODEL_DIRECTORY / "best.pt"
DEFAULT_CLASS_MAPPING_PATH = DEFAULT_MODEL_DIRECTORY / "class_mapping.json"
DEFAULT_SPECIES_NAMES_PATH = Path(__file__).with_name("species_names.json")


class Prediction(TypedDict):
    species_name: str
    taxon_id: int
    score: float


@dataclass(frozen=True)
class Predictor:
    model: nn.Module
    transform: transforms.Compose
    class_to_taxon: dict[int, int]
    taxon_to_name: dict[int, str]
    device: torch.device


def load_predictor(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING_PATH,
    species_names_path: str | Path = DEFAULT_SPECIES_NAMES_PATH,
    device: str | torch.device | None = None,
) -> Predictor:
    checkpoint_path = Path(checkpoint_path)
    class_mapping_path = Path(class_mapping_path)
    species_names_path = Path(species_names_path)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
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

    taxon_to_name = {
        int(taxon_id): str(name)
        for taxon_id, name in json.loads(
            species_names_path.read_text(encoding="utf-8")
        ).items()
    }
    if set(taxon_to_name) != set(class_to_taxon.values()):
        raise ValueError("Species names do not match the class mapping")

    checkpoint = torch.load(
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

    preprocessing = checkpoint["preprocessing"]
    image_size = int(preprocessing["image_size"])
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
        model,
        transform,
        class_to_taxon,
        taxon_to_name,
        selected_device,
    )


def predict_image(
    predictor: Predictor,
    image_path: str | Path,
    top_k: int = 5,
) -> list[Prediction]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    with Image.open(image_path) as image:
        image_tensor = predictor.transform(image.convert("RGB"))
    image_tensor = image_tensor.unsqueeze(0).to(predictor.device)

    with torch.inference_mode():
        logits = predictor.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        scores, indices = probabilities.topk(
            min(top_k, len(predictor.class_to_taxon)), dim=1
        )

    predictions: list[Prediction] = []
    for index, score in zip(indices[0].tolist(), scores[0].tolist()):
        class_index = int(index)
        taxon_id = predictor.class_to_taxon[class_index]
        predictions.append(
            {
                "species_name": predictor.taxon_to_name[taxon_id],
                "taxon_id": taxon_id,
                "score": float(score),
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one bird image.")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    predictor = load_predictor()
    for rank, prediction in enumerate(predict_image(predictor, args.image), start=1):
        print(
            f"{rank}. "
            f"species_name={prediction['species_name']} "
            f"taxon_id={prediction['taxon_id']} "
            f"score={prediction['score']:.6f}"
        )


if __name__ == "__main__":
    main()
