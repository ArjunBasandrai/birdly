from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from birdly.data.dataset import BirdlyDataset
from birdly.training.io_utils import atomic_write_json
from birdly.training.model import build_model
from birdly.training.train import _loader, evaluate_loader


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    split: str,
    manifest_path: Path,
    image_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    taxon_to_class = {
        int(key): int(value) for key, value in checkpoint["taxon_to_class"].items()
    }
    class_to_taxon = {value: key for key, value in taxon_to_class.items()}
    dataset = BirdlyDataset(
        split,
        manifest_path,
        image_root,
        image_size=int(checkpoint["preprocessing"]["image_size"]),
        taxon_to_class=taxon_to_class,
    )
    loader = _loader(
        dataset,
        config,
        shuffle=False,
        seed=int(config["training"]["seed"]) + 200_000,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for evaluation.")
    model = build_model(
        len(taxon_to_class),
        model_name=str(config["model"]["name"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    metrics = evaluate_loader(
        model,
        loader,
        device,
        class_to_taxon,
        amp_enabled=bool(config["training"]["amp"]),
    )

    first_image, _ = dataset[0]
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        inference_logits = model(first_image.unsqueeze(0).to(device))
    inference_verified = bool(
        inference_logits.shape == (1, len(taxon_to_class))
        and torch.isfinite(inference_logits).all().item()
    )
    if not inference_verified:
        raise RuntimeError("Winner inference verification failed.")

    result = {
        "run_id": checkpoint["run_id"],
        "git_commit": checkpoint["git_commit"],
        "data_version": checkpoint["data_version"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["best_epoch"],
        "split": split,
        "metrics": metrics,
        "inference_verified": inference_verified,
    }
    atomic_write_json(output_path, result)
    return result
