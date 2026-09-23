from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BirdlyDataset(Dataset):
    def __init__(
        self,
        split: str,
        manifest_path: str | Path,
        image_root: str | Path,
        image_size: int = 384,
        taxon_to_class: dict[int, int] | None = None,
        augmentation: dict[str, Any] | None = None,
        max_samples_per_class: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train, val, or test.")

        self.split = split
        self.image_root = Path(image_root)
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.default_dtype = torch.float32

        manifest = pl.read_parquet(self.manifest_path)
        if "image_path" not in manifest.columns:
            manifest = manifest.with_columns(
                pl.concat_str(
                    [
                        pl.col("photo_id").cast(pl.String),
                        pl.lit("."),
                        pl.col("extension").cast(pl.String),
                    ]
                ).alias("image_path")
            )

        self.manifest = manifest.filter(pl.col("split") == split).sort(
            ["taxon_id", "photo_id"]
        )
        if max_samples_per_class is not None:
            self.manifest = self.manifest.group_by(
                "taxon_id", maintain_order=True
            ).head(max_samples_per_class)
        if self.manifest.height == 0:
            raise ValueError(f"No rows found for split '{split}' in {self.manifest_path}.")

        if taxon_to_class is None:
            taxon_ids = sorted(manifest["taxon_id"].unique().to_list())
            self.taxon_to_class = {taxon_id: index for index, taxon_id in enumerate(taxon_ids)}
        else:
            self.taxon_to_class = taxon_to_class

        self.records = self.manifest.to_dicts()
        self.augmentation = augmentation or {}
        self.transform = self._build_transform(split)

    def _build_transform(self, split: str):
        if split == "train":
            operations: list[Any] = [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=tuple(self.augmentation.get("crop_scale", [0.8, 1.0])),
                ),
                transforms.RandomHorizontalFlip(
                    p=float(self.augmentation.get("horizontal_flip", 0.5))
                ),
            ]
            color_jitter = float(self.augmentation.get("color_jitter", 0.0))
            if color_jitter > 0:
                operations.append(
                    transforms.ColorJitter(
                        brightness=color_jitter,
                        contrast=color_jitter,
                        saturation=color_jitter,
                        hue=min(color_jitter / 2.0, 0.5),
                    )
                )
            randaugment_magnitude = int(
                self.augmentation.get("randaugment_magnitude", 0)
            )
            if randaugment_magnitude > 0:
                operations.append(
                    transforms.RandAugment(
                        num_ops=int(self.augmentation.get("randaugment_ops", 2)),
                        magnitude=randaugment_magnitude,
                    )
                )
            operations.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    transforms.RandomErasing(
                        p=float(self.augmentation.get("random_erasing", 0.0))
                    ),
                ]
            )
            return transforms.Compose(operations)

        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image_path = self.image_root / str(row["image_path"])
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        label = self.taxon_to_class[int(row["taxon_id"])]
        return image_tensor.to(dtype=self.default_dtype), int(label)
