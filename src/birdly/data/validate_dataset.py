from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from PIL import Image, UnidentifiedImageError

from birdly.data.utils import load_config

MAX_VALIDATION_WORKERS = 8
VALIDATION_BATCH_SIZE = 256
REQUIRED_COLUMNS = {
    "photo_uuid",
    "photo_id",
    "observation_uuid",
    "taxon_id",
    "name",
    "extension",
    "license",
    "position",
    "split",
    "origin_scope",
}


def _normalize_image_path(row: dict[str, Any], image_root: Path) -> tuple[Path, str]:
    if row.get("image_path") is not None:
        candidate = Path(str(row["image_path"]))
        image_path = candidate if candidate.is_absolute() else image_root / candidate
    else:
        extension = str(row.get("extension", "")).strip()
        photo_id = row.get("photo_id")
        if photo_id is None:
            raise ValueError(f"Manifest row missing photo_id: {row}")
        image_path = image_root / f"{photo_id}.{extension}"

    resolved_root = image_root.resolve()
    resolved_path = image_path.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(f"Image path is outside image_data_root: {image_path}") from error

    return resolved_path, relative_path


def _validate_image_entry(
    row: dict[str, Any], image_root: Path
) -> tuple[bool, dict[str, Any], str | None]:
    try:
        image_path, relative_path = _normalize_image_path(row, image_root)
        if not image_path.exists():
            return False, row, "missing"
        if image_path.stat().st_size <= 0:
            return False, row, "empty"
        with Image.open(image_path) as image:
            image.convert("RGB").load()
    except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError) as error:
        return False, row, type(error).__name__

    validated_row = {**row, "image_path": relative_path}
    return True, validated_row, None


def _validate_split_integrity(
    manifest: pl.DataFrame,
    n_val: int,
    n_test: int,
    *,
    reject_duplicate_photos: bool = True,
) -> dict[str, Any]:
    missing_columns = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    split_values = set(manifest["split"].unique().to_list())
    if split_values != {"train", "val", "test"}:
        raise ValueError(f"Manifest has invalid split values: {sorted(split_values)}")

    leakage = (
        manifest.group_by("observation_uuid")
        .agg([
            pl.len().alias("n_rows"),
            pl.col("split").n_unique().alias("n_splits"),
        ])
        .filter(pl.col("n_splits") > 1)
    )
    if leakage.height > 0:
        observation_ids = leakage["observation_uuid"].to_list()
        raise ValueError(
            "Split leakage detected: observation_uuid appears in multiple splits: "
            + ", ".join(str(value) for value in observation_ids[:10])
        )

    duplicate_counts = {}
    for column in ("photo_uuid", "photo_id"):
        count = manifest.group_by(column).len().filter(pl.col("len") > 1).height
        duplicate_counts[column] = count
        if count and reject_duplicate_photos:
            raise ValueError(f"Duplicate leakage detected for {column}: {count} groups")

    bad_eval_rows = manifest.filter(
        (pl.col("split").is_in(["val", "test"]))
        & (pl.col("position") != 0)
    )
    if bad_eval_rows.height > 0:
        raise ValueError(
            "Validation/test rows must use position == 0. "
            f"Found {bad_eval_rows.height} invalid rows."
        )

    split_class_counts = manifest.group_by(["taxon_id", "split"]).len()
    class_count = manifest["taxon_id"].n_unique()
    for split in ("train", "val", "test"):
        present_classes = split_class_counts.filter(pl.col("split") == split)[
            "taxon_id"
        ].n_unique()
        if present_classes != class_count:
            raise ValueError(
                f"Split '{split}' has {present_classes} of {class_count} classes."
            )

    per_class_split_counts = {}
    for split, expected_count in (("train", None), ("val", n_val), ("test", n_test)):
        counts = split_class_counts.filter(pl.col("split") == split)["len"]
        split_report = {
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
        }
        if expected_count is not None:
            split_report["expected"] = expected_count
            split_report["classes_below_expected"] = int(
                (counts < expected_count).sum()
            )
        per_class_split_counts[split] = split_report

    return {
        "class_count": class_count,
        "duplicate_groups": duplicate_counts,
        "observation_cross_split_groups": leakage.height,
        "per_class_split_counts": per_class_split_counts,
    }


def _remove_ambiguous_photos(manifest: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    ambiguous_rows = manifest.filter(
        pl.col("photo_uuid").is_duplicated() | pl.col("photo_id").is_duplicated()
    )
    if ambiguous_rows.height == 0:
        return manifest, 0

    cleaned_manifest = manifest.join(
        ambiguous_rows.select("photo_uuid").unique(),
        on="photo_uuid",
        how="anti",
    ).join(
        ambiguous_rows.select("photo_id").unique(),
        on="photo_id",
        how="anti",
    )
    return cleaned_manifest, ambiguous_rows.height


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def validate_dataset() -> None:
    config, local_config, params = load_config()
    with open("configs/validation.yaml", "r", encoding="utf-8") as handle:
        validation_config = yaml.safe_load(handle)

    manifest_path = Path(config["processed"]["manifest"])
    validated_manifest_path = Path(validation_config["validated_manifest"])
    validation_report_path = Path(validation_config["validation_report"])
    image_root = Path(local_config["image_data_root"])

    source_manifest = pl.read_parquet(manifest_path)
    source_integrity = _validate_split_integrity(
        source_manifest,
        n_val=int(params["split"]["n_val"]),
        n_test=int(params["split"]["n_test"]),
        reject_duplicate_photos=False,
    )
    manifest, removed_ambiguous_rows = _remove_ambiguous_photos(source_manifest)
    integrity = _validate_split_integrity(
        manifest,
        n_val=int(params["split"]["n_val"]),
        n_test=int(params["split"]["n_test"]),
    )
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    rows = list(manifest.iter_rows(named=True))
    with ThreadPoolExecutor(max_workers=MAX_VALIDATION_WORKERS) as executor:
        for start in range(0, len(rows), VALIDATION_BATCH_SIZE):
            batch = rows[start:start + VALIDATION_BATCH_SIZE]
            futures = {
                executor.submit(_validate_image_entry, row, image_root): row
                for row in batch
            }
            for future in as_completed(futures):
                is_valid, validated_row, error_type = future.result()
                if is_valid:
                    valid_rows.append(validated_row)
                else:
                    invalid_rows.append(
                        {
                            "photo_id": int(validated_row["photo_id"]),
                            "split": str(validated_row["split"]),
                            "error": error_type,
                        }
                    )

    invalid_eval_rows = [
        row for row in invalid_rows if row["split"] in {"val", "test"}
    ]
    if invalid_eval_rows:
        raise ValueError(
            "Validation and test images must be available and readable. "
            f"Found {len(invalid_eval_rows)} invalid evaluation images."
        )

    validated_manifest = pl.DataFrame(valid_rows, infer_schema_length=None)
    columns = [*manifest.columns, "image_path"]
    if "image_path" in manifest.columns:
        columns = manifest.columns
    validated_manifest = validated_manifest.select(columns)
    validated_manifest = validated_manifest.sort(["taxon_id", "split", "photo_id"])
    _validate_split_integrity(
        validated_manifest,
        n_val=int(params["split"]["n_val"]),
        n_test=int(params["split"]["n_test"]),
    )

    validated_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest_path = validated_manifest_path.with_suffix(".parquet.tmp")
    validated_manifest.write_parquet(
        temporary_manifest_path,
        compression="lz4",
    )
    temporary_manifest_path.replace(validated_manifest_path)

    species_count = validated_manifest["taxon_id"].n_unique()
    train_count = validated_manifest.filter(pl.col("split") == "train").height
    val_count = validated_manifest.filter(pl.col("split") == "val").height
    test_count = validated_manifest.filter(pl.col("split") == "test").height

    per_species_counts = validated_manifest.group_by("taxon_id").agg(pl.len().alias("count"))
    min_count = per_species_counts["count"].min()
    max_count = per_species_counts["count"].max()

    train_state_count = validated_manifest.filter(
        (pl.col("split") == "train") & (pl.col("origin_scope") == "state")
    ).height
    train_global_count = validated_manifest.filter(
        (pl.col("split") == "train") & (pl.col("origin_scope") == "global")
    ).height

    print(f"Species: {species_count}")
    print(f"Train: {train_count} | Val: {val_count} | Test: {test_count}")
    print(f"Min count per species: {min_count}")
    print(f"Max count per species: {max_count}")
    print(f"State training rows: {train_state_count}")
    print(f"Global training rows: {train_global_count}")
    print(f"Invalid or missing training images: {len(invalid_rows)}")

    if train_count == 0 or val_count == 0 or test_count == 0:
        raise ValueError("Training, validation, and test splits must all be non-empty.")

    class_ids = sorted(int(value) for value in validated_manifest["taxon_id"].unique())
    report = {
        "source_manifest": {
            "path": manifest_path.as_posix(),
            "sha256": _sha256(manifest_path),
            "rows": source_manifest.height,
        },
        "validated_manifest": {
            "path": validated_manifest_path.as_posix(),
            "sha256": _sha256(validated_manifest_path),
            "rows": validated_manifest.height,
        },
        "split_counts": {
            split: validated_manifest.filter(pl.col("split") == split).height
            for split in ("train", "val", "test")
        },
        "class_count": species_count,
        "taxon_to_class": {str(taxon_id): index for index, taxon_id in enumerate(class_ids)},
        "image_data": {
            "dvc_version": str(validation_config["image_data_version"]),
            "dvc_files": int(validation_config["image_data_files"]),
            "dvc_bytes": int(validation_config["image_data_bytes"]),
            "validated_references": validated_manifest.height,
        },
        "invalid_training_images": invalid_rows,
        "integrity": integrity,
        "source_integrity": source_integrity,
        "removed_ambiguous_photo_rows": removed_ambiguous_rows,
        "position_zero_evaluation": True,
    }
    _atomic_write_json(validation_report_path, report)

    print(f"Validated manifest saved to {validated_manifest_path}")
    print(f"Validation report saved to {validation_report_path}")


if __name__ == "__main__":
    validate_dataset()
