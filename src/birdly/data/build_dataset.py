from pathlib import Path

import polars as pl
import yaml


def load_config():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(".birdly/local.yaml", "r") as f:
        local_config = yaml.safe_load(f)

    return config, local_config


def build_dataset() -> None:
    config, local_config = load_config()

    data_root = Path(local_config["raw_data_root"])

    observations_data_path = data_root / config["raw"]["observations"]
    photos_data_path = data_root / config["raw"]["photos"]

    filtered_photos_data_path = Path(
        config["processed"]["filtered_photos"]
    )

    dataset_manifest_path = Path(
        config["processed"]["dataset_manifest"]
    )

    dataset_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    min_state_count = config["min_counts"]["min_state_count"]
    min_total_count = config["min_counts"]["min_total_count"]
    max_total_count = config["min_counts"]["max_total_count"]

    allowed_licenses = config["allowed_licenses"]

    bbox = config["geography"]["bounding_box"]

    min_lat = bbox["min_lat"]
    max_lat = bbox["max_lat"]
    min_lon = bbox["min_lon"]
    max_lon = bbox["max_lon"]

    print("Finding supported species...")

    supported_species = (
        pl.scan_parquet(filtered_photos_data_path)
        .filter(pl.col("position") == 0)
        .group_by(["taxon_id", "name"])
        .len()
        .filter(pl.col("len") >= min_state_count)
        .select(["taxon_id", "name"])
        .collect()
    )

    print(f"Supported state species: {supported_species.height:,}")
    print("Scanning global observations...")

    observations = (
        pl.scan_csv(
            observations_data_path,
            separator="\t",
            quote_char=None,
        )
        .filter(
            (pl.col("quality_grade") == "research")
            & pl.col("anomaly_score").is_not_null()
            & (pl.col("anomaly_score") >= 1)
        )
        .join(
            supported_species.select("taxon_id").lazy(),
            on="taxon_id",
            how="semi",
        )
        .with_columns(
            pl.when(
                pl.col("latitude").is_between(min_lat, max_lat)
                & pl.col("longitude").is_between(min_lon, max_lon)
            )
            .then(pl.lit("state"))
            .otherwise(pl.lit("global"))
            .alias("origin_scope")
        )
        .select([
            "observation_uuid",
            "taxon_id",
            "origin_scope",
        ])
    )

    print("Scanning global photos...")

    photos = (
        pl.scan_csv(
            photos_data_path,
            separator="\t",
            quote_char=None,
        )
        .filter(
            pl.col("license").is_in(allowed_licenses)
        )
        .join(
            observations,
            on="observation_uuid",
            how="inner",
        )
        .join(
            supported_species.lazy(),
            on="taxon_id",
            how="left",
        )
        .with_columns(
            (pl.col("position") == 0).alias("is_primary")
        )
        .with_columns(
            pl.len().over("taxon_id").alias("_total_count")
        )
        .filter(
            pl.col("_total_count") >= min_total_count
        )
        .with_columns(
            pl.col("photo_uuid").hash(seed=42).alias("_sample_key")
        )
        .sort(
            [
                "taxon_id",
                "is_primary",
                "origin_scope",
                "_sample_key",
            ],
            descending=[
                False,
                True,
                True,
                False,
            ],
        )
        .group_by("taxon_id")
        .head(max_total_count)
        .drop([
            "_total_count",
            "_sample_key",
        ])
    )

    print(f"Writing: {dataset_manifest_path}")

    photos.sink_parquet(
        dataset_manifest_path,
        compression="lz4",
        maintain_order=False,
        engine="streaming",
    )

    print("Done.")


if __name__ == "__main__":
    build_dataset()