from pathlib import Path

import polars as pl

from birdly.data.utils import load_config


MANIFEST_COLUMNS = [
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
]


def prepare_state_data(
    filtered_photos_data_path: Path,
    min_state_count: int,
    n_test: int,
    n_val: int,
):
    filtered_photos = (
        pl.scan_parquet(filtered_photos_data_path)
        .select([
            "photo_uuid",
            "photo_id",
            "observation_uuid",
            "taxon_id",
            "name",
            "extension",
            "license",
            "position",
        ])
    )

    observation_stats = (
        filtered_photos
        .group_by([
            "taxon_id",
            "observation_uuid",
        ])
        .agg(
            pl.len().alias("n_photos")
        )
        .sort([
            "taxon_id",
            "n_photos",
            "observation_uuid",
        ])
        .with_columns(
            pl.col("taxon_id")
            .cum_count()
            .over("taxon_id")
            .alias("observation_rank")
        )
    )

    supported_species = (
        observation_stats
        .group_by("taxon_id")
        .agg(
            pl.len().alias("state_observation_count")
        )
        .filter(
            pl.col("state_observation_count") >= min_state_count
        )
        .select("taxon_id")
    )

    ranked_observations = (
        observation_stats
        .join(
            supported_species,
            on="taxon_id",
            how="semi",
        )
        .with_columns(
            pl.when(
                pl.col("observation_rank") <= n_test
            )
            .then(pl.lit("test"))
            .when(
                pl.col("observation_rank") <= n_test + n_val
            )
            .then(pl.lit("val"))
            .otherwise(pl.lit("train"))
            .alias("split")
        )
    )

    eval_photos = (
        filtered_photos
        .filter(pl.col("position") == 0)
        .join(
            ranked_observations
            .filter(
                pl.col("split").is_in(["test", "val"])
            )
            .select([
                "observation_uuid",
                "split",
            ]),
            on="observation_uuid",
            how="inner",
        )
        .with_columns(
            pl.lit("state").alias("origin_scope")
        )
        .select(MANIFEST_COLUMNS)
    )

    state_train = (
        filtered_photos
        .join(
            ranked_observations
            .filter(pl.col("split") == "train")
            .select([
                "observation_uuid",
                "split",
            ]),
            on="observation_uuid",
            how="inner",
        )
        .with_columns(
            pl.lit("state").alias("origin_scope")
        )
        .select(MANIFEST_COLUMNS)
    )

    species_names = (
        filtered_photos
        .select([
            "taxon_id",
            "name",
        ])
        .unique()
    )

    supported_species = (
        supported_species
        .join(
            species_names,
            on="taxon_id",
            how="left",
        )
    )

    return supported_species, state_train, eval_photos


def add_global_photos(
    state_train,
    supported_species,
    observations_data_path: Path,
    photos_data_path: Path,
    allowed_licenses: list[str],
    bbox: dict,
    train_cap: int,
):
    state_primary_counts = (
        state_train
        .filter(pl.col("position") == 0)
        .group_by("taxon_id")
        .agg(
            pl.len().alias("state_primary_count")
        )
    )

    species_needing_global = (
        supported_species
        .join(
            state_primary_counts,
            on="taxon_id",
            how="left",
        )
        .with_columns(
            pl.col("state_primary_count").fill_null(0)
        )
        .filter(
            pl.col("state_primary_count") < train_cap
        )
        .select([
            "taxon_id",
            "name",
        ])
        .collect(engine="streaming")
    )

    print(
        f"Species needing global supplementation: "
        f"{species_needing_global.height:,}"
    )

    state_train = (
        state_train
        .sort([
            "taxon_id",
            "position",
            "observation_uuid",
            "photo_uuid",
        ])
        .group_by(
            "taxon_id",
            maintain_order=True,
        )
        .head(train_cap)
        .select(MANIFEST_COLUMNS)
    )

    if species_needing_global.height == 0:
        return state_train

    global_observations = (
        pl.scan_csv(
            observations_data_path,
            separator="\t",
            quote_char=None,
        )
        .filter(
            (pl.col("quality_grade") == "research")
            & pl.col("anomaly_score").is_not_null()
            & (pl.col("anomaly_score") >= 1)
            & ~(
                pl.col("latitude").is_between(
                    bbox["min_lat"],
                    bbox["max_lat"],
                )
                & pl.col("longitude").is_between(
                    bbox["min_lon"],
                    bbox["max_lon"],
                )
            )
        )
        .join(
            species_needing_global
            .select("taxon_id")
            .lazy(),
            on="taxon_id",
            how="semi",
        )
        .select([
            "observation_uuid",
            "taxon_id",
        ])
    )

    global_photos = (
        pl.scan_csv(
            photos_data_path,
            separator="\t",
            quote_char=None,
        )
        .filter(
            pl.col("license").is_in(allowed_licenses)
        )
        .join(
            global_observations,
            on="observation_uuid",
            how="inner",
        )
        .join(
            species_needing_global
            .select([
                "taxon_id",
                "name",
            ])
            .lazy(),
            on="taxon_id",
            how="inner",
        )
        .sort([
            "taxon_id",
            "position",
            "observation_uuid",
            "photo_uuid",
        ])
        .group_by(
            "taxon_id",
            maintain_order=True,
        )
        .head(train_cap)
        .with_columns([
            pl.lit("train").alias("split"),
            pl.lit("global").alias("origin_scope"),
        ])
        .select(MANIFEST_COLUMNS)
    )

    return (
        pl.concat([
            state_train,
            global_photos,
        ])
        .with_columns(
            pl.when(
                pl.col("origin_scope") == "state"
            )
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("_origin_order")
        )
        .sort([
            "taxon_id",
            "position",
            "_origin_order",
            "observation_uuid",
            "photo_uuid",
        ])
        .group_by(
            "taxon_id",
            maintain_order=True,
        )
        .head(train_cap)
        .drop("_origin_order")
        .select(MANIFEST_COLUMNS)
    )


def build_manifest() -> None:
    config, local_config, params = load_config()

    data_root = Path(local_config["raw_data_root"])

    observations_data_path = (
        data_root / config["raw"]["observations"]
    )

    photos_data_path = (
        data_root / config["raw"]["photos"]
    )

    filtered_photos_data_path = Path(
        config["processed"]["filtered_photos"]
    )

    manifest_data_path = Path(
        config["processed"]["manifest"]
    )

    manifest_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    min_state_count = params["min_counts"]["min_state_count"]
    min_total_count = params["min_counts"]["min_total_count"]
    max_total_count = params["min_counts"]["max_total_count"]

    n_test = params["split"]["n_test"]
    n_val = params["split"]["n_val"]

    train_cap = max_total_count - n_test - n_val

    supported_species, state_train, eval_photos = prepare_state_data(
        filtered_photos_data_path,
        min_state_count,
        n_test,
        n_val,
    )

    selected_train = add_global_photos(
        state_train,
        supported_species,
        observations_data_path,
        photos_data_path,
        config["allowed_licenses"],
        config["geography"]["bounding_box"],
        train_cap,
    )

    manifest = pl.concat([
        selected_train,
        eval_photos,
    ])

    valid_species = (
        manifest
        .group_by("taxon_id")
        .agg(
            pl.len().alias("total_count")
        )
        .filter(
            pl.col("total_count") >= min_total_count
        )
        .select("taxon_id")
    )

    manifest = (
        manifest
        .join(
            valid_species,
            on="taxon_id",
            how="semi",
        )
        .select(MANIFEST_COLUMNS)
    )

    manifest.sink_parquet(
        manifest_data_path,
        compression="lz4",
        maintain_order=False,
        engine="streaming",
    )

    print(f"Manifest saved to {manifest_data_path}")


if __name__ == "__main__":
    build_manifest()