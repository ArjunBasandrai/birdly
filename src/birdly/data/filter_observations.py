from pathlib import Path

import polars as pl

from birdly.data.utils import load_config


def filter_observations() -> None:
    config, local_config, _ = load_config()

    bbox = config["geography"]["bounding_box"]

    min_lat = bbox["min_lat"]
    max_lat = bbox["max_lat"]
    min_lon = bbox["min_lon"]
    max_lon = bbox["max_lon"]

    data_root = Path(local_config["raw_data_root"])

    observations_data_path = data_root / config["raw"]["observations"]
    taxa_data_path = data_root / config["raw"]["taxa"]

    filtered_observations_data_path = Path(
        config["processed"]["filtered_observations"]
    )

    filtered_observations_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Observations: {observations_data_path}")
    print(f"Taxa:         {taxa_data_path}")
    print(f"Output:       {filtered_observations_data_path}")
    print(f"Threads:      {pl.thread_pool_size()}")
    print("Scanning and filtering observations...")

    taxa = (
        pl.read_csv(
            taxa_data_path,
            separator="\t",
            quote_char=None,
            columns=[
                "taxon_id",
                "name",
                "rank",
                "ancestry",
                "active",
            ],
            schema_overrides={
                "active": pl.Boolean,
            },
        )
        .filter(
            (pl.col("rank") == "species")
            & (
                pl.col("ancestry")
                .str.starts_with("48460/1/2/355675/3")
                .fill_null(False)
            )
            & pl.col("active")
        )
        .select(["taxon_id", "name"])
    )

    observations = (
        pl.scan_csv(
            observations_data_path,
            separator="\t",
            quote_char=None,
        )
        .filter(
            pl.col("latitude").is_between(min_lat, max_lat)
            & pl.col("longitude").is_between(min_lon, max_lon)
            & (pl.col("quality_grade") == "research")
        )
        .join(
            taxa.lazy(),
            on="taxon_id",
            how="inner",
        )
        .filter(
            pl.col("anomaly_score").is_not_null()
            & (pl.col("anomaly_score") >= 1)
        )
    )

    observations.sink_parquet(
        filtered_observations_data_path,
        compression="lz4",
        maintain_order=False,
        engine="streaming",
    )


if __name__ == "__main__":
    filter_observations()