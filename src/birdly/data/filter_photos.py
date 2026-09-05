from pathlib import Path

import polars as pl

from birdly.data.utils import load_config


def filter_photos() -> None:
    config, local_config, _ = load_config()

    data_root = Path(local_config["raw_data_root"])

    photos_data_path = data_root / config["raw"]["photos"]

    filtered_observations_data_path = Path(
        config["processed"]["filtered_observations"]
    )

    filtered_photos_data_path = Path(
        config["processed"]["filtered_photos"]
    )

    filtered_photos_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    allowed_licenses = config["allowed_licenses"]

    print(f"Photos:       {photos_data_path}")
    print(f"Observations: {filtered_observations_data_path}")
    print(f"Output:       {filtered_photos_data_path}")
    print(f"Threads:      {pl.thread_pool_size()}")
    print("Scanning and filtering photos...")

    filtered_observations = (
        pl.scan_parquet(filtered_observations_data_path)
        .select([
            "observation_uuid",
            "taxon_id",
            "name",
        ])
    )

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
            filtered_observations,
            on="observation_uuid",
            how="inner",
        )
    )

    photos.sink_parquet(
        filtered_photos_data_path,
        compression="lz4",
        maintain_order=False,
        engine="streaming",
    )


if __name__ == "__main__":
    filter_photos()