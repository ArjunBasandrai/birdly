from pathlib import Path

import polars as pl
import yaml


def filter_to_bounds() -> None:
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(".birdly/local.yaml", "r") as f:
        local_config = yaml.safe_load(f)

    bbox = config["geography"]["bounding_box"]

    min_lat = bbox["min_lat"]
    max_lat = bbox["max_lat"]
    min_lon = bbox["min_lon"]
    max_lon = bbox["max_lon"]

    data_root = Path(local_config["raw_data_root"])

    observations_data_path = (
        data_root / config["raw"]["observations"]
    )

    filtered_observations_data_path = Path(
        config["processed"]["filtered_observations"]
    )

    filtered_observations_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input:   {observations_data_path}")
    print(f"Output:  {filtered_observations_data_path}")
    print(f"Threads: {pl.thread_pool_size()}")
    print("Scanning and filtering observations...")

    (
        pl.scan_csv(observations_data_path, separator='\t')
        .filter(
            pl.col("latitude").is_between(min_lat, max_lat)
            & pl.col("longitude").is_between(min_lon, max_lon)
        )
        .sink_parquet(
            filtered_observations_data_path,
            compression="lz4",
            maintain_order=False,
            engine="streaming",
        )
    )

    print("Done.")


if __name__ == "__main__":
    filter_to_bounds()