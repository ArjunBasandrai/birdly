import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPException
from itertools import islice
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import polars as pl

from birdly.data.utils import load_config

BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos"
VALID_IMAGE_SIZES = {
    "square",
    "thumb",
    "small",
    "medium",
    "large",
    "original",
}
TEMPORARY_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_WORKERS = 16
MAX_ATTEMPTS = 3
CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30


def download_image(
    photo_id: int,
    extension: str,
    image_size: str,
    image_data_root: Path,
) -> bool:
    final_path = image_data_root / f"{photo_id}.{extension}"
    if final_path.exists():
        return False

    part_path = final_path.with_name(f"{final_path.name}.part")
    url = f"{BASE_URL}/{photo_id}/{image_size}.{extension}"
    request = Request(url, headers={"User-Agent": "birdly/0.1.0"})

    for attempt in range(MAX_ATTEMPTS):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Unexpected HTTP status {response.status} for {url}"
                    )

                with part_path.open("wb") as output_file:
                    shutil.copyfileobj(
                        response,
                        output_file,
                        length=CHUNK_SIZE,
                    )

            part_path.replace(final_path)
            return True
        except HTTPError as error:
            if (
                error.code not in TEMPORARY_HTTP_STATUS_CODES
                or attempt == MAX_ATTEMPTS - 1
            ):
                raise
        except (URLError, TimeoutError, ConnectionError, HTTPException):
            if attempt == MAX_ATTEMPTS - 1:
                raise

        time.sleep(2**attempt)

    return False


def download_images() -> None:
    config, local_config, params = load_config()

    manifest_path = Path(config["processed"]["manifest"])
    image_data_root = Path(local_config["image_data_root"])
    image_size = params["download"]["image_size"]

    if image_size not in VALID_IMAGE_SIZES:
        valid_sizes = ", ".join(sorted(VALID_IMAGE_SIZES))
        raise ValueError(
            f"Invalid image size '{image_size}'. Valid sizes: {valid_sizes}"
        )

    image_data_root.mkdir(parents=True, exist_ok=True)

    photos = (
        pl.read_parquet(
            manifest_path,
            columns=["photo_id", "extension"],
        )
        .unique()
        .iter_rows()
    )

    downloaded = 0
    skipped = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while batch := list(islice(photos, MAX_WORKERS * 4)):
            futures = {
                executor.submit(
                    download_image,
                    photo_id,
                    extension,
                    image_size,
                    image_data_root,
                ): (photo_id, extension)
                for photo_id, extension in batch
            }

            for future in as_completed(futures):
                photo_id, extension = futures[future]
                try:
                    if future.result():
                        downloaded += 1
                    else:
                        skipped += 1
                except (OSError, HTTPException, RuntimeError) as error:
                    failures += 1
                    print(f"Failed: {photo_id}.{extension}: {error}")

                processed = downloaded + skipped + failures
                if processed % 1000 == 0:
                    print(
                        f"Processed: {processed:,} | "
                        f"Downloaded: {downloaded:,} | "
                        f"Skipped: {skipped:,} | "
                        f"Failed: {failures:,}"
                    )

    print(
        f"Downloaded: {downloaded:,} | "
        f"Skipped: {skipped:,} | "
        f"Failed: {failures:,}"
    )

    if failures:
        raise RuntimeError(f"Failed to download {failures:,} images")


if __name__ == "__main__":
    download_images()
