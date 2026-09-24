from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT / "src" / "birdly" / "inference" / "taxon_names.json"
)
INATURALIST_TAXA_URL = "https://api.inaturalist.org/v1/taxa/{taxon_ids}"
DEFAULT_LOCALE = "en"
DEFAULT_PREFERRED_PLACE_ID = 48
DEFAULT_BATCH_SIZE = 30
DEFAULT_REQUEST_INTERVAL_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_USER_AGENT = (
    "Birdly/0.1 taxon-name-generator (contact: arjunbasandrai2004@gmail.com)"
)

TaxonNames = dict[str, dict[str, str]]


def load_supported_taxon_ids(class_mapping_path: Path) -> list[int]:
    payload = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    class_to_taxon = payload.get("class_to_taxon")
    if not isinstance(class_to_taxon, dict):
        raise TypeError("Class mapping must contain class_to_taxon.")

    ordered_items = sorted(class_to_taxon.items(), key=lambda item: int(item[0]))
    expected_indices = list(range(len(ordered_items)))
    actual_indices = [int(class_index) for class_index, _ in ordered_items]
    if actual_indices != expected_indices:
        raise ValueError("Class mapping indices must be contiguous and start at zero.")

    taxon_ids = [int(taxon_id) for _, taxon_id in ordered_items]
    if len(taxon_ids) != len(set(taxon_ids)):
        raise ValueError("Class mapping contains duplicate taxon IDs.")
    if int(payload.get("class_count", len(taxon_ids))) != len(taxon_ids):
        raise ValueError("Class count does not match class_to_taxon.")
    return taxon_ids


def validate_taxon_names(payload: object, taxon_ids: Sequence[int]) -> bool:
    if not isinstance(payload, dict):
        return False
    expected_ids = {str(taxon_id) for taxon_id in taxon_ids}
    if set(payload) != expected_ids:
        return False
    for value in payload.values():
        if not isinstance(value, dict):
            return False
        common_name = value.get("common_name")
        scientific_name = value.get("scientific_name")
        if not isinstance(common_name, str) or not common_name.strip():
            return False
        if not isinstance(scientific_name, str) or not scientific_name.strip():
            return False
    return True


def parse_taxa_response(payload: object, requested_ids: Sequence[int]) -> TaxonNames:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise TypeError("iNaturalist response does not contain a results list.")

    requested = set(requested_ids)
    parsed: TaxonNames = {}
    for item in payload["results"]:
        if not isinstance(item, dict):
            raise TypeError("iNaturalist returned an invalid taxon record.")
        taxon_id = int(item.get("id", -1))
        if taxon_id not in requested:
            continue
        scientific_name = item.get("name")
        if not isinstance(scientific_name, str) or not scientific_name.strip():
            raise ValueError(f"Taxon {taxon_id} has no scientific name.")
        scientific_name = scientific_name.strip()
        preferred_common_name = item.get("preferred_common_name")
        common_name = (
            preferred_common_name.strip()
            if isinstance(preferred_common_name, str) and preferred_common_name.strip()
            else scientific_name
        )
        key = str(taxon_id)
        if key in parsed:
            raise ValueError(f"iNaturalist returned taxon {taxon_id} more than once.")
        parsed[key] = {
            "common_name": common_name,
            "scientific_name": scientific_name,
        }

    missing_ids = requested - {int(taxon_id) for taxon_id in parsed}
    if missing_ids:
        missing_text = ", ".join(str(taxon_id) for taxon_id in sorted(missing_ids))
        raise ValueError(f"iNaturalist did not return these taxon IDs: {missing_text}")
    return parsed


def _fetch_json(url: str, *, user_agent: str, timeout_seconds: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": user_agent},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == 2:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay_seconds = max(1.0, float(retry_after))
            except (TypeError, ValueError):
                delay_seconds = float(2**attempt)
            time.sleep(delay_seconds)
    raise RuntimeError("The iNaturalist request did not complete.")


def fetch_taxon_names(
    taxon_ids: Sequence[int],
    *,
    locale: str = DEFAULT_LOCALE,
    preferred_place_id: int = DEFAULT_PREFERRED_PLACE_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
) -> TaxonNames:
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if request_interval_seconds < 0:
        raise ValueError("Request interval must not be negative.")

    names: TaxonNames = {}
    batches = [
        taxon_ids[index : index + batch_size]
        for index in range(0, len(taxon_ids), batch_size)
    ]
    for batch_index, batch in enumerate(batches):
        if batch_index:
            time.sleep(request_interval_seconds)
        taxon_id_text = ",".join(str(taxon_id) for taxon_id in batch)
        query = urllib.parse.urlencode(
            {"locale": locale, "preferred_place_id": preferred_place_id}
        )
        url = f"{INATURALIST_TAXA_URL.format(taxon_ids=taxon_id_text)}?{query}"
        print(f"Requesting iNaturalist taxon batch {batch_index + 1}/{len(batches)}...")
        response_payload = _fetch_json(
            url,
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
        )
        batch_names = parse_taxa_response(response_payload, batch)
        overlap = set(names).intersection(batch_names)
        if overlap:
            raise ValueError(f"Duplicate taxon IDs across batches: {sorted(overlap)}")
        names.update(batch_names)
    return names


def generate_taxon_names(
    class_mapping_path: Path,
    output_path: Path,
    *,
    refresh: bool = False,
    fetcher: Callable[[Sequence[int]], TaxonNames] = fetch_taxon_names,
) -> tuple[TaxonNames, bool]:
    taxon_ids = load_supported_taxon_ids(class_mapping_path)
    if output_path.exists() and not refresh:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if validate_taxon_names(existing, taxon_ids):
            return existing, True

    names = fetcher(taxon_ids)
    if not validate_taxon_names(names, taxon_ids):
        raise ValueError("Generated taxon-name metadata is incomplete or invalid.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(names, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return names, False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Birdly taxon-name metadata from the iNaturalist API."
    )
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fetcher = lambda taxon_ids: fetch_taxon_names(
        taxon_ids,
        batch_size=args.batch_size,
        request_interval_seconds=args.request_interval_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    names, reused = generate_taxon_names(
        args.class_mapping.resolve(),
        args.output.resolve(),
        refresh=args.refresh,
        fetcher=fetcher,
    )
    action = "Reused" if reused else "Generated"
    print(f"{action} {len(names)} taxon-name records at {args.output.resolve()}")


if __name__ == "__main__":
    main()
