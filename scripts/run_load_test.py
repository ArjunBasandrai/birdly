from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_VIRTUAL_USERS = (1, 2, 4, 8, 16)
DEFAULT_DURATION = "30s"
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_K6_IMAGE = "grafana/k6:1.3.0"
RELEVANT_METRIC_PREFIX = "birdly_"


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _fetch_text(url: str, timeout_seconds: float = 10.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot read {url}: {error}") from error


def _fetch_json(url: str) -> dict[str, Any]:
    payload = json.loads(_fetch_text(url))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object from {url}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
    ).strip()


def _create_results_directory(output_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _image_content_type(path: Path) -> str:
    content_types = {
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    try:
        return content_types[path.suffix.lower()]
    except KeyError as error:
        raise ValueError("Load-test image must be JPEG, PNG, or WebP") from error


def _docker_api_url(api_url: str) -> str:
    parsed = urllib.parse.urlsplit(api_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return api_url.rstrip("/")
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            f"host.docker.internal{port}",
            parsed.path.rstrip("/"),
            parsed.query,
            parsed.fragment,
        )
    )


def _parse_prometheus_metrics(payload: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in payload.splitlines():
        if not line.startswith(RELEVANT_METRIC_PREFIX):
            continue
        try:
            name, value = line.rsplit(None, 1)
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def _metric_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    return {
        name: value - before.get(name, 0.0)
        for name, value in after.items()
        if not name.split("{", 1)[0].endswith("_created")
    }


def _select_runner(requested_runner: str) -> tuple[str, str]:
    k6_executable = shutil.which("k6")
    docker_executable = shutil.which("docker")
    if requested_runner == "local":
        if k6_executable is None:
            raise RuntimeError("k6 is not installed or is not on PATH")
        return "local", k6_executable
    if requested_runner == "docker":
        if docker_executable is None:
            raise RuntimeError("Docker is not installed or is not on PATH")
        return "docker", docker_executable
    if k6_executable is not None:
        return "local", k6_executable
    if docker_executable is not None:
        return "docker", docker_executable
    raise RuntimeError("Install k6 or Docker before running the load test")


def _runner_version(runner: str, executable: str, k6_image: str) -> str:
    command = (
        [executable, "version"]
        if runner == "local"
        else [executable, "run", "--rm", k6_image, "version"]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return (result.stdout or result.stderr).strip()


def _k6_command(
    *,
    runner: str,
    executable: str,
    k6_image: str,
    script_path: Path,
    image_path: Path,
    level_directory: Path,
    api_url: str,
    virtual_users: int,
    duration: str,
    request_timeout: str,
    iterations: int | None,
) -> list[str]:
    environment = {
        "API_BASE_URL": api_url.rstrip("/"),
        "DURATION": duration,
        "IMAGE_CONTENT_TYPE": _image_content_type(image_path),
        "IMAGE_NAME": image_path.name,
        "REQUEST_TIMEOUT": request_timeout,
        "VUS": str(virtual_users),
    }
    if iterations is not None:
        environment["ITERATIONS"] = str(iterations)

    if runner == "local":
        environment["IMAGE_PATH"] = str(image_path)
        command = [executable, "run"]
        for name, value in environment.items():
            command.extend(["--env", f"{name}={value}"])
        command.extend(
            [
                "--quiet",
                "--summary-export",
                str(level_directory / "k6-summary.json"),
                "--out",
                f"json={level_directory / 'k6-samples.jsonl'}",
                "--summary-trend-stats",
                "avg,med,p(95),p(99),min,max",
                str(script_path),
            ]
        )
        return command

    environment["API_BASE_URL"] = _docker_api_url(api_url)
    environment["IMAGE_PATH"] = f"/input/{image_path.name}"
    command = [
        executable,
        "run",
        "--rm",
        "--mount",
        f"type=bind,source={script_path.parent},target=/scripts,readonly",
        "--mount",
        f"type=bind,source={image_path.parent},target=/input,readonly",
        "--mount",
        f"type=bind,source={level_directory},target=/results",
    ]
    for name, value in environment.items():
        command.extend(["--env", f"{name}={value}"])
    command.extend(
        [
            k6_image,
            "run",
            "--quiet",
            "--summary-export",
            "/results/k6-summary.json",
            "--out",
            "json=/results/k6-samples.jsonl",
            "--summary-trend-stats",
            "avg,med,p(95),p(99),min,max",
            "/scripts/predict.js",
        ]
    )
    return command


def _metric_value(
    summary: dict[str, Any],
    metric_name: str,
    value_name: str,
    default: float = 0.0,
) -> float:
    metric = summary.get("metrics", {}).get(metric_name, {})
    values = metric.get("values", metric)
    return float(values.get(value_name, default))


def _comparison_row(
    summary: dict[str, Any],
    virtual_users: int,
) -> dict[str, int | float]:
    request_count = int(_metric_value(summary, "http_reqs", "count"))
    error_count = int(_metric_value(summary, "birdly_error_requests", "count"))
    return {
        "virtual_users": virtual_users,
        "request_count": request_count,
        "requests_per_second": _metric_value(summary, "http_reqs", "rate"),
        "successful_request_count": int(
            _metric_value(summary, "birdly_successful_requests", "count")
        ),
        "error_count": error_count,
        "error_rate": error_count / request_count if request_count else 0.0,
        "latency_mean_ms": _metric_value(summary, "http_req_duration", "avg"),
        "latency_p50_ms": _metric_value(summary, "http_req_duration", "med"),
        "latency_p95_ms": _metric_value(summary, "http_req_duration", "p(95)"),
        "latency_p99_ms": _metric_value(summary, "http_req_duration", "p(99)"),
    }


def _print_table(rows: list[dict[str, int | float]]) -> None:
    print()
    print(" VUs |   req/s |  p50 ms |  p95 ms |  p99 ms | errors")
    print("-----+---------+---------+---------+---------+-------")
    for row in rows:
        print(
            f"{int(row['virtual_users']):4d} | "
            f"{float(row['requests_per_second']):7.2f} | "
            f"{float(row['latency_p50_ms']):7.2f} | "
            f"{float(row['latency_p95_ms']):7.2f} | "
            f"{float(row['latency_p99_ms']):7.2f} | "
            f"{int(row['error_count']):6d}"
        )


def _write_comparison_csv(
    path: Path,
    rows: list[dict[str, int | float]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Birdly API load tests.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--duration", default=DEFAULT_DURATION)
    parser.add_argument(
        "--vus",
        nargs="+",
        type=int,
        default=list(DEFAULT_VIRTUAL_USERS),
    )
    parser.add_argument("--request-timeout", default="35s")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--runner", choices=("auto", "local", "docker"), default="auto")
    parser.add_argument("--k6-image", default=DEFAULT_K6_IMAGE)
    parser.add_argument("--allow-rate-limit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_arguments()
    repository_root = Path(__file__).resolve().parents[1]
    script_path = repository_root / "loadtest" / "predict.js"
    image_path = args.image.expanduser().resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else repository_root / "artifacts" / "load_test"
    )
    results_directory = _create_results_directory(output_root)
    final_summary_path = results_directory / "final-summary.json"
    started_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "api_url": args.api_url.rstrip("/"),
        "duration": "1 iteration" if args.smoke else args.duration,
        "git_commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_output(repository_root, "status", "--porcelain")),
        "image": {
            "bytes": image_path.stat().st_size if image_path.is_file() else None,
            "path": str(image_path),
            "sha256": _file_sha256(image_path) if image_path.is_file() else None,
        },
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "started_at_utc": started_at.isoformat(),
        "virtual_users": [1] if args.smoke else args.vus,
    }
    _write_json(results_directory / "configuration.json", metadata)

    try:
        if not image_path.is_file():
            raise FileNotFoundError(f"Load-test image is missing: {image_path}")
        if any(virtual_users < 1 for virtual_users in args.vus):
            raise ValueError("Virtual-user counts must be positive")
        health = _fetch_json(f"{args.api_url.rstrip('/')}/health")
        rate_limit = health.get("rate_limit_per_minute")
        if rate_limit is None:
            raise RuntimeError(
                "The API health response does not include its rate limit"
            )
        if int(rate_limit) != 0 and not args.allow_rate_limit:
            raise RuntimeError(
                "API rate limiting is enabled. Restart the API with "
                "configs/load_test.env before this test."
            )

        runner, executable = _select_runner(args.runner)
        k6_version = _runner_version(runner, executable, args.k6_image)
        levels = [1] if args.smoke else list(dict.fromkeys(args.vus))
        metadata.update(
            {
                "api_health": health,
                "backend": health.get("backend"),
                "configured_cpu_threads": health.get("cpu_threads"),
                "configured_inference_concurrency": health.get("inference_concurrency"),
                "k6_image": args.k6_image if runner == "docker" else None,
                "k6_version": k6_version,
                "runner": runner,
                "virtual_users": levels,
            }
        )
        _write_json(results_directory / "configuration.json", metadata)

        comparison: list[dict[str, int | float]] = []
        for index, virtual_users in enumerate(levels, start=1):
            level_directory = results_directory / f"vus-{virtual_users:02d}"
            level_directory.mkdir()
            print(
                f"[{index}/{len(levels)}] Running VUs={virtual_users} "
                f"duration={'1 iteration' if args.smoke else args.duration}"
            )
            before_text = _fetch_text(f"{args.api_url.rstrip('/')}/metrics")
            before_metrics = _parse_prometheus_metrics(before_text)
            (level_directory / "api-metrics-before.prom").write_text(
                before_text, encoding="utf-8"
            )
            _write_json(level_directory / "api-metrics-before.json", before_metrics)

            command = _k6_command(
                runner=runner,
                executable=executable,
                k6_image=args.k6_image,
                script_path=script_path,
                image_path=image_path,
                level_directory=level_directory.resolve(),
                api_url=args.api_url,
                virtual_users=virtual_users,
                duration=args.duration,
                request_timeout=args.request_timeout,
                iterations=1 if args.smoke else None,
            )
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            (level_directory / "k6-console.txt").write_text(
                result.stdout + result.stderr,
                encoding="utf-8",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"k6 failed for VUs={virtual_users}. See "
                    f"{level_directory / 'k6-console.txt'}"
                )

            after_text = _fetch_text(f"{args.api_url.rstrip('/')}/metrics")
            after_metrics = _parse_prometheus_metrics(after_text)
            (level_directory / "api-metrics-after.prom").write_text(
                after_text, encoding="utf-8"
            )
            _write_json(level_directory / "api-metrics-after.json", after_metrics)
            delta = _metric_delta(before_metrics, after_metrics)
            _write_json(level_directory / "api-metrics-delta.json", delta)

            k6_summary_path = level_directory / "k6-summary.json"
            k6_summary = json.loads(k6_summary_path.read_text(encoding="utf-8"))
            row = _comparison_row(k6_summary, virtual_users)
            comparison.append(row)
            _write_json(
                level_directory / "result.json",
                {
                    "api_metrics": {
                        "after": after_metrics,
                        "before": before_metrics,
                        "delta": delta,
                    },
                    "configuration": {
                        "api_url": args.api_url.rstrip("/"),
                        "duration": "1 iteration" if args.smoke else args.duration,
                        "image_path": str(image_path),
                        "virtual_users": virtual_users,
                    },
                    "summary": row,
                },
            )
            print(
                f"    completed: {row['request_count']} requests, "
                f"{float(row['requests_per_second']):.2f} req/s, "
                f"{row['error_count']} errors"
            )

        _write_json(results_directory / "comparison.json", comparison)
        _write_comparison_csv(results_directory / "comparison.csv", comparison)
        final_summary = {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "configuration": metadata,
            "results": comparison,
            "status": "completed",
        }
        _write_json(final_summary_path, final_summary)
        _print_table(comparison)
        print(f"\nResults: {results_directory}")
        return 0
    except Exception as error:  # noqa: BLE001
        _write_json(
            final_summary_path,
            {
                "error": f"{type(error).__name__}: {error}",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "configuration": metadata,
                "status": "failed",
            },
        )
        print(f"Load test failed: {error}", file=sys.stderr)
        print(f"Partial results: {results_directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
