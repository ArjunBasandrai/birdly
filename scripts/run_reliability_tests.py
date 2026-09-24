from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_CONCURRENT_REQUESTS = 4
DEFAULT_BURST_REQUESTS = 8
DEFAULT_MAXIMUM_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 35.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0
MAX_REMOTE_RATE_LIMIT_REQUESTS = 100
RELEVANT_METRIC_PREFIX = "birdly_"
VALID_GIF_1X1 = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes
    latency_ms: float


@dataclass
class TestResult:
    name: str
    expected: str
    actual: str
    result: str
    http_status: int | None
    latency_ms: float | None
    detail: str = ""


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
    ).strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_results_directory(output_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _request(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> HttpResult:
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers or {},
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read()
            status = response.status
            response_headers = {
                key.lower(): value for key, value in response.headers.items()
            }
    except urllib.error.HTTPError as error:
        response_body = error.read()
        status = error.code
        response_headers = {key.lower(): value for key, value in error.headers.items()}
    return HttpResult(
        status=status,
        headers=response_headers,
        body=response_body,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def _multipart_body(
    payload: bytes,
    *,
    filename: str,
    content_type: str,
) -> tuple[bytes, str]:
    boundary = f"birdly-{uuid.uuid4().hex}"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="image"; '
        f'filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    return prefix + payload + suffix, boundary


def _post_image(
    api_url: str,
    payload: bytes,
    *,
    filename: str = "bird.jpg",
    content_type: str = "image/jpeg",
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> HttpResult:
    body, boundary = _multipart_body(
        payload,
        filename=filename,
        content_type=content_type,
    )
    return _request(
        "POST",
        f"{api_url.rstrip('/')}/predict",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout_seconds=timeout_seconds,
    )


def _health(api_url: str) -> tuple[HttpResult, dict[str, Any] | None]:
    result = _request("GET", f"{api_url.rstrip('/')}/health")
    try:
        payload = json.loads(result.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = None
    return result, payload if isinstance(payload, dict) else None


def _fetch_metrics(api_url: str) -> tuple[str, dict[str, float]]:
    result = _request("GET", f"{api_url.rstrip('/')}/metrics")
    if result.status != 200:
        raise RuntimeError(f"Metrics endpoint returned {result.status}")
    payload = result.body.decode("utf-8")
    metrics: dict[str, float] = {}
    for line in payload.splitlines():
        if not line.startswith(RELEVANT_METRIC_PREFIX):
            continue
        try:
            name, value = line.rsplit(None, 1)
            metrics[name] = float(value)
        except ValueError:
            continue
    return payload, metrics


def _metric_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    return {
        name: value - before.get(name, 0.0)
        for name, value in after.items()
        if not name.split("{", 1)[0].endswith("_created")
    }


def _body_detail(body: bytes) -> str:
    try:
        payload = json.loads(body)
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return body[:200].decode("utf-8", errors="replace")


def _status_result(
    name: str,
    expected_status: int,
    response: HttpResult,
    *,
    extra_pass: bool = True,
    detail: str = "",
) -> TestResult:
    passed = response.status == expected_status and extra_pass
    return TestResult(
        name=name,
        expected=str(expected_status),
        actual=str(response.status),
        result="PASS" if passed else "FAIL",
        http_status=response.status,
        latency_ms=response.latency_ms,
        detail=detail or _body_detail(response.body),
    )


def _concurrent_posts(
    api_url: str,
    requests: list[tuple[bytes, str]],
    *,
    timeout_seconds: float,
) -> list[HttpResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [
            executor.submit(
                _post_image,
                api_url,
                payload,
                content_type=content_type,
                timeout_seconds=timeout_seconds,
            )
            for payload, content_type in requests
        ]
        return [future.result() for future in futures]


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _docker_json(*arguments: str) -> Any:
    output = subprocess.check_output(["docker", *arguments], text=True)
    return json.loads(output)


def _inspect_container(container_name: str) -> tuple[str, str]:
    details = _docker_json("inspect", container_name)[0]
    image = str(details["Config"]["Image"])
    for mount in details.get("Mounts", []):
        if mount.get("Destination") == "/model":
            return image, str(mount["Source"])
    raise RuntimeError(f"Container {container_name} has no /model mount")


def _start_temporary_container(
    *,
    name: str,
    image: str,
    model_source: str | None,
    environment: dict[str, str],
) -> tuple[str, int]:
    port = _available_port()
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{port}:8000",
    ]
    for key, value in environment.items():
        command.extend(["--env", f"{key}={value}"])
    if model_source is not None:
        command.extend(
            [
                "--mount",
                f"type=bind,source={model_source},target=/model,readonly",
            ]
        )
    command.append(image)
    subprocess.run(command, capture_output=True, text=True, check=True)
    return f"http://127.0.0.1:{port}", port


def _remove_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True,
        text=True,
        check=False,
    )


def _wait_for_http(
    api_url: str,
    *,
    expected_statuses: set[int],
    timeout_seconds: float,
) -> HttpResult:
    deadline = time.monotonic() + timeout_seconds
    last_error = "No response"
    while time.monotonic() < deadline:
        try:
            result, _ = _health(api_url)
            if result.status in expected_statuses:
                return result
            last_error = f"HTTP {result.status}"
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise TimeoutError(f"API did not become ready: {last_error}")


def _test_health(api_url: str, expected_status: int, name: str) -> TestResult:
    response, _ = _health(api_url)
    return _status_result(name, expected_status, response)


def _test_valid_prediction(api_url: str, image: bytes) -> TestResult:
    response = _post_image(api_url, image)
    valid_body = False
    if response.status == 200:
        try:
            payload = json.loads(response.body)
            valid_body = (
                isinstance(payload, list)
                and len(payload) <= 5
                and all(
                    isinstance(prediction, dict)
                    and set(prediction)
                    == {"taxon_id", "common_name", "scientific_name", "score"}
                    for prediction in payload
                )
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return _status_result(
        "valid_prediction",
        200,
        response,
        extra_pass=valid_body,
        detail="zero to five predictions"
        if valid_body
        else _body_detail(response.body),
    )


def _test_concurrent_valid(
    api_url: str,
    image: bytes,
    count: int,
    timeout_seconds: float,
) -> TestResult:
    started = time.perf_counter()
    responses = _concurrent_posts(
        api_url,
        [(image, "image/jpeg")] * count,
        timeout_seconds=timeout_seconds,
    )
    statuses = [response.status for response in responses]
    return TestResult(
        name="concurrent_valid_requests",
        expected=f"{count} x 200",
        actual=str(statuses),
        result="PASS" if statuses == [200] * count else "FAIL",
        http_status=200 if all(status == 200 for status in statuses) else None,
        latency_ms=(time.perf_counter() - started) * 1000,
        detail=f"statuses={statuses}",
    )


def _test_mixed_concurrent(
    api_url: str,
    image: bytes,
    count: int,
    timeout_seconds: float,
) -> TestResult:
    valid_count = max(1, count // 2)
    invalid_count = max(1, count - valid_count)
    requests = [(image, "image/jpeg")] * valid_count + [
        (b"not an image", "image/jpeg")
    ] * invalid_count
    started = time.perf_counter()
    responses = _concurrent_posts(
        api_url,
        requests,
        timeout_seconds=timeout_seconds,
    )
    statuses = [response.status for response in responses]
    expected = [200] * valid_count + [400] * invalid_count
    return TestResult(
        name="mixed_concurrent_requests",
        expected=str(expected),
        actual=str(statuses),
        result="PASS" if statuses == expected else "FAIL",
        http_status=None,
        latency_ms=(time.perf_counter() - started) * 1000,
        detail=f"valid={valid_count}, invalid={invalid_count}",
    )


def _test_burst(
    api_url: str,
    image: bytes,
    count: int,
    timeout_seconds: float,
) -> TestResult:
    started = time.perf_counter()
    responses = _concurrent_posts(
        api_url,
        [(image, "image/jpeg")] * count,
        timeout_seconds=timeout_seconds,
    )
    statuses = [response.status for response in responses]
    health_response, _ = _health(api_url)
    passed = statuses == [200] * count and health_response.status == 200
    return TestResult(
        name="safe_burst_traffic",
        expected=f"{count} x 200 and healthy",
        actual=f"statuses={statuses}, health={health_response.status}",
        result="PASS" if passed else "FAIL",
        http_status=health_response.status,
        latency_ms=(time.perf_counter() - started) * 1000,
        detail="bounded safe burst",
    )


def _test_rate_limit(
    *,
    image: bytes,
    image_name: str,
    model_source: str,
    startup_timeout_seconds: float,
) -> list[TestResult]:
    temporary_name = f"birdly-reliability-rate-{os.getpid()}"
    try:
        api_url, _ = _start_temporary_container(
            name=temporary_name,
            image=image_name,
            model_source=model_source,
            environment={"BIRDLY_RATE_LIMIT_PER_MINUTE": "2"},
        )
        _wait_for_http(
            api_url,
            expected_statuses={200},
            timeout_seconds=startup_timeout_seconds,
        )
        responses = [_post_image(api_url, image) for _ in range(3)]
        statuses = [response.status for response in responses]
        retry_after = responses[-1].headers.get("retry-after")
        overflow_passed = statuses == [200, 200, 429]
        retry_after_passed = responses[-1].status == 429 and retry_after is not None
        return [
            TestResult(
                name="rate_limit_overflow",
                expected="[200, 200, 429]",
                actual=str(statuses),
                result="PASS" if overflow_passed else "FAIL",
                http_status=responses[-1].status,
                latency_ms=responses[-1].latency_ms,
                detail=f"temporary_container={temporary_name}",
            ),
            TestResult(
                name="rate_limit_retry_after",
                expected="Retry-After header",
                actual=retry_after or "missing",
                result="PASS" if retry_after_passed else "FAIL",
                http_status=responses[-1].status,
                latency_ms=responses[-1].latency_ms,
                detail=f"temporary_container={temporary_name}",
            ),
        ]
    finally:
        _remove_container(temporary_name)


def _test_remote_rate_limit(
    *,
    api_url: str,
    image: bytes,
    rate_limit_per_minute: int,
) -> list[TestResult]:
    maximum_requests = rate_limit_per_minute + 1
    if maximum_requests > MAX_REMOTE_RATE_LIMIT_REQUESTS:
        raise RuntimeError(
            "The reported rate limit is too high for the bounded remote test: "
            f"{rate_limit_per_minute}"
        )

    responses: list[HttpResult] = []
    for _ in range(maximum_requests):
        response = _post_image(api_url, image)
        responses.append(response)
        if response.status == 429:
            break

    statuses = [response.status for response in responses]
    rejected_response = next(
        (response for response in responses if response.status == 429),
        None,
    )
    rejected_index = statuses.index(429) if rejected_response is not None else None
    preceding_statuses = statuses[:rejected_index] if rejected_index is not None else []
    overflow_passed = rejected_response is not None and all(
        status == 200 for status in preceding_statuses
    )
    retry_after = (
        rejected_response.headers.get("retry-after")
        if rejected_response is not None
        else None
    )
    health_response, _ = _health(api_url)
    return [
        TestResult(
            name="rate_limit_overflow",
            expected=f"429 within {maximum_requests} requests",
            actual=str(statuses),
            result="PASS" if overflow_passed else "FAIL",
            http_status=(
                rejected_response.status if rejected_response is not None else None
            ),
            latency_ms=(
                rejected_response.latency_ms if rejected_response is not None else None
            ),
            detail=f"reported_limit={rate_limit_per_minute}",
        ),
        TestResult(
            name="rate_limit_retry_after",
            expected="Retry-After header",
            actual=retry_after or "missing",
            result="PASS" if retry_after is not None else "FAIL",
            http_status=(
                rejected_response.status if rejected_response is not None else None
            ),
            latency_ms=(
                rejected_response.latency_ms if rejected_response is not None else None
            ),
            detail=f"reported_limit={rate_limit_per_minute}",
        ),
        _status_result(
            "health_after_rate_limit",
            200,
            health_response,
            detail="health after rate-limit verification",
        ),
    ]


def _test_queue_timeout(
    *,
    image: bytes,
    image_name: str,
    model_source: str,
    startup_timeout_seconds: float,
) -> TestResult:
    temporary_name = f"birdly-reliability-timeout-{os.getpid()}"
    try:
        api_url, _ = _start_temporary_container(
            name=temporary_name,
            image=image_name,
            model_source=model_source,
            environment={
                "BIRDLY_MAX_INFERENCE_CONCURRENCY": "1",
                "BIRDLY_RATE_LIMIT_PER_MINUTE": "0",
                "BIRDLY_REQUEST_TIMEOUT_SECONDS": "0.05",
            },
        )
        _wait_for_http(
            api_url,
            expected_statuses={200},
            timeout_seconds=startup_timeout_seconds,
        )
        responses = _concurrent_posts(
            api_url,
            [(image, "image/jpeg"), (image, "image/jpeg")],
            timeout_seconds=10.0,
        )
        statuses = [response.status for response in responses]
        passed = 503 in statuses and all(status in {200, 503} for status in statuses)
        return TestResult(
            name="inference_queue_timeout",
            expected="at least one 503",
            actual=str(statuses),
            result="PASS" if passed else "FAIL",
            http_status=503 if 503 in statuses else None,
            latency_ms=max(response.latency_ms for response in responses),
            detail="temporary timeout=0.05s, concurrency=1",
        )
    finally:
        _remove_container(temporary_name)


def _test_missing_model(
    *,
    image_name: str,
    startup_timeout_seconds: float,
) -> TestResult:
    temporary_name = f"birdly-reliability-missing-{os.getpid()}"
    try:
        api_url, _ = _start_temporary_container(
            name=temporary_name,
            image=image_name,
            model_source=None,
            environment={"BIRDLY_RATE_LIMIT_PER_MINUTE": "0"},
        )
        response = _wait_for_http(
            api_url,
            expected_statuses={503},
            timeout_seconds=startup_timeout_seconds,
        )
        return _status_result("missing_model_health", 503, response)
    finally:
        _remove_container(temporary_name)


def _test_restart(
    *,
    container_name: str,
    api_url: str,
    image: bytes,
    startup_timeout_seconds: float,
) -> TestResult:
    started = time.perf_counter()
    subprocess.run(
        ["docker", "restart", container_name],
        capture_output=True,
        text=True,
        check=True,
    )
    health_response = _wait_for_http(
        api_url,
        expected_statuses={200},
        timeout_seconds=startup_timeout_seconds,
    )
    prediction = _post_image(api_url, image)
    passed = health_response.status == 200 and prediction.status == 200
    return TestResult(
        name="container_restart_recovery",
        expected="health 200 and prediction 200",
        actual=f"health={health_response.status}, prediction={prediction.status}",
        result="PASS" if passed else "FAIL",
        http_status=prediction.status,
        latency_ms=(time.perf_counter() - started) * 1000,
        detail=f"container={container_name}",
    )


def _test_metrics_increase(
    before: dict[str, float],
    after: dict[str, float],
) -> TestResult:
    prediction_increase = sum(
        value - before.get(name, 0.0)
        for name, value in after.items()
        if name.startswith("birdly_prediction_requests_total")
    )
    error_increase = sum(
        value - before.get(name, 0.0)
        for name, value in after.items()
        if name.startswith("birdly_prediction_errors_total")
    )
    passed = prediction_increase > 0 and error_increase > 0
    return TestResult(
        name="prometheus_failure_counters",
        expected="prediction and error counters increase",
        actual=f"predictions=+{prediction_increase:g}, errors=+{error_increase:g}",
        result="PASS" if passed else "FAIL",
        http_status=200,
        latency_ms=None,
        detail="main API metrics",
    )


def _test_active_inference_zero(api_url: str) -> TestResult:
    deadline = time.monotonic() + 10.0
    active = float("nan")
    while time.monotonic() < deadline:
        _, metrics = _fetch_metrics(api_url)
        active = metrics.get("birdly_active_inference_requests", float("nan"))
        if active == 0:
            break
        time.sleep(0.25)
    return TestResult(
        name="active_inference_returns_to_zero",
        expected="0",
        actual=str(active),
        result="PASS" if active == 0 else "FAIL",
        http_status=200,
        latency_ms=None,
        detail="birdly_active_inference_requests",
    )


def _run_named_test(
    results: list[TestResult],
    name: str,
    test: Callable[[], TestResult],
) -> None:
    print(f"Running {name} ...", flush=True)
    try:
        result = test()
    except Exception as error:  # noqa: BLE001
        result = TestResult(
            name=name,
            expected="test completes",
            actual=type(error).__name__,
            result="FAIL",
            http_status=None,
            latency_ms=None,
            detail=str(error),
        )
    results.append(result)
    print(f"  {result.result}: expected={result.expected} actual={result.actual}")


def _run_rate_limit_tests(
    results: list[TestResult],
    test: Callable[[], list[TestResult]],
) -> None:
    print("Running rate_limit_overflow and rate_limit_retry_after ...", flush=True)
    try:
        rate_results = test()
    except Exception as error:  # noqa: BLE001
        rate_results = [
            TestResult(
                name=name,
                expected=expected,
                actual=type(error).__name__,
                result="FAIL",
                http_status=None,
                latency_ms=None,
                detail=str(error),
            )
            for name, expected in (
                ("rate_limit_overflow", "[200, 200, 429]"),
                ("rate_limit_retry_after", "Retry-After header"),
            )
        ]
    results.extend(rate_results)
    for result in rate_results:
        print(f"  {result.result}: {result.name} actual={result.actual}")


def _run_remote_rate_limit_tests(
    results: list[TestResult],
    test: Callable[[], list[TestResult]],
) -> None:
    print("Running remote rate-limit verification ...", flush=True)
    try:
        rate_results = test()
    except Exception as error:  # noqa: BLE001
        rate_results = [
            TestResult(
                name=name,
                expected=expected,
                actual=type(error).__name__,
                result="FAIL",
                http_status=None,
                latency_ms=None,
                detail=str(error),
            )
            for name, expected in (
                ("rate_limit_overflow", "eventual 429"),
                ("rate_limit_retry_after", "Retry-After header"),
                ("health_after_rate_limit", "200"),
            )
        ]
    results.extend(rate_results)
    for result in rate_results:
        print(f"  {result.result}: {result.name} actual={result.actual}")


def _skip(
    results: list[TestResult],
    name: str,
    reason: str,
    *,
    expected: str = "Docker condition enabled",
    actual: str = "not enabled",
) -> None:
    results.append(
        TestResult(
            name=name,
            expected=expected,
            actual=actual,
            result="SKIP",
            http_status=None,
            latency_ms=None,
            detail=reason,
        )
    )
    print(f"Skipping {name}: {reason}")


def _write_results_csv(path: Path, results: list[TestResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def _print_table(results: list[TestResult]) -> None:
    widths = {
        "name": max(28, *(len(result.name) for result in results)),
        "expected": max(10, *(len(result.expected) for result in results)),
        "actual": min(32, max(8, *(len(result.actual) for result in results))),
    }
    header = (
        f"{'Test':<{widths['name']}}  "
        f"{'Expected':<{widths['expected']}}  "
        f"{'Actual':<{widths['actual']}}  Result"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for result in results:
        actual = result.actual
        if len(actual) > widths["actual"]:
            actual = actual[: widths["actual"] - 3] + "..."
        print(
            f"{result.name:<{widths['name']}}  "
            f"{result.expected:<{widths['expected']}}  "
            f"{actual:<{widths['actual']}}  {result.result}"
        )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Birdly API reliability tests.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--container")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--remote", action="store_true")
    mode_group.add_argument("--remote-rate-limit-test", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=DEFAULT_CONCURRENT_REQUESTS,
    )
    parser.add_argument("--burst-requests", type=int, default=DEFAULT_BURST_REQUESTS)
    parser.add_argument(
        "--maximum-upload-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_UPLOAD_BYTES,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )
    parser.add_argument("--docker-restart-tests", action="store_true")
    parser.add_argument("--docker-missing-model-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.remote_rate_limit_test and args.smoke:
        parser.error("--smoke cannot be used with --remote-rate-limit-test")
    if (args.remote or args.remote_rate_limit_test) and (
        args.docker_restart_tests or args.docker_missing_model_test
    ):
        parser.error("Docker condition flags cannot be used in remote mode")
    return args


def main() -> int:
    args = _parse_arguments()
    mode = (
        "remote-rate-limit"
        if args.remote_rate_limit_test
        else "remote"
        if args.remote
        else "local"
    )
    repository_root = Path(__file__).resolve().parents[1]
    image_path = args.image.expanduser().resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else repository_root / "artifacts" / "reliability"
    )
    results_directory = _create_results_directory(output_root)
    results: list[TestResult] = []
    started_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "api_url": args.api_url.rstrip("/"),
        "burst_requests": args.burst_requests,
        "concurrent_requests": args.concurrent_requests,
        "container": args.container,
        "docker_missing_model_test": args.docker_missing_model_test,
        "docker_restart_tests": args.docker_restart_tests,
        "git_commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_output(repository_root, "status", "--porcelain")),
        "image": {
            "bytes": image_path.stat().st_size if image_path.is_file() else None,
            "path": str(image_path),
            "sha256": _file_sha256(image_path) if image_path.is_file() else None,
        },
        "maximum_upload_bytes": args.maximum_upload_bytes,
        "mode": mode,
        "metrics_snapshots": {
            "after": "after request cases and before an optional target restart",
            "before": "before request cases",
            "final": "after all cases, including an optional target restart",
        },
        "request_timeout_seconds": args.request_timeout_seconds,
        "startup_timeout_seconds": args.startup_timeout_seconds,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "smoke": args.smoke,
        "started_at_utc": started_at.isoformat(),
    }
    _write_json(results_directory / "configuration.json", metadata)

    try:
        if not image_path.is_file():
            raise FileNotFoundError(f"Valid test image is missing: {image_path}")
        if args.concurrent_requests < 2 or args.burst_requests < 2:
            raise ValueError("Concurrency and burst counts must be at least 2")
        if mode == "local" and not args.smoke and not args.container:
            raise ValueError("--container is required for the full reliability suite")
        valid_image = image_path.read_bytes()
        health_response, health_payload = _health(args.api_url)
        metadata["api_health"] = health_payload
        _write_json(results_directory / "configuration.json", metadata)
        if mode == "remote" and not args.smoke:
            if health_response.status != 200 or not isinstance(health_payload, dict):
                raise RuntimeError(
                    "The remote suite requires a healthy API with configuration "
                    "metadata"
                )
            if health_payload.get("rate_limit_per_minute") != 0:
                raise RuntimeError(
                    "The remote suite requires rate_limit_per_minute=0. Change "
                    "the Cloud Run setting externally before this test."
                )
        elif mode == "remote-rate-limit":
            if health_response.status != 200 or not isinstance(health_payload, dict):
                raise RuntimeError(
                    "The remote rate-limit test requires a healthy API with "
                    "configuration metadata"
                )
            rate_limit_per_minute = health_payload.get("rate_limit_per_minute")
            if not isinstance(rate_limit_per_minute, int) or rate_limit_per_minute < 1:
                raise RuntimeError(
                    "The remote rate-limit test requires "
                    "rate_limit_per_minute>0. Restore the Cloud Run setting "
                    "externally before this test."
                )
        elif (
            not args.smoke
            and health_response.status == 200
            and isinstance(health_payload, dict)
            and health_payload.get("rate_limit_per_minute") != 0
        ):
            raise RuntimeError(
                "The full suite requires BIRDLY_RATE_LIMIT_PER_MINUTE=0 on the "
                "target container"
            )

        before_text, before_metrics = _fetch_metrics(args.api_url)
        (results_directory / "api-metrics-before.prom").write_text(
            before_text, encoding="utf-8"
        )
        _write_json(results_directory / "api-metrics-before.json", before_metrics)

        if mode == "remote-rate-limit":
            _run_remote_rate_limit_tests(
                results,
                lambda: _test_remote_rate_limit(
                    api_url=args.api_url,
                    image=valid_image,
                    rate_limit_per_minute=rate_limit_per_minute,
                ),
            )
            after_text, after_metrics = _fetch_metrics(args.api_url)
            (results_directory / "api-metrics-after.prom").write_text(
                after_text, encoding="utf-8"
            )
            _write_json(results_directory / "api-metrics-after.json", after_metrics)
            _write_json(
                results_directory / "api-metrics-delta.json",
                _metric_delta(before_metrics, after_metrics),
            )
        else:
            _run_named_test(
                results,
                "healthy_model_health",
                lambda: _status_result(
                    "healthy_model_health", 200, health_response, detail="model loaded"
                ),
            )
            _run_named_test(
                results,
                "valid_prediction",
                lambda: _test_valid_prediction(args.api_url, valid_image),
            )

        if mode != "remote-rate-limit" and not args.smoke:
            _run_named_test(
                results,
                "corrupt_image",
                lambda: _status_result(
                    "corrupt_image",
                    400,
                    _post_image(args.api_url, b"not an image"),
                ),
            )
            _run_named_test(
                results,
                "unsupported_content_type",
                lambda: _status_result(
                    "unsupported_content_type",
                    400,
                    _post_image(
                        args.api_url,
                        valid_image,
                        content_type="application/octet-stream",
                    ),
                ),
            )
            _run_named_test(
                results,
                "unsupported_image_format",
                lambda: _status_result(
                    "unsupported_image_format",
                    400,
                    _post_image(
                        args.api_url,
                        VALID_GIF_1X1,
                        filename="bird.gif",
                        content_type="image/jpeg",
                    ),
                ),
            )
            _run_named_test(
                results,
                "empty_upload",
                lambda: _status_result(
                    "empty_upload", 400, _post_image(args.api_url, b"")
                ),
            )
            _run_named_test(
                results,
                "oversized_upload",
                lambda: _status_result(
                    "oversized_upload",
                    413,
                    _post_image(
                        args.api_url,
                        bytes(args.maximum_upload_bytes + 1),
                    ),
                ),
            )

            def malformed_request() -> TestResult:
                response = _request(
                    "POST",
                    f"{args.api_url.rstrip('/')}/predict",
                    body=b"--broken",
                    headers={
                        "Content-Type": "multipart/form-data; boundary=birdly-broken"
                    },
                )
                return _status_result("malformed_multipart", 400, response)

            _run_named_test(results, "malformed_multipart", malformed_request)
            _run_named_test(
                results,
                "concurrent_valid_requests",
                lambda: _test_concurrent_valid(
                    args.api_url,
                    valid_image,
                    args.concurrent_requests,
                    args.request_timeout_seconds,
                ),
            )
            _run_named_test(
                results,
                "mixed_concurrent_requests",
                lambda: _test_mixed_concurrent(
                    args.api_url,
                    valid_image,
                    args.concurrent_requests,
                    args.request_timeout_seconds,
                ),
            )
            _run_named_test(
                results,
                "safe_burst_traffic",
                lambda: _test_burst(
                    args.api_url,
                    valid_image,
                    args.burst_requests,
                    args.request_timeout_seconds,
                ),
            )

            if mode == "remote":
                remote_skip_reason = (
                    "requires control of the underlying container; skipped in "
                    "remote mode"
                )
                for name in (
                    "rate_limit_overflow",
                    "rate_limit_retry_after",
                    "inference_queue_timeout",
                    "missing_model_health",
                ):
                    _skip(
                        results,
                        name,
                        remote_skip_reason,
                        expected="local container control",
                        actual="remote mode",
                    )
            else:
                if shutil.which("docker") is None:
                    raise RuntimeError(
                        "Docker is required for container condition tests"
                    )
                image_name, model_source = _inspect_container(args.container)
                metadata["container_image"] = image_name
                metadata["model_mount_source"] = model_source
                _run_rate_limit_tests(
                    results,
                    lambda: _test_rate_limit(
                        image=valid_image,
                        image_name=image_name,
                        model_source=model_source,
                        startup_timeout_seconds=args.startup_timeout_seconds,
                    ),
                )
                _run_named_test(
                    results,
                    "inference_queue_timeout",
                    lambda: _test_queue_timeout(
                        image=valid_image,
                        image_name=image_name,
                        model_source=model_source,
                        startup_timeout_seconds=args.startup_timeout_seconds,
                    ),
                )
                if args.docker_missing_model_test:
                    _run_named_test(
                        results,
                        "missing_model_health",
                        lambda: _test_missing_model(
                            image_name=image_name,
                            startup_timeout_seconds=args.startup_timeout_seconds,
                        ),
                    )
                else:
                    _skip(
                        results,
                        "missing_model_health",
                        "use --docker-missing-model-test",
                    )
            after_text, after_metrics = _fetch_metrics(args.api_url)
            (results_directory / "api-metrics-after.prom").write_text(
                after_text, encoding="utf-8"
            )
            _write_json(results_directory / "api-metrics-after.json", after_metrics)
            _write_json(
                results_directory / "api-metrics-delta.json",
                _metric_delta(before_metrics, after_metrics),
            )
            _run_named_test(
                results,
                "prometheus_failure_counters",
                lambda: _test_metrics_increase(before_metrics, after_metrics),
            )
            if mode == "remote":
                _skip(
                    results,
                    "container_restart_recovery",
                    "requires control of the underlying container; skipped in "
                    "remote mode",
                    expected="local container control",
                    actual="remote mode",
                )
            elif args.docker_restart_tests:
                _run_named_test(
                    results,
                    "container_restart_recovery",
                    lambda: _test_restart(
                        container_name=args.container,
                        api_url=args.api_url,
                        image=valid_image,
                        startup_timeout_seconds=args.startup_timeout_seconds,
                    ),
                )
            else:
                _skip(
                    results,
                    "container_restart_recovery",
                    "use --docker-restart-tests",
                )
            _run_named_test(
                results,
                "active_inference_returns_to_zero",
                lambda: _test_active_inference_zero(args.api_url),
            )
        elif mode != "remote-rate-limit":
            after_text, after_metrics = _fetch_metrics(args.api_url)
            (results_directory / "api-metrics-after.prom").write_text(
                after_text, encoding="utf-8"
            )
            _write_json(results_directory / "api-metrics-after.json", after_metrics)
            _write_json(
                results_directory / "api-metrics-delta.json",
                _metric_delta(before_metrics, after_metrics),
            )

        final_text, final_metrics = _fetch_metrics(args.api_url)
        (results_directory / "api-metrics-final.prom").write_text(
            final_text, encoding="utf-8"
        )
        _write_json(results_directory / "api-metrics-final.json", final_metrics)
        _write_json(results_directory / "configuration.json", metadata)
        _write_json(
            results_directory / "results.json", [asdict(result) for result in results]
        )
        _write_results_csv(results_directory / "results.csv", results)

        passed = sum(result.result == "PASS" for result in results)
        failed = [result.name for result in results if result.result == "FAIL"]
        skipped = sum(result.result == "SKIP" for result in results)
        skipped_tests = [result.name for result in results if result.result == "SKIP"]
        final_summary = {
            "artifact_directory": str(results_directory),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "failed_tests": failed,
            "mode": mode,
            "passed": passed,
            "skipped": skipped,
            "skipped_tests": skipped_tests,
            "status": "failed" if failed else "passed",
            "total": len(results),
        }
        _write_json(results_directory / "final-summary.json", final_summary)
        _print_table(results)
        print(f"\nPassed: {passed}/{len(results)}")
        print(f"Failed tests: {', '.join(failed) if failed else 'none'}")
        print(f"Skipped: {skipped}")
        print(f"Artifacts: {results_directory}")
        return 1 if failed else 0
    except Exception as error:  # noqa: BLE001
        _write_json(
            results_directory / "results.json",
            [asdict(result) for result in results],
        )
        _write_json(
            results_directory / "final-summary.json",
            {
                "error": f"{type(error).__name__}: {error}",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "mode": mode,
                "status": "failed",
            },
        )
        print(f"Reliability tests failed: {error}", file=sys.stderr)
        print(f"Partial artifacts: {results_directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
