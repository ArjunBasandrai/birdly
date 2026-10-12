from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from smoke_test import upload


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def process_rss_kib() -> int | None:
    try:
        pids = [int(item) for item in os.listdir("/proc") if item.isdigit()]
        for pid in pids:
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes()
                if b"uvicorn main:app" in command:
                    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
                    for line in status.splitlines():
                        if line.startswith("VmHWM:"):
                            return int(line.split()[1])
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return None


def cpu_ticks() -> tuple[int, int] | None:
    try:
        fields = (
            Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        )
        ticks = [int(field) for field in fields]
        return sum(ticks), ticks[3] + ticks[4]
    except (OSError, ValueError, IndexError):
        return None


def device_stats() -> dict[str, object]:
    result: dict[str, object] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                result[line.split(":")[0] + "_kib"] = int(line.split()[1])
        result["load_average"] = os.getloadavg()
        thermal = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal.exists():
            result["cpu_temperature_c"] = int(thermal.read_text().strip()) / 1000
    except (OSError, ValueError):
        pass
    result["process_peak_rss_kib"] = process_rss_kib()
    result["cpu_ticks"] = cpu_ticks()
    try:
        result["throttling_status"] = subprocess.check_output(
            ["vcgencmd", "get_throttled"], text=True, timeout=2
        ).strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        result["throttling_status"] = None
    return result


def one_request(url: str, image: bytes) -> tuple[int, float]:
    start = time.perf_counter()
    try:
        status, _ = upload(url, image)
    except (OSError, urllib.error.URLError):
        status = 0
    return status, (time.perf_counter() - start) * 1000


def run(
    url: str, image: bytes, requests: int, warmup: int, concurrency: int
) -> dict[str, object]:
    for _ in range(warmup):
        one_request(url, image)
    before = device_stats()
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        observations = list(
            pool.map(lambda _: one_request(url, image), range(requests))
        )
    elapsed = time.perf_counter() - start
    after = device_stats()
    start_ticks, end_ticks = before.get("cpu_ticks"), after.get("cpu_ticks")
    cpu_percent = None
    if start_ticks and end_ticks and end_ticks[0] > start_ticks[0]:
        cpu_percent = 100 * (
            1 - (end_ticks[1] - start_ticks[1]) / (end_ticks[0] - start_ticks[0])
        )
    latencies = [latency for _, latency in observations]
    successes = sum(status == 200 for status, _ in observations)
    return {
        "mode": "request",
        "requests": requests,
        "warmup": warmup,
        "concurrency": concurrency,
        "successes": successes,
        "failures": requests - successes,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
        "throughput_requests_per_second": requests / elapsed,
        "duration_seconds": elapsed,
        "before": before,
        "after": after,
        "device_cpu_utilization_percent": cpu_percent,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0 or args.concurrency < 1:
        parser.error(
            "requests and concurrency must be positive; warmup must not be negative"
        )
    result = run(
        args.url, args.image.read_bytes(), args.requests, args.warmup, args.concurrency
    )
    print(json.dumps(result, indent=2))
