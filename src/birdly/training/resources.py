from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import torch

from birdly.training.io_utils import append_jsonl


class SafetyLimit(RuntimeError):
    pass


@dataclass
class ResourceGuard:
    run_dir: Path
    disk_path: Path
    minimum_free_ram_gb: float
    minimum_free_disk_gb: float
    maximum_gpu_temperature_c: int
    poll_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.last_check = 0.0
        self.telemetry_path = self.run_dir / "resource_telemetry.jsonl"

    def _gpu_snapshot(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {
                "temperature_c": None,
                "utilization_percent": None,
                "memory_used_mib": None,
                "power_w": None,
            }
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        return {
            "temperature_c": float(fields[0]),
            "utilization_percent": float(fields[1]),
            "memory_used_mib": float(fields[2]),
            "power_w": None if fields[3] == "[N/A]" else float(fields[3]),
        }

    def check(self, *, force: bool = False) -> dict[str, Any] | None:
        now = time.monotonic()
        if not force and now - self.last_check < self.poll_seconds:
            return None
        self.last_check = now

        memory = psutil.virtual_memory()
        disk = shutil.disk_usage(self.disk_path)
        gpu = self._gpu_snapshot()
        snapshot = {
            "timestamp_unix": time.time(),
            "free_ram_gb": memory.available / (1024**3),
            "free_disk_gb": disk.free / (1024**3),
            "gpu": gpu,
        }
        append_jsonl(self.telemetry_path, snapshot)

        failures = []
        if snapshot["free_ram_gb"] < self.minimum_free_ram_gb:
            failures.append(f"free RAM {snapshot['free_ram_gb']:.2f} GiB")
        if snapshot["free_disk_gb"] < self.minimum_free_disk_gb:
            failures.append(f"free disk {snapshot['free_disk_gb']:.2f} GiB")
        temperature = gpu["temperature_c"]
        if temperature is not None and temperature >= self.maximum_gpu_temperature_c:
            failures.append(f"GPU temperature {temperature:.0f} C")
        if failures:
            raise SafetyLimit("; ".join(failures))
        return snapshot

    def summary(self) -> dict[str, Any]:
        if not self.telemetry_path.exists():
            return {}
        snapshots = [
            json.loads(line)
            for line in self.telemetry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        temperatures = [
            row["gpu"]["temperature_c"]
            for row in snapshots
            if row["gpu"]["temperature_c"] is not None
        ]
        return {
            "minimum_free_ram_gb": min(
                (row["free_ram_gb"] for row in snapshots), default=None
            ),
            "minimum_free_disk_gb": min(
                (row["free_disk_gb"] for row in snapshots), default=None
            ),
            "maximum_gpu_temperature_c": max(temperatures, default=None),
            "samples": len(snapshots),
        }
