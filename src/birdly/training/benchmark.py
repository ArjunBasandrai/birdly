from __future__ import annotations

import gc
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from birdly.data.dataset import BirdlyDataset
from birdly.data.utils import load_config
from birdly.training.config import load_local_paths, load_training_config
from birdly.training.io_utils import atomic_write_json, current_git_commit
from birdly.training.model import build_model


def _measure_candidate(
    *,
    batch_size: int,
    image_size: int,
    config: dict[str, Any],
    manifest_path: Path,
    image_root: Path,
    num_classes: int,
) -> dict[str, Any]:
    dataset = BirdlyDataset(
        "train",
        manifest_path,
        image_root,
        image_size=image_size,
        augmentation=config["augmentation"],
        max_samples_per_class=6,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["benchmark"]["num_workers"]),
        pin_memory=True,
        generator=torch.Generator().manual_seed(42),
    )
    model = build_model(
        num_classes,
        model_name=str(config["model"]["name"]),
        pretrained=bool(config["model"]["pretrained"]),
    ).cuda()
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    iterator = iter(loader)
    warmup_steps = int(config["benchmark"]["warmup_steps"])
    measured_steps = int(config["benchmark"]["measured_steps"])
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup_steps):
        images, targets = next(iterator)
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(measured_steps):
        images, targets = next(iterator)
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "image_size": image_size,
        "warmup_steps": warmup_steps,
        "steps": measured_steps,
        "seconds": seconds,
        "images_per_second": batch_size * measured_steps / seconds,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "loss": float(loss.item()),
    }


def run_benchmark(output_path: Path = Path("reports/research/benchmark.json")) -> dict[str, Any]:
    data_config, _, _ = load_config()
    local_paths = load_local_paths()
    local_paths["training_runs_root"].mkdir(parents=True, exist_ok=True)
    config = load_training_config()
    validation_report = json.loads(
        Path("data/processed/validation_report.json").read_text(encoding="utf-8")
    )
    manifest_path = Path(
        config.get("manifest_path", "data/processed/validated_manifest.parquet")
    )
    device_properties = torch.cuda.get_device_properties(0)
    maximum_bytes = int(
        device_properties.total_memory
        * float(config["benchmark"]["maximum_vram_fraction"])
    )
    candidates = [int(value) for value in config["benchmark"]["candidate_batch_sizes"]]
    image_sizes = [int(value) for value in config["benchmark"]["candidate_image_sizes"]]
    results = []
    safe_by_resolution = []
    oom_count = 0
    for image_size in image_sizes:
        resolution_oom_count = 0
        for batch_size in candidates:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                measurement = _measure_candidate(
                    batch_size=batch_size,
                    image_size=image_size,
                    config=config,
                    manifest_path=manifest_path,
                    image_root=local_paths["image_data_root"],
                    num_classes=int(validation_report["class_count"]),
                )
                measurement["fits_memory_limit"] = (
                    measurement["peak_memory_mib"] * 1024**2 <= maximum_bytes
                )
                measurement["projected_epoch_minutes"] = (
                    validation_report["split_counts"]["train"]
                    / measurement["images_per_second"]
                    / 60
                )
                results.append(measurement)
                if measurement["fits_memory_limit"]:
                    safe_by_resolution.append(measurement)
                    break
            except torch.OutOfMemoryError as error:
                oom_count += 1
                resolution_oom_count += 1
                results.append(
                    {
                        "batch_size": batch_size,
                        "image_size": image_size,
                        "status": "oom",
                        "error": str(error).splitlines()[0],
                    }
                )
                torch.cuda.empty_cache()
                if resolution_oom_count >= 2:
                    break

    if not safe_by_resolution:
        raise RuntimeError("No benchmark setting met the VRAM limit.")
    target_epoch_minutes = float(config["benchmark"]["target_epoch_minutes"])
    eligible = [
        row
        for row in safe_by_resolution
        if row["projected_epoch_minutes"] <= target_epoch_minutes
    ]
    selected = (
        max(eligible, key=lambda row: row["image_size"])
        if eligible
        else max(safe_by_resolution, key=lambda row: row["images_per_second"])
    )
    target_effective_batch = int(config["training"]["target_effective_batch_size"])
    result = {
        "git_commit": current_git_commit(),
        "data_version": validation_report["validated_manifest"]["sha256"],
        "manifest_path": data_config["processed"].get(
            "validated_manifest", "data/processed/validated_manifest.parquet"
        ),
        "system": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "vram_mib": device_properties.total_memory / (1024**2),
            "ram_gib": psutil.virtual_memory().total / (1024**3),
            "run_disk_free_gib": shutil.disk_usage(local_paths["training_runs_root"]).free
            / (1024**3),
        },
        "measurements": results,
        "selection_policy": (
            "highest resolution at or below the projected epoch limit; "
            "otherwise highest throughput"
        ),
        "target_epoch_minutes": target_epoch_minutes,
        "selected_image_size": int(selected["image_size"]),
        "selected_batch_size": int(selected["batch_size"]),
        "selected_accumulation_steps": max(
            1, math.ceil(target_effective_batch / int(selected["batch_size"]))
        ),
        "oom_count": oom_count,
    }
    atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run_benchmark()
