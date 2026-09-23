from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import mlflow
import polars as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from birdly.data.dataset import BirdlyDataset
from birdly.training.io_utils import (
    atomic_torch_save,
    atomic_write_json,
    capture_rng_state,
    config_hash,
    restore_rng_state,
    set_reproducible_seed,
)
from birdly.training.metrics import ClassificationMetrics
from birdly.training.model import build_model
from birdly.training.resources import ResourceGuard, SafetyLimit

_REQUIRED_CHECKPOINT_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
    "config",
    "config_sha256",
    "taxon_to_class",
    "class_to_taxon",
    "preprocessing",
    "data_version",
    "git_commit",
    "run_id",
    "mlflow_run_id",
    "epoch",
    "next_batch_index",
    "best_macro_f1",
    "best_epoch",
    "epochs_without_improvement",
    "elapsed_seconds",
    "metric_history",
}


def _load_completed_summary(
    *,
    summary_path: Path,
    latest_checkpoint: Path,
    config: dict[str, Any],
    git_commit: str,
    run_id: str,
) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in {"completed", "pruned"}:
        return None
    if not latest_checkpoint.exists():
        raise FileNotFoundError(
            f"Completed run {run_id} has no latest checkpoint."
        )
    checkpoint = torch.load(
        latest_checkpoint, map_location="cpu", weights_only=False
    )
    missing_keys = sorted(_REQUIRED_CHECKPOINT_KEYS - checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Resume checkpoint is missing keys: {missing_keys}")
    if checkpoint["config_sha256"] != config_hash(config):
        raise ValueError("Resume checkpoint configuration does not match the run.")
    if checkpoint["git_commit"] != git_commit:
        raise ValueError("Resume checkpoint commit does not match the run.")
    if checkpoint["run_id"] != run_id:
        raise ValueError("Resume checkpoint run ID does not match the run.")
    return summary


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    import random

    import numpy as np

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader(
    dataset: BirdlyDataset,
    config: dict[str, Any],
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    workers = int(config["training"]["num_workers"])
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["training"]["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        "generator": torch.Generator().manual_seed(seed),
        "persistent_workers": False,
    }
    if workers > 0:
        options["prefetch_factor"] = int(config["training"]["prefetch_factor"])
    return DataLoader(**options)


def _lr_schedule(
    warmup_updates: int, total_updates: int, minimum_ratio: float
):
    def schedule(update_index: int) -> float:
        if warmup_updates > 0 and update_index < warmup_updates:
            return max((update_index + 1) / warmup_updates, 1e-8)
        progress = (update_index - warmup_updates) / max(
            total_updates - warmup_updates, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return schedule


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    taxon_to_class: dict[int, int],
    data_version: dict[str, Any],
    git_commit: str,
    run_id: str,
    mlflow_run_id: str,
    epoch: int,
    next_batch_index: int,
    best_macro_f1: float,
    best_epoch: int,
    epochs_without_improvement: int,
    elapsed_seconds: float,
    metric_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "config": config,
        "config_sha256": config_hash(config),
        "taxon_to_class": taxon_to_class,
        "class_to_taxon": {value: key for key, value in taxon_to_class.items()},
        "preprocessing": {
            "image_size": int(config["image"]["size"]),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "data_version": data_version,
        "git_commit": git_commit,
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "best_macro_f1": best_macro_f1,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "elapsed_seconds": elapsed_seconds,
        "metric_history": metric_history,
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_to_taxon: dict[int, int],
    *,
    amp_enabled: bool,
    label_smoothing: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing, reduction="sum"
    )
    metrics = ClassificationMetrics(len(class_to_taxon), calibration_bins=15)
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss_sum = criterion(logits, targets)
            metrics.update(logits, targets, float(loss_sum.item()))
    return metrics.compute(class_to_taxon)


def train_run(
    *,
    run_id: str,
    config: dict[str, Any],
    run_dir: Path,
    manifest_path: Path,
    image_root: Path,
    data_version: dict[str, Any],
    git_commit: str,
    tracking_uri: str,
    parent_macro_f1: float | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = run_dir / "latest.pt"
    best_checkpoint = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics_history.json"

    completed_summary = _load_completed_summary(
        summary_path=summary_path,
        latest_checkpoint=latest_checkpoint,
        config=config,
        git_commit=git_commit,
        run_id=run_id,
    )
    if completed_summary is not None:
        return completed_summary

    seed = int(config["training"]["seed"])
    deterministic = bool(config["training"]["deterministic"])
    set_reproducible_seed(seed, deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the local research cycle.")

    manifest = pl.read_parquet(manifest_path, columns=["taxon_id"])
    taxon_ids = sorted(int(value) for value in manifest["taxon_id"].unique())
    taxon_to_class = {taxon_id: index for index, taxon_id in enumerate(taxon_ids)}
    class_to_taxon = {index: taxon_id for taxon_id, index in taxon_to_class.items()}
    image_size = int(config["image"]["size"])
    smoke_limit = config["training"].get("max_samples_per_class")

    train_dataset = BirdlyDataset(
        "train",
        manifest_path,
        image_root,
        image_size=image_size,
        taxon_to_class=taxon_to_class,
        augmentation=config["augmentation"],
        max_samples_per_class=(int(smoke_limit) if smoke_limit else None),
    )
    val_dataset = BirdlyDataset(
        "val",
        manifest_path,
        image_root,
        image_size=image_size,
        taxon_to_class=taxon_to_class,
        max_samples_per_class=(int(smoke_limit) if smoke_limit else None),
    )
    validation_loader = _loader(
        val_dataset, config, shuffle=False, seed=seed + 100_000
    )

    model = build_model(
        len(taxon_to_class),
        model_name=str(config["model"]["name"]),
        pretrained=(
            bool(config["model"]["pretrained"])
            if not latest_checkpoint.exists()
            else False
        ),
    ).to(device)
    initial_parameter = next(model.parameters()).detach().cpu().clone()
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"]["label_smoothing"])
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    accumulation_steps = int(config["training"]["accumulation_steps"])
    estimated_batches = math.ceil(
        len(train_dataset) / int(config["training"]["batch_size"])
    )
    updates_per_epoch = math.ceil(estimated_batches / accumulation_steps)
    total_epochs = int(config["training"]["epochs"])
    total_updates = total_epochs * updates_per_epoch
    warmup_updates = int(config["training"]["warmup_epochs"] * updates_per_epoch)
    scheduler = LambdaLR(
        optimizer,
        _lr_schedule(
            warmup_updates,
            total_updates,
            float(config["training"]["minimum_lr_ratio"]),
        ),
    )
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 1
    start_batch_index = 0
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    elapsed_before_resume = 0.0
    metric_history: list[dict[str, Any]] = []
    mlflow_run_id = ""
    if latest_checkpoint.exists():
        checkpoint = torch.load(
            latest_checkpoint, map_location="cpu", weights_only=False
        )
        if checkpoint["config_sha256"] != config_hash(config):
            raise ValueError("Resume checkpoint configuration does not match the run.")
        if checkpoint["git_commit"] != git_commit:
            raise ValueError("Resume checkpoint commit does not match the run.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch_index = int(checkpoint["next_batch_index"])
        best_macro_f1 = float(checkpoint["best_macro_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        elapsed_before_resume = float(checkpoint["elapsed_seconds"])
        metric_history = list(checkpoint["metric_history"])
        mlflow_run_id = str(checkpoint["mlflow_run_id"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("birdly-local-research")
    mlflow_context = (
        mlflow.start_run(run_id=mlflow_run_id)
        if mlflow_run_id
        else mlflow.start_run(run_name=run_id)
    )
    wall_start = time.monotonic()
    guard = ResourceGuard(
        run_dir=run_dir,
        disk_path=run_dir,
        minimum_free_ram_gb=float(config["safety"]["minimum_free_ram_gb"]),
        minimum_free_disk_gb=float(config["safety"]["minimum_free_disk_gb"]),
        maximum_gpu_temperature_c=int(
            config["safety"]["maximum_gpu_temperature_c"]
        ),
        poll_seconds=float(config["safety"]["poll_seconds"]),
    )
    torch.cuda.reset_peak_memory_stats()

    with mlflow_context as active_run:
        mlflow_run_id = active_run.info.run_id
        mlflow.log_params(
            {
                "run_id": run_id,
                "git_commit": git_commit,
                "model": config["model"]["name"],
                "image_size": image_size,
                "batch_size": config["training"]["batch_size"],
                "accumulation_steps": accumulation_steps,
                "learning_rate": config["training"]["learning_rate"],
                "weight_decay": config["training"]["weight_decay"],
                "seed": seed,
                "data_sha256": data_version["validated_manifest"]["sha256"],
            }
        )
        guard.check(force=True)
        status = "completed"
        stop_reason = "epoch_limit"
        first_batch_loss: float | None = None
        last_batch_loss: float | None = None
        try:
            for epoch in range(start_epoch, total_epochs + 1):
                epoch_start = time.monotonic()
                training_loader = _loader(
                    train_dataset,
                    config,
                    shuffle=True,
                    seed=seed + epoch,
                )
                model.train()
                optimizer.zero_grad(set_to_none=True)
                epoch_loss_sum = 0.0
                epoch_samples = 0
                update_count = 0

                for batch_index, (images, targets) in enumerate(training_loader):
                    if epoch == start_epoch and batch_index < start_batch_index:
                        continue
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        logits = model(images)
                        loss = criterion(logits, targets)
                        scaled_loss = loss / accumulation_steps
                    scaler.scale(scaled_loss).backward()
                    if first_batch_loss is None:
                        first_batch_loss = float(loss.item())
                    last_batch_loss = float(loss.item())
                    epoch_loss_sum += float(loss.item()) * images.size(0)
                    epoch_samples += images.size(0)

                    is_update = (
                        (batch_index + 1) % accumulation_steps == 0
                        or batch_index + 1 == len(training_loader)
                    )
                    if not is_update:
                        continue
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["training"]["gradient_clip_norm"]),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    update_count += 1

                    elapsed = elapsed_before_resume + time.monotonic() - wall_start
                    payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        taxon_to_class=taxon_to_class,
                        data_version=data_version,
                        git_commit=git_commit,
                        run_id=run_id,
                        mlflow_run_id=mlflow_run_id,
                        epoch=epoch,
                        next_batch_index=batch_index + 1,
                        best_macro_f1=best_macro_f1,
                        best_epoch=best_epoch,
                        epochs_without_improvement=epochs_without_improvement,
                        elapsed_seconds=elapsed,
                        metric_history=metric_history,
                    )
                    if update_count % int(
                        config["training"]["checkpoint_interval_updates"]
                    ) == 0:
                        atomic_torch_save(latest_checkpoint, payload)
                    try:
                        guard.check()
                    except SafetyLimit:
                        atomic_torch_save(latest_checkpoint, payload)
                        raise
                    if elapsed / 60 > float(config["training"]["max_minutes"]):
                        atomic_torch_save(latest_checkpoint, payload)
                        status = "pruned"
                        stop_reason = "time_limit"
                        break

                if status == "pruned":
                    break

                train_loss = epoch_loss_sum / max(epoch_samples, 1)
                validation_metrics = evaluate_loader(
                    model,
                    validation_loader,
                    device,
                    class_to_taxon,
                    amp_enabled=amp_enabled,
                )
                epoch_seconds = time.monotonic() - epoch_start
                scalar_metrics = {
                    key: value
                    for key, value in validation_metrics.items()
                    if key != "per_class"
                }
                epoch_record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": epoch_seconds,
                    "training_throughput_images_per_second": epoch_samples
                    / max(epoch_seconds, 1e-9),
                    "validation": validation_metrics,
                }
                metric_history.append(epoch_record)
                atomic_write_json(metrics_path, metric_history)
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        **{f"val_{key}": value for key, value in scalar_metrics.items()},
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch_seconds": epoch_seconds,
                    },
                    step=epoch,
                )

                macro_f1 = float(validation_metrics["macro_f1"])
                improved = macro_f1 > best_macro_f1
                if improved:
                    best_macro_f1 = macro_f1
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                elapsed = elapsed_before_resume + time.monotonic() - wall_start
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    taxon_to_class=taxon_to_class,
                    data_version=data_version,
                    git_commit=git_commit,
                    run_id=run_id,
                    mlflow_run_id=mlflow_run_id,
                    epoch=epoch + 1,
                    next_batch_index=0,
                    best_macro_f1=best_macro_f1,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                    elapsed_seconds=elapsed,
                    metric_history=metric_history,
                )
                atomic_torch_save(latest_checkpoint, payload)
                if improved:
                    atomic_torch_save(best_checkpoint, payload)
                    atomic_write_json(
                        run_dir / "best_validation_metrics.json", validation_metrics
                    )

                print(
                    f"{run_id} epoch={epoch} train_loss={train_loss:.4f} "
                    f"val_macro_f1={macro_f1:.4f} "
                    f"val_top1={validation_metrics['top1_accuracy']:.4f} "
                    f"seconds={epoch_seconds:.1f}",
                    flush=True,
                )

                minimum_epochs = int(config["training"]["minimum_epochs"])
                if (
                    parent_macro_f1 is not None
                    and epoch >= minimum_epochs
                    and macro_f1
                    < parent_macro_f1 - float(config["training"]["prune_margin"])
                ):
                    status = "pruned"
                    stop_reason = "below_parent"
                    break
                if (
                    epoch >= minimum_epochs
                    and epochs_without_improvement
                    >= int(config["training"]["early_stopping_patience"])
                ):
                    stop_reason = "early_stopping"
                    break
        except SafetyLimit as error:
            status = "paused"
            stop_reason = f"safety_limit: {error}"
            raise

        elapsed_seconds = elapsed_before_resume + time.monotonic() - wall_start
        best_metrics = (
            next(
                row["validation"]
                for row in metric_history
                if row["epoch"] == best_epoch
            )
            if best_epoch
            else {}
        )
        summary = {
            "run_id": run_id,
            "status": status,
            "stop_reason": stop_reason,
            "git_commit": git_commit,
            "config_sha256": config_hash(config),
            "data_version": data_version,
            "best_epoch": best_epoch,
            "best_validation": best_metrics,
            "epochs_completed": len(metric_history),
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "first_batch_loss": first_batch_loss,
            "last_batch_loss": last_batch_loss,
            "parameter_delta_norm": float(
                (
                    next(model.parameters()).detach().cpu() - initial_parameter
                ).norm().item()
            ),
            "resource_summary": guard.summary(),
            "latest_checkpoint": str(latest_checkpoint),
            "best_checkpoint": str(best_checkpoint),
            "mlflow_run_id": mlflow_run_id,
        }
        summary["learning_verified"] = bool(
            summary["parameter_delta_norm"] > 0
            and first_batch_loss is not None
            and last_batch_loss is not None
            and math.isfinite(first_batch_loss)
            and math.isfinite(last_batch_loss)
        )
        atomic_write_json(summary_path, summary)
        mlflow.log_metrics(
            {
                "best_val_macro_f1": best_macro_f1,
                "elapsed_seconds": elapsed_seconds,
                "peak_gpu_memory_mib": summary["peak_gpu_memory_mib"],
            }
        )
        return summary
