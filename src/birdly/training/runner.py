from __future__ import annotations

import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from birdly.training.config import (
    deep_merge,
    load_local_paths,
    load_training_config,
    load_yaml,
)
from birdly.training.evaluate import evaluate_checkpoint
from birdly.training.io_utils import (
    append_jsonl,
    atomic_write_json,
    atomic_write_yaml,
    config_hash,
    current_git_commit,
    sqlite_tracking_uri,
)
from birdly.training.resources import SafetyLimit
from birdly.training.train import train_run


def _best_summary(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        summary
        for summary in summaries.values()
        if summary.get("best_validation", {}).get("macro_f1") is not None
        and summary.get("status") in {"completed", "pruned"}
    ]
    if not candidates:
        raise RuntimeError("No completed validation result is available.")
    return max(candidates, key=lambda row: row["best_validation"]["macro_f1"])


def _run_with_retries(
    *,
    logical_run_id: str,
    reason: str,
    parent_id: str | None,
    config: dict[str, Any],
    cycle_root: Path,
    ledger_path: Path,
    manifest_path: Path,
    image_root: Path,
    data_version: dict[str, Any],
    git_commit: str,
    tracking_uri: str,
    parent_macro_f1: float | None,
    runner_config: dict[str, Any],
) -> dict[str, Any]:
    target_effective_batch = int(config["training"]["target_effective_batch_size"])
    maximum_oom_retries = int(runner_config["maximum_oom_retries"])
    maximum_safety_pauses = int(runner_config["maximum_safety_pauses"])
    safety_pauses = 0

    for attempt in range(maximum_oom_retries + 1):
        run_id = logical_run_id if attempt == 0 else f"{logical_run_id}-retry{attempt}"
        run_config = deep_merge(config, {})
        if attempt:
            original_batch = int(config["training"]["batch_size"])
            batch_size = max(1, original_batch // (2**attempt))
            run_config["training"]["batch_size"] = batch_size
            run_config["training"]["accumulation_steps"] = max(
                1, math.ceil(target_effective_batch / batch_size)
            )
        run_dir = cycle_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        frozen_config_path = run_dir / "config.yaml"
        if frozen_config_path.exists():
            existing_config = load_yaml(frozen_config_path)
            if config_hash(existing_config) != config_hash(run_config):
                raise ValueError(f"Frozen configuration mismatch for {run_id}.")
        else:
            atomic_write_yaml(frozen_config_path, run_config)
        trial_record = {
            "event": "trial_planned",
            "logical_run_id": logical_run_id,
            "run_id": run_id,
            "attempt": attempt,
            "parent_run_id": parent_id,
            "reason": reason,
            "git_commit": git_commit,
            "data_version": data_version["validated_manifest"]["sha256"],
            "config_sha256": config_hash(run_config),
            "timestamp_unix": time.time(),
        }
        atomic_write_json(run_dir / "trial_record.json", trial_record)
        append_jsonl(ledger_path, trial_record)

        existing_summary_path = run_dir / "summary.json"
        if existing_summary_path.exists():
            existing_summary = json.loads(
                existing_summary_path.read_text(encoding="utf-8")
            )
            if existing_summary.get("status") in {"completed", "pruned"}:
                return existing_summary

        while True:
            try:
                summary = train_run(
                    run_id=run_id,
                    config=run_config,
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    image_root=image_root,
                    data_version=data_version,
                    git_commit=git_commit,
                    tracking_uri=tracking_uri,
                    parent_macro_f1=parent_macro_f1,
                )
                summary["logical_run_id"] = logical_run_id
                summary["parent_run_id"] = parent_id
                summary["reason"] = reason
                atomic_write_json(existing_summary_path, summary)
                append_jsonl(
                    ledger_path,
                    {
                        "event": "trial_finished",
                        "timestamp_unix": time.time(),
                        **summary,
                    },
                )
                return summary
            except SafetyLimit as error:
                safety_pauses += 1
                append_jsonl(
                    ledger_path,
                    {
                        "event": "safety_pause",
                        "run_id": run_id,
                        "reason": str(error),
                        "pause": safety_pauses,
                        "timestamp_unix": time.time(),
                    },
                )
                if safety_pauses > maximum_safety_pauses:
                    raise RuntimeError("Maximum safety pauses reached.") from error
                time.sleep(float(runner_config["safety_pause_seconds"]))
            except torch.OutOfMemoryError as error:
                append_jsonl(
                    ledger_path,
                    {
                        "event": "trial_failed",
                        "run_id": run_id,
                        "failure": "out_of_memory",
                        "error": str(error).splitlines()[0],
                        "timestamp_unix": time.time(),
                    },
                )
                torch.cuda.empty_cache()
                break
            except Exception as error:
                append_jsonl(
                    ledger_path,
                    {
                        "event": "trial_failed",
                        "run_id": run_id,
                        "failure": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "timestamp_unix": time.time(),
                    },
                )
                raise

    raise RuntimeError(f"OOM retry limit reached for {logical_run_id}.")


def _uncertainty(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    critical_values = {2: 12.706, 3: 4.303}
    critical = critical_values.get(len(values), 1.96)
    half_width = (
        critical * standard_deviation / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return {
        "values": values,
        "n": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "confidence_interval_95": [mean - half_width, mean + half_width],
        "method": "two-sided t interval" if len(values) <= 3 else "normal interval",
    }


def run_cycle() -> dict[str, Any]:
    base_config = load_training_config()
    plan = load_yaml("configs/research_plan.yaml")
    local_paths = load_local_paths()
    local_paths["training_runs_root"].mkdir(parents=True, exist_ok=True)
    local_paths["torch_cache_root"].mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(local_paths["torch_cache_root"])
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    validation_report = json.loads(
        Path("data/processed/validation_report.json").read_text(encoding="utf-8")
    )
    benchmark = json.loads(
        Path("reports/research/benchmark.json").read_text(encoding="utf-8")
    )
    git_commit = current_git_commit()
    if (
        benchmark["data_version"]
        != validation_report["validated_manifest"]["sha256"]
    ):
        raise ValueError("Benchmark data version does not match the frozen manifest.")

    base_config["training"]["batch_size"] = int(benchmark["selected_batch_size"])
    base_config["training"]["accumulation_steps"] = int(
        benchmark["selected_accumulation_steps"]
    )
    base_config["image"]["size"] = int(benchmark["selected_image_size"])
    runner_config = base_config["runner"]
    cycle_id = str(runner_config["cycle_id"])
    cycle_root = local_paths["training_runs_root"] / cycle_id
    cycle_root.mkdir(parents=True, exist_ok=True)
    ledger_path = cycle_root / "experiment_ledger.jsonl"
    report_dir = Path("reports/research")
    report_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = sqlite_tracking_uri(cycle_root / "mlflow.db")
    manifest_path = Path("data/processed/validated_manifest.parquet")
    cycle_start = time.monotonic()

    smoke_config = deep_merge(
        base_config,
        {
            "training": {
                "epochs": 1,
                "minimum_epochs": 1,
                "early_stopping_patience": 1,
                "max_minutes": 30,
                "max_samples_per_class": int(
                    runner_config["smoke_samples_per_class"]
                ),
                "num_workers": 0,
            }
        },
    )
    smoke = _run_with_retries(
        logical_run_id="smoke-one-epoch",
        reason="Verify one training epoch, validation, checkpoint save, and reload.",
        parent_id=None,
        config=smoke_config,
        cycle_root=cycle_root,
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        image_root=local_paths["image_data_root"],
        data_version=validation_report,
        git_commit=git_commit,
        tracking_uri=tracking_uri,
        parent_macro_f1=None,
        runner_config=runner_config,
    )
    smoke_resume = train_run(
        run_id=smoke["run_id"],
        config=smoke_config,
        run_dir=cycle_root / "runs" / smoke["run_id"],
        manifest_path=manifest_path,
        image_root=local_paths["image_data_root"],
        data_version=validation_report,
        git_commit=git_commit,
        tracking_uri=tracking_uri,
    )
    if smoke_resume["best_epoch"] != smoke["best_epoch"]:
        raise RuntimeError("Smoke checkpoint resume verification failed.")
    if not smoke.get("learning_verified"):
        raise RuntimeError("Smoke training did not change model parameters.")
    if not Path(smoke["best_checkpoint"]).exists():
        raise RuntimeError("Smoke best checkpoint is missing.")
    append_jsonl(
        ledger_path,
        {
            "event": "smoke_resume_verified",
            "run_id": smoke["run_id"],
            "timestamp_unix": time.time(),
        },
    )

    summaries: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    baseline_spec = plan["baseline"]
    baseline = _run_with_retries(
        logical_run_id=str(baseline_spec["run_id"]),
        reason=str(baseline_spec["reason"]),
        parent_id=None,
        config=base_config,
        cycle_root=cycle_root,
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        image_root=local_paths["image_data_root"],
        data_version=validation_report,
        git_commit=git_commit,
        tracking_uri=tracking_uri,
        parent_macro_f1=None,
        runner_config=runner_config,
    )
    summaries["baseline"] = baseline
    configs["baseline"] = base_config

    for trial_spec in plan["trials"]:
        if (
            time.monotonic() - cycle_start
            > float(runner_config["cycle_max_hours"]) * 3600
        ):
            raise RuntimeError("Research cycle time limit reached.")
        parent_selector = str(trial_spec["parent"])
        if parent_selector == "best_so_far":
            parent_summary = _best_summary(summaries)
            parent_id = str(parent_summary["logical_run_id"])
        else:
            parent_id = parent_selector
            parent_summary = summaries[parent_id]
        parent_config = configs[parent_id]
        trial_config = deep_merge(parent_config, trial_spec.get("overrides", {}))
        logical_run_id = str(trial_spec["run_id"])
        summary = _run_with_retries(
            logical_run_id=logical_run_id,
            reason=str(trial_spec["reason"]),
            parent_id=parent_id,
            config=trial_config,
            cycle_root=cycle_root,
            ledger_path=ledger_path,
            manifest_path=manifest_path,
            image_root=local_paths["image_data_root"],
            data_version=validation_report,
            git_commit=git_commit,
            tracking_uri=tracking_uri,
            parent_macro_f1=float(
                parent_summary["best_validation"]["macro_f1"]
            ),
            runner_config=runner_config,
        )
        summaries[logical_run_id] = summary
        configs[logical_run_id] = trial_config

    winner = _best_summary(summaries)
    winner_id = str(winner["logical_run_id"])
    winner_config = configs[winner_id]
    finalist_summaries = []
    for seed in runner_config["finalist_seeds"]:
        repeat_config = deep_merge(
            winner_config, {"training": {"seed": int(seed)}}
        )
        finalist = _run_with_retries(
            logical_run_id=f"{winner_id}-seed-{seed}",
            reason=f"Repeat finalist {winner_id} with seed {seed} for uncertainty.",
            parent_id=winner_id,
            config=repeat_config,
            cycle_root=cycle_root,
            ledger_path=ledger_path,
            manifest_path=manifest_path,
            image_root=local_paths["image_data_root"],
            data_version=validation_report,
            git_commit=git_commit,
            tracking_uri=tracking_uri,
            # Seed repeats estimate uncertainty. Do not prune them against
            # the selected run, because a slow start can recover later.
            parent_macro_f1=None,
            runner_config=runner_config,
        )
        finalist_summaries.append(finalist)

    selection = {
        "selection_metric": "validation macro F1",
        "winner_logical_run_id": winner_id,
        "winner_run_id": winner["run_id"],
        "winner_validation_macro_f1": winner["best_validation"]["macro_f1"],
        "selected_before_test": True,
        "timestamp_unix": time.time(),
        "git_commit": git_commit,
        "data_version": validation_report["validated_manifest"]["sha256"],
    }
    atomic_write_json(report_dir / "selection.json", selection)
    append_jsonl(ledger_path, {"event": "selection_frozen", **selection})

    baseline_test = evaluate_checkpoint(
        checkpoint_path=Path(baseline["best_checkpoint"]),
        split="test",
        manifest_path=manifest_path,
        image_root=local_paths["image_data_root"],
        output_path=report_dir / "baseline_test_metrics.json",
    )
    winner_test = evaluate_checkpoint(
        checkpoint_path=Path(winner["best_checkpoint"]),
        split="test",
        manifest_path=manifest_path,
        image_root=local_paths["image_data_root"],
        output_path=report_dir / "winner_test_metrics.json",
    )

    uncertainty_values = [float(winner["best_validation"]["macro_f1"])] + [
        float(summary["best_validation"]["macro_f1"])
        for summary in finalist_summaries
    ]
    final_report = {
        "status": "completed",
        "git_commit": git_commit,
        "data_version": validation_report,
        "benchmark": benchmark,
        "smoke": smoke,
        "trials": list(summaries.values()),
        "finalist_seed_repeats": finalist_summaries,
        "selection": selection,
        "validation_macro_f1_uncertainty": _uncertainty(uncertainty_values),
        "baseline_test": baseline_test,
        "winner_test": winner_test,
        "mlflow_tracking_uri": tracking_uri,
        "ledger_path": str(ledger_path),
    }
    atomic_write_json(report_dir / "final_report.json", final_report)
    Path(report_dir / "experiment_ledger.jsonl").write_text(
        ledger_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "winner": winner_id}, indent=2))
    return final_report


if __name__ == "__main__":
    run_cycle()
