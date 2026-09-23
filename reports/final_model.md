# Final model record

The final model is `trial-01-lr-low-seed-2027`. The model stopped after
epoch 8 because epochs 7 and 8 did not improve the validation macro F1.
The best checkpoint is the epoch-6 checkpoint. No training was run during
the consolidation.

## Final model files

The canonical directory is:

`B:/birdly-training/birdly-research-v3/runs/trial-01-lr-low-seed-2027/`

| Information | File |
| --- | --- |
| Best checkpoint | `best.pt` |
| Latest resumable checkpoint | `latest.pt` |
| Class mapping | `class_mapping.json` |
| Preprocessing | `preprocessing.json` |
| Input size | `preprocessing.json` (`image_size: 224`) and `config.yaml` |
| Test metrics | `test_metrics.json` |
| Training configuration | `config.yaml` |
| Validation metrics | `best_validation_metrics.json` |
| Epoch history | `metrics_history.json` |
| Resource records | `resource_telemetry.jsonl` |
| Source segment records | `segment_records.json` |
| Training logs | `logs/` |
| File index and hashes | `final_model_manifest.json` |

The best-checkpoint SHA-256 is
`6ffd18b6edaa248b594c625e2a48f74358b3a771cf807f019b14f1324d719203`.
The data version is
`5b218b62ea25aec49a5237aaf2a7257b157ea9e914110e0bbfdbda638964fb2e`.
The model has 236 output classes.

The checkpoint also contains the class mapping, preprocessing values,
training configuration, optimizer state, scheduler state, AMP scaler state,
and random-number states. The separate JSON and YAML files make the final
model information easy to inspect without loading the checkpoint.

## Validation and test results

| Metric | Validation | Test |
| --- | ---: | ---: |
| Macro F1 | .883374 | .866013 |
| Top-1 accuracy | .883553 | .866299 |
| Top-5 accuracy | .969401 | .967530 |
| MCC | .883099 | .865755 |
| Macro recall | .883427 | .866183 |
| Loss | .523362 | .569868 |
| Expected calibration error | .070521 | .061297 |

The test split has 4,712 images. The test result contains 236 per-class
records. The selected checkpoint produced a finite output with 236 classes.
The final model selection used validation data. No test result was used for a
new trial.

## Consolidated training record

The original run and two continuation directories are now one canonical
directory. `metrics_history.json` contains epochs 1 through 8.
`resource_telemetry.jsonl` contains the resource records from all three
segments. `segment_records.json` contains the source run IDs, commits,
configuration records, checkpoint hashes, and prior summaries. The `logs/`
directory contains the six source log files with canonical names.

The obsolete continuation directories and root log copies were removed from
`B:` after their hashes were verified. The retired continuation DVC stages,
configuration files, source files, tests, and reports were also removed.

## MLflow

The canonical training run ID is
`77f7cd76e2cf4ed0829c6ebbf8de8b65`. Its artifact directory is:

`B:/birdly-training/birdly-research-v3/mlflow-run-artifacts/1/77f7cd76e2cf4ed0829c6ebbf8de8b65/artifacts/`

The run contains the consolidated files under `final/`. MLflow shows one
active training run for this model. The removed continuation runs do not
remain in the runs table.

The selected registry model is
`models:/birdly-bird-classifier@selected`. The alias points to version 2,
which has status `READY`. MLflow shows no other public version. The package
run ID is `ffad6324832c4c40a76ea02c9de94efa`. The package contains the
checkpoint, inference code, dependency files, class mapping, and
preprocessing values.

The MLflow database is:

`B:/birdly-training/birdly-research-v3/mlflow.db`

The database backup made before this consolidation is:

`B:/birdly-training/birdly-research-v3/backups/mlflow-pre-final-consolidation-20261003-032032.db`

The database integrity check returned `ok` after the consolidation.
