# Birdly local research result

The selection and test results below describe the original research cycle.
On 2026-09-21, the user replaced that selection with the seed-2027 model
after a later validation result. The consolidated final model record is in
`reports/final_model.md`. The test results below apply only to the
original selected run and baseline.

The local research cycle is complete. The fixed training worktree ran commit
`b56c0b5bf23a73940397c8169af9514264e4b7e5`. The validated manifest
SHA-256 is `5b218b62ea25aec49a5237aaf2a7257b157ea9e914110e0bbfdbda638964fb2e`.
The image data DVC version is `14fdb8a01c29673a5b2c3cb003cebaed.dir`.
All run summaries record the commit, data version, frozen configuration, and
reason. No test result was used to select a trial.

## Data and method

The validated data has 236 classes, 226,478 training images, 2,353 validation
images, and 4,712 test images. Validation removed 253 ambiguous photo-reference
rows. The final manifest has no duplicate photo ID or photo UUID group and no
observation group that crosses splits. Validation and test have position-0
images. The full checks are in `data/processed/validation_report.json`.

The measured safe setting was a 224-pixel image, batch size 32, AMP, two data
workers, and prefetch factor 2. The benchmark measured a full training step.
Peak allocated GPU memory was about 2,296 MiB on a 4,096 MiB GPU. The runner
used one GPU trial at a time. Each trial had a six-epoch and ten-hour limit,
one reduced-batch OOM retry, safety pauses, and atomic checkpoints. Trials
that were clearly below their parent after epoch 3 were pruned. The two seed
repeats ran to epoch 6. The benchmark, run configurations, checkpoints, and
event ledger remain on `B:`.

## Validation results

The selection metric is macro F1. The delta is against the named parent, not
against a concurrent control. Each listed cost is wall-clock GPU trial time.
Peak GPU memory is allocated memory, not total process memory. A pruned run's
epoch-3 metric is not a full six-epoch comparison.

| Run | Parent | Change and reason | Best epoch | F1 | Top-1 | MCC | ECE | F1 delta | Time | Peak GPU MiB | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | none | ConvNeXt Tiny reference | 6 | .8733 | .8738 | .8733 | .068 | — | 5.20 h | 2297 | complete |
| trial-01-lr-low | baseline | LR .0001; test a lower rate | 5 | .8809 | .8823 | .8818 | .069 | +.0076 | 7.35 h | 2296 | select |
| trial-02-lr-high | baseline | LR .0009; test a higher rate | 3 | .7193 | .7233 | .7223 | .098 | -.1540 | 4.34 h | 2296 | prune |
| trial-03-weight-decay-low | trial-01 | WD .01; test less regularization | 3 | .8475 | .8487 | .8481 | .078 | -.0334 | 2.85 h | 2296 | prune |
| trial-04-weight-decay-high | trial-01 | WD .10; test more regularization | 3 | .8403 | .8423 | .8417 | .073 | -.0405 | 2.72 h | 2296 | prune |
| trial-05-augmentation | trial-01 | Stronger crop, color, RandAugment, and erasing | 3 | .8409 | .8428 | .8422 | .092 | -.0400 | 3.63 h | 2296 | prune |
| trial-06-resolution-192 | trial-01 | 192 pixels; test lower image cost | 3 | .8300 | .8317 | .8311 | .083 | -.0509 | 2.44 h | 1819 | prune |
| trial-07-mobilenet-v3 | trial-01 | Pretrained MobileNet V3 Large; test a smaller model | 3 | .7399 | .7408 | .7398 | .115 | -.1410 | 2.69 h | 801 | prune |
| trial-08-efficientnet-b0 | trial-01 | Pretrained EfficientNet B0; test a smaller model | 3 | .7609 | .7641 | .7632 | .118 | -.1199 | 2.65 h | 1465 | prune |
| trial-01-lr-low-seed-1337 | trial-01 | New seed; measure variation | 6 | .8861 | .8870 | .8865 | .068 | +.0053 | 6.58 h | 2297 | complete |
| trial-01-lr-low-seed-2027 | trial-01 | New seed; measure variation | 6 | .8834 | .8836 | .8831 | .071 | +.0025 | 5.44 h | 2297 | complete |

The full JSON report records top-5 accuracy, macro recall, loss, per-class
precision, recall, F1, support, throughput, and safety resource samples for
each run. The selected run and two repeats have mean validation macro F1
`.88346`, standard deviation `.00264`, and a two-sided 95% t interval
`[.87691, .89001]`. This interval has only three seeds. It is a rough measure
of seed variation, not a claim about all data or hardware.

## Selection and test

The runner selected `trial-01-lr-low` by validation macro F1. The selection
record was written before either test evaluation. The test result uses the
baseline and selected winner best checkpoints. Each test evaluation loaded
the checkpoint and verified a finite 236-class inference output.

| Checkpoint | Test macro F1 | Top-1 | Top-5 | MCC | Macro recall | Loss | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | .866331 | .866723 | .964559 | .866186 | .866592 | .588666 | .061963 |
| trial-01-lr-low | .865941 | .866299 | .964983 | .865758 | .866157 | .583324 | .058838 |

The selected winner is lower than the baseline by `.000390` test macro F1.
This small difference does not change the validation-based selection. Do not
use this test difference to plan another trial in this frozen cycle.

Macro F1 is the unweighted mean of class F1 values. Top-1 is the fraction of
correct first predictions. Top-5 is the fraction with the true class in the
five highest predictions. MCC is the multiclass Matthews correlation
coefficient. Macro recall is the unweighted mean of class recalls. Loss is
mean cross-entropy loss. ECE is expected calibration error from 15 confidence
bins; lower is better. The JSON reports contain the exact values and each
class's result.

## Recovery and saved records

Windows stopped with a BSOD during seed 1337. After `B:` was remounted, the
fixed worktree resumed from its atomic latest checkpoint. The checkpoint
restored model, optimizer, scheduler, scaler, random states, class mapping,
preprocessing, and configuration. No checkpoint corruption was found. A
stopped epoch can replay batches after the last checkpoint. This recovery
showed that a system crash can be resumed. It does not prove that every system
failure will preserve the last checkpoint.

The original runner pruned seed 2027 after epoch 3. The same committed
training code and frozen configuration then completed seed 2027 to epoch 6
without parent-based pruning. Two continuation epochs confirmed that epoch 6
was the best epoch. The final model directory contains the source records
and the combined training history. See `reports/final_model.md`.

The saved records are:

- DVC report outputs: `reports/research/` (local DVC cache, not Git content).
- Full run directories and checkpoints: `B:/birdly-training/birdly-research-v3/runs/`.
- Event ledger: `B:/birdly-training/birdly-research-v3/experiment_ledger.jsonl`.
- MLflow database: `B:/birdly-training/birdly-research-v3/mlflow.db`.
- Cycle log after recovery: `B:/birdly-training/birdly-research-v3/research_cycle.resume-20260912-2016.stdout.log`.
- Final model record: `reports/final_model.md`.
- Final model files: `B:/birdly-training/birdly-research-v3/runs/trial-01-lr-low-seed-2027/`.

The final consolidation removed the two obsolete continuation run rows. The
MLflow database integrity check returned `ok` after this change. The final
model has one active training run and one public registry version.

The DVC lock records the outputs from the fixed worktree. That worktree showed
an up-to-date DVC pipeline after the result correction. A Windows checkout on
`C:` can report changed dependencies because its Git checkout uses CRLF line
ends while the fixed worktree used LF. The later seed-repeat code fix also
changes the runner dependency. Do not run `dvc repro research_cycle` in this
checkout to relabel the completed trial results.
