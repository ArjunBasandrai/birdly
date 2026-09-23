from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ClassificationMetrics:
    num_classes: int
    calibration_bins: int = 15

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )
        self.loss_sum = 0.0
        self.sample_count = 0
        self.top5_correct = 0
        self.bin_count = torch.zeros(self.calibration_bins, dtype=torch.float64)
        self.bin_confidence = torch.zeros(self.calibration_bins, dtype=torch.float64)
        self.bin_accuracy = torch.zeros(self.calibration_bins, dtype=torch.float64)

    def update(
        self, logits: torch.Tensor, targets: torch.Tensor, loss_sum: float
    ) -> None:
        probabilities = torch.softmax(logits.detach(), dim=1)
        confidence, predictions = probabilities.max(dim=1)
        targets = targets.detach()
        batch_size = targets.numel()

        indices = targets.to(torch.int64) * self.num_classes + predictions.to(
            torch.int64
        )
        batch_confusion = torch.bincount(
            indices.cpu(), minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += batch_confusion

        top_k = min(5, self.num_classes)
        top_indices = logits.detach().topk(top_k, dim=1).indices
        self.top5_correct += int(
            top_indices.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        )
        self.loss_sum += float(loss_sum)
        self.sample_count += batch_size

        bin_indices = torch.clamp(
            (confidence * self.calibration_bins).to(torch.int64),
            max=self.calibration_bins - 1,
        ).cpu()
        correct = predictions.eq(targets).to(torch.float64).cpu()
        confidence = confidence.to(torch.float64).cpu()
        for bin_index in range(self.calibration_bins):
            mask = bin_indices == bin_index
            count = int(mask.sum().item())
            if count:
                self.bin_count[bin_index] += count
                self.bin_confidence[bin_index] += confidence[mask].sum()
                self.bin_accuracy[bin_index] += correct[mask].sum()

    def compute(self, class_to_taxon: dict[int, int]) -> dict[str, Any]:
        confusion = self.confusion.to(torch.float64)
        true_positive = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        recall = torch.where(support > 0, true_positive / support, 0.0)
        precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
        f1 = torch.where(
            precision + recall > 0,
            2.0 * precision * recall / (precision + recall),
            0.0,
        )

        total = confusion.sum()
        correct = true_positive.sum()
        covariance = correct * total - torch.dot(predicted, support)
        denominator = torch.sqrt(
            (total.square() - predicted.square().sum())
            * (total.square() - support.square().sum())
        )
        mcc = float((covariance / denominator).item()) if denominator > 0 else 0.0

        nonempty = self.bin_count > 0
        average_confidence = torch.zeros_like(self.bin_count)
        average_accuracy = torch.zeros_like(self.bin_count)
        average_confidence[nonempty] = (
            self.bin_confidence[nonempty] / self.bin_count[nonempty]
        )
        average_accuracy[nonempty] = (
            self.bin_accuracy[nonempty] / self.bin_count[nonempty]
        )
        calibration_error = (
            (self.bin_count / max(self.sample_count, 1))
            * (average_accuracy - average_confidence).abs()
        ).sum()

        per_class = []
        for class_index in range(self.num_classes):
            per_class.append(
                {
                    "class_index": class_index,
                    "taxon_id": int(class_to_taxon[class_index]),
                    "support": int(support[class_index].item()),
                    "precision": float(precision[class_index].item()),
                    "recall": float(recall[class_index].item()),
                    "f1": float(f1[class_index].item()),
                }
            )

        return {
            "loss": self.loss_sum / max(self.sample_count, 1),
            "top1_accuracy": float(correct.item()) / max(self.sample_count, 1),
            "top5_accuracy": self.top5_correct / max(self.sample_count, 1),
            "macro_precision": float(precision.mean().item()),
            "macro_recall": float(recall.mean().item()),
            "macro_f1": float(f1.mean().item()),
            "mcc": mcc,
            "expected_calibration_error": float(calibration_error.item()),
            "sample_count": self.sample_count,
            "per_class": per_class,
        }
