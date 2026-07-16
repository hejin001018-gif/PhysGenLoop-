"""Dependency-free smoke metrics with explicit unknown handling."""

from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _spearman(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _rank(left)
    y = _rank(right)
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_scale = sqrt(sum((a - x_mean) ** 2 for a in x))
    y_scale = sqrt(sum((b - y_mean) ** 2 for b in y))
    return None if x_scale == 0 or y_scale == 0 else numerator / (x_scale * y_scale)


def compute_smoke_metrics(
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[BenchmarkPrediction],
) -> dict[str, float | int | None]:
    if not samples:
        raise ValueError("metrics require at least one sample")
    gold = {item.sample_id: item for item in samples}
    predicted = {item.sample_id: item for item in predictions}
    if len(gold) != len(samples) or len(predicted) != len(predictions):
        raise ValueError("duplicate sample IDs are not allowed")
    if set(gold) != set(predicted):
        raise ValueError("gold and prediction sample IDs must match exactly")
    pairs = [(gold[key], predicted[key]) for key in sorted(gold)]
    classes = ("physical", "violation")
    class_f1: list[float] = []
    class_recall: list[float] = []
    precision_by_class: dict[str, float] = {}
    for target in classes:
        tp = sum(
            g.physics_label == target and p.physics_label == target for g, p in pairs
        )
        fp = sum(
            g.physics_label != target and p.physics_label == target for g, p in pairs
        )
        fn = sum(
            g.physics_label == target and p.physics_label != target for g, p in pairs
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precision_by_class[target] = precision
        class_recall.append(recall)
        class_f1.append(f1)
    ordinal = [
        (g.physics_score, p.physics_score)
        for g, p in pairs
        if g.physics_score is not None and p.physics_score is not None
    ]
    return {
        "count": len(pairs),
        "accuracy": sum(
            g.physics_label == p.physics_label for g, p in pairs
        )
        / len(pairs),
        "balanced_accuracy": mean(class_recall),
        "macro_f1": mean(class_f1),
        "violation_precision": precision_by_class["violation"],
        "violation_recall": class_recall[1],
        "unknown_rate": sum(p.physics_label == "unknown" for _, p in pairs)
        / len(pairs),
        "failure_rate": sum(p.failure is not None for _, p in pairs) / len(pairs),
        "mean_latency_sec": mean(p.latency_sec for _, p in pairs),
        "mean_visible_frames": mean(p.visible_frame_count for _, p in pairs),
        "physics_spearman": _spearman(
            [a for a, _ in ordinal],
            [b for _, b in ordinal],
        ),
    }
