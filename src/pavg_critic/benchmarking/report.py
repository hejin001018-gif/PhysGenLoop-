"""Stage A smoke report aggregation and deterministic rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample
from .metrics import compute_smoke_metrics


WARNING = (
    "Stage A smoke results validate the pipeline only and are not benchmark "
    "performance claims."
)


def build_smoke_report(
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[BenchmarkPrediction],
) -> dict:
    by_method: dict[str, list[BenchmarkPrediction]] = {}
    for prediction in predictions:
        by_method.setdefault(prediction.method_id, []).append(prediction)
    sample_by_id = {item.sample_id: item for item in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("duplicate sample IDs are not allowed")
    methods = {}
    for method_id in sorted(by_method):
        records = by_method[method_id]
        predicted_ids = {item.sample_id for item in records}
        extra = sorted(predicted_ids - set(sample_by_id))
        if extra:
            raise ValueError(
                f"predictions contain unknown sample IDs for {method_id}: {extra}"
            )
        matching_samples = tuple(
            sample_by_id[item_id] for item_id in sorted(predicted_ids)
        )
        matching_predictions = tuple(
            sorted(records, key=lambda item: item.sample_id)
        )
        missing = sorted(set(sample_by_id) - predicted_ids)
        metrics = (
            compute_smoke_metrics(matching_samples, matching_predictions)
            if matching_samples
            else None
        )
        methods[method_id] = {
            "metrics": metrics,
            "missing_sample_ids": missing,
            "failures": [
                {"sample_id": item.sample_id, **dict(item.failure)}
                for item in matching_predictions
                if item.failure is not None
            ],
        }
    return {
        "schema_version": "1.0",
        "stage": "A_SMOKE",
        "claims_allowed": False,
        "warning": WARNING,
        "sample_count": len(samples),
        "methods": methods,
    }


def write_smoke_report(
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[BenchmarkPrediction],
    output_dir: str | Path,
) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = build_smoke_report(samples, predictions)
    (destination / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# VideoPhy-2 Stage A Smoke",
        "",
        f"> **Warning:** {WARNING}",
        "",
        "| Method | Count | Macro-F1 | Unknown | Failures | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_id, item in report["methods"].items():
        metrics = item["metrics"]
        if metrics is None:
            lines.append(f"| {method_id} | 0 | N/A | N/A | 0 | N/A |")
        else:
            lines.append(
                f"| {method_id} | {metrics['count']} | "
                f"{metrics['macro_f1']:.3f} | {metrics['unknown_rate']:.3f} | "
                f"{len(item['failures'])} | {metrics['mean_latency_sec']:.3f} |"
            )
    lines.extend(["", "## Failures", ""])
    failures = [
        (method_id, failure)
        for method_id, item in report["methods"].items()
        for failure in item["failures"]
    ]
    lines.extend(
        [
            f"- `{method_id}` / `{failure['sample_id']}`: "
            f"{failure['type']}"
            + (
                f" — {failure['message']}"
                if failure.get("message")
                else ""
            )
            for method_id, failure in failures
        ]
        or ["- None"]
    )
    (destination / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report
