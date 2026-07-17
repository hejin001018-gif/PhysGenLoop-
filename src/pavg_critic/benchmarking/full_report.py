"""Strict, deterministic prediction merging for full benchmark reports."""

from __future__ import annotations

import hashlib
import ast
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample
from .metrics import compute_smoke_metrics


@dataclass(frozen=True)
class PredictionMergeResult:
    """Validated predictions and the audit of their source/output artifacts."""

    predictions: tuple[BenchmarkPrediction, ...]
    artifact_audit: dict[str, Any]


_BASELINE_METHOD = "D0_DIRECT_VLM"
_CANDIDATE_METHOD = "B1_RULE"
_BINARY_PHYSICS_LABELS = ("physical", "violation")


def _prediction_index(
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[BenchmarkPrediction],
    *,
    expected_method: str,
) -> dict[str, BenchmarkPrediction]:
    """Validate one method's exact sample coverage and return an ID index."""

    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("benchmark samples contain duplicate sample_id values")
    indexed: dict[str, BenchmarkPrediction] = {}
    for prediction in predictions:
        if prediction.method_id != expected_method:
            raise ValueError(
                f"expected method_id {expected_method!r}, got "
                f"{prediction.method_id!r}"
            )
        if prediction.sample_id in indexed:
            raise ValueError(
                f"duplicate {expected_method} prediction sample_id "
                f"{prediction.sample_id!r}"
            )
        indexed[prediction.sample_id] = prediction
    expected_ids = set(sample_ids)
    actual_ids = set(indexed)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"{expected_method} prediction sample IDs must match exactly; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return indexed


def _is_correct(sample: BenchmarkSample, prediction: BenchmarkPrediction) -> bool:
    """Return strict binary correctness, treating unknown/failure as misses."""

    return (
        sample.physics_label in _BINARY_PHYSICS_LABELS
        and prediction.failure is None
        and prediction.physics_label in _BINARY_PHYSICS_LABELS
        and sample.physics_label == prediction.physics_label
    )


def _strict_macro_f1(
    samples: Sequence[BenchmarkSample],
    predictions: Mapping[str, BenchmarkPrediction],
) -> float:
    """Compute binary macro-F1 while retaining duplicate bootstrap rows."""

    f1_values: list[float] = []
    for target in _BINARY_PHYSICS_LABELS:
        true_positive = false_positive = false_negative = 0
        for sample in samples:
            prediction = predictions[sample.sample_id]
            predicted_label = (
                prediction.physics_label
                if prediction.failure is None
                else "unknown"
            )
            if sample.physics_label == target and predicted_label == target:
                true_positive += 1
            elif sample.physics_label != target and predicted_label == target:
                false_positive += 1
            elif sample.physics_label == target and predicted_label != target:
                false_negative += 1
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(f1_values) / len(f1_values)


def _strict_metrics(
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[BenchmarkPrediction],
) -> dict[str, float | int | None]:
    """Return smoke metrics with strict unknown/failure correctness semantics."""

    sanitized = tuple(
        prediction
        if prediction.failure is None
        else BenchmarkPrediction(
            sample_id=prediction.sample_id,
            method_id=prediction.method_id,
            model_id=prediction.model_id,
            semantic_score=prediction.semantic_score,
            physics_score=prediction.physics_score,
            semantic_label=prediction.semantic_label,
            physics_label="unknown",
            confidence=prediction.confidence,
            coverage=prediction.coverage,
            latency_sec=prediction.latency_sec,
            visible_frame_count=prediction.visible_frame_count,
            violation_categories=prediction.violation_categories,
            evidence_frames=prediction.evidence_frames,
            repair_instruction=prediction.repair_instruction,
            failure=prediction.failure,
        )
        for prediction in predictions
    )
    metrics = compute_smoke_metrics(samples, sanitized)
    indexed = {prediction.sample_id: prediction for prediction in predictions}
    metrics["accuracy"] = sum(
        _is_correct(sample, indexed[sample.sample_id]) for sample in samples
    ) / len(samples)
    metrics["macro_f1"] = _strict_macro_f1(samples, indexed)
    return metrics


def paired_outcomes(
    samples: Sequence[BenchmarkSample],
    baseline_predictions: Sequence[BenchmarkPrediction],
    candidate_predictions: Sequence[BenchmarkPrediction],
) -> dict[str, int]:
    """Count the four paired correctness outcomes for D0 versus B1."""

    baseline = _prediction_index(
        samples, baseline_predictions, expected_method=_BASELINE_METHOD
    )
    candidate = _prediction_index(
        samples, candidate_predictions, expected_method=_CANDIDATE_METHOD
    )
    outcomes = {
        "both_correct": 0,
        "baseline_only_correct": 0,
        "candidate_only_correct": 0,
        "both_wrong": 0,
    }
    for sample in samples:
        baseline_correct = _is_correct(sample, baseline[sample.sample_id])
        candidate_correct = _is_correct(sample, candidate[sample.sample_id])
        if baseline_correct and candidate_correct:
            outcomes["both_correct"] += 1
        elif baseline_correct:
            outcomes["baseline_only_correct"] += 1
        elif candidate_correct:
            outcomes["candidate_only_correct"] += 1
        else:
            outcomes["both_wrong"] += 1
    return outcomes


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def action_group_bootstrap(
    samples: Sequence[BenchmarkSample],
    baseline_predictions: Sequence[BenchmarkPrediction],
    candidate_predictions: Sequence[BenchmarkPrediction],
    *,
    resamples: int = 2000,
    seed: int = 20260717,
) -> dict[str, float | int]:
    """Bootstrap candidate-minus-baseline Macro-F1 by action group.

    Each draw samples the sorted action-group list with replacement and adds
    every row in each selected group, preserving multiplicity. This avoids
    the duplicate-ID restriction used by ordinary smoke metrics.
    """

    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    baseline = _prediction_index(
        samples, baseline_predictions, expected_method=_BASELINE_METHOD
    )
    candidate = _prediction_index(
        samples, candidate_predictions, expected_method=_CANDIDATE_METHOD
    )
    ordered_samples = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    groups: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for sample in ordered_samples:
        groups[sample.prompt_group_id].append(sample)
    group_ids = sorted(groups)
    point_estimate = _strict_macro_f1(ordered_samples, candidate) - _strict_macro_f1(
        ordered_samples, baseline
    )
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        draw: list[BenchmarkSample] = []
        for _ in group_ids:
            draw.extend(groups[rng.choice(group_ids)])
        candidate_f1 = _strict_macro_f1(draw, candidate)
        baseline_f1 = _strict_macro_f1(draw, baseline)
        deltas.append(candidate_f1 - baseline_f1)
    return {
        "point_estimate": point_estimate,
        "lower": _linear_percentile(deltas, 0.025),
        "upper": _linear_percentile(deltas, 0.975),
        "resamples": resamples,
        "seed": seed,
        "group_count": len(group_ids),
    }


def parse_rule_families(sample: BenchmarkSample) -> tuple[str, ...]:
    """Parse exact source rule-family labels from ``metadata_rules`` only."""

    raw_metadata = sample.raw_labels.get("metadata_rules")
    if isinstance(raw_metadata, str):
        try:
            metadata = ast.literal_eval(raw_metadata)
        except (SyntaxError, ValueError, TypeError, MemoryError):
            return ("__unmapped__",)
    elif isinstance(raw_metadata, Mapping):
        metadata = raw_metadata
    else:
        return ("__unmapped__",)
    if not isinstance(metadata, Mapping):
        return ("__unmapped__",)
    families: set[str] = set()
    for value in metadata.values():
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, (list, tuple, set)):
            values = value
        else:
            continue
        for family in values:
            if isinstance(family, str) and family.strip():
                families.add(family.strip())
    return tuple(sorted(families)) or ("__unmapped__",)


def build_slices(
    samples: Sequence[BenchmarkSample],
    baseline_predictions: Sequence[BenchmarkPrediction],
    candidate_predictions: Sequence[BenchmarkPrediction],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build deterministic generator, action and source-rule-family slices."""

    baseline = _prediction_index(
        samples, baseline_predictions, expected_method=_BASELINE_METHOD
    )
    candidate = _prediction_index(
        samples, candidate_predictions, expected_method=_CANDIDATE_METHOD
    )
    ordered_samples = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    dimensions: dict[str, dict[str, list[BenchmarkSample]]] = {
        "action": defaultdict(list),
        "generator": defaultdict(list),
        "rule_family": defaultdict(list),
    }
    for sample in ordered_samples:
        dimensions["action"][sample.prompt_group_id].append(sample)
        dimensions["generator"][sample.generator].append(sample)
        for family in parse_rule_families(sample):
            dimensions["rule_family"][family].append(sample)

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in ("action", "generator", "rule_family"):
        slices: dict[str, dict[str, Any]] = {}
        for name in sorted(dimensions[dimension]):
            members = tuple(dimensions[dimension][name])
            baseline_predictions_for_slice = tuple(
                baseline[sample.sample_id] for sample in members
            )
            candidate_predictions_for_slice = tuple(
                candidate[sample.sample_id] for sample in members
            )
            baseline_metrics = _strict_metrics(
                members, baseline_predictions_for_slice
            )
            candidate_metrics = _strict_metrics(
                members, candidate_predictions_for_slice
            )
            slices[name] = {
                "count": len(members),
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "candidate_minus_baseline": {
                    "accuracy": candidate_metrics["accuracy"]
                    - baseline_metrics["accuracy"],
                    "macro_f1": candidate_metrics["macro_f1"]
                    - baseline_metrics["macro_f1"],
                },
            }
        result[dimension] = slices
    return result


def summarize_observation_latencies(
    samples: Sequence[BenchmarkSample],
    observation_meta_dirs: Sequence[str | Path],
) -> dict[str, Any]:
    """Summarize SAM2 production latency from metadata-only cache files."""

    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("benchmark samples contain duplicate sample_id values")
    if not observation_meta_dirs:
        raise ValueError("at least one observation metadata directory is required")

    known_sample_ids = set(sample_ids)
    metadata_by_sample: dict[str, dict[str, Any]] = {}
    latency_by_sample: dict[str, float] = {}
    directories = sorted(
        (Path(raw_path) for raw_path in observation_meta_dirs),
        key=_canonical_path_order,
    )
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(
                f"observation metadata directory does not exist: {directory}"
            )
        for path in sorted(directory.rglob("*.meta.json"), key=_canonical_path_order):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"invalid observation metadata JSON: {path}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"observation metadata must be an object: {path}")
            sample_id = raw.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"invalid observation metadata sample_id in {path}: {sample_id!r}"
                )
            if sample_id not in known_sample_ids:
                raise ValueError(
                    f"unknown observation sample_id {sample_id!r} in {path}"
                )
            previous = metadata_by_sample.get(sample_id)
            if previous is not None and raw != previous:
                raise ValueError(
                    f"conflicting observation metadata for sample_id {sample_id!r}"
                )
            metadata_by_sample[sample_id] = raw

            latency = raw.get("production_latency_sec")
            if latency is None:
                continue
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
            ):
                raise ValueError(
                    f"invalid production_latency_sec for sample_id "
                    f"{sample_id!r}: {latency!r}"
                )
            try:
                valid_latency = math.isfinite(latency) and latency >= 0
            except (OverflowError, TypeError):
                valid_latency = False
            if not valid_latency:
                raise ValueError(
                    f"invalid production_latency_sec for sample_id "
                    f"{sample_id!r}: {latency!r}"
                )
            latency_by_sample[sample_id] = float(latency)

    latencies = [latency_by_sample[sample_id] for sample_id in sample_ids if sample_id in latency_by_sample]
    missing_sample_ids = sorted(known_sample_ids - latency_by_sample.keys())
    return {
        "expected_count": len(sample_ids),
        "valid_count": len(latencies),
        "missing_count": len(missing_sample_ids),
        "missing_sample_ids": missing_sample_ids,
        "mean_production_latency_sec": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "p50_production_latency_sec": (
            _linear_percentile(latencies, 0.50) if latencies else None
        ),
        "p95_production_latency_sec": (
            _linear_percentile(latencies, 0.95) if latencies else None
        ),
    }


def _required_finite_metric(
    values: Mapping[str, Any],
    key: str,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {context} {key}: {value!r}")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError):
        finite = False
    if not finite:
        raise ValueError(f"invalid {context} {key}: {value!r}")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(
            f"{context} {key} is outside [{minimum}, {maximum}]: {value!r}"
        )
    if maximum is not None and numeric > maximum:
        raise ValueError(
            f"{context} {key} is outside [{minimum}, {maximum}]: {value!r}"
        )
    return numeric


def evaluate_material_improvement(
    *,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    slices: Mapping[str, Any],
    baseline_failure_rate: float,
    candidate_failure_rate: float,
) -> dict[str, Any]:
    """Apply frozen VideoPhy-2 gates while keeping OOD verdict deferred."""

    baseline_macro_f1 = _required_finite_metric(
        baseline_metrics,
        "macro_f1",
        context="baseline metric",
        minimum=0.0,
        maximum=1.0,
    )
    candidate_macro_f1 = _required_finite_metric(
        candidate_metrics,
        "macro_f1",
        context="candidate metric",
        minimum=0.0,
        maximum=1.0,
    )
    physical_recall = _required_finite_metric(
        candidate_metrics,
        "physical_recall",
        context="candidate metric",
        minimum=0.0,
        maximum=1.0,
    )
    violation_recall = _required_finite_metric(
        candidate_metrics,
        "violation_recall",
        context="candidate metric",
        minimum=0.0,
        maximum=1.0,
    )
    bootstrap_lower = _required_finite_metric(
        bootstrap,
        "lower",
        context="bootstrap",
        minimum=-1.0,
        maximum=1.0,
    )
    bootstrap_upper = _required_finite_metric(
        bootstrap,
        "upper",
        context="bootstrap",
        minimum=-1.0,
        maximum=1.0,
    )
    if bootstrap_lower > bootstrap_upper:
        raise ValueError(
            "bootstrap lower must not exceed upper: "
            f"lower={bootstrap_lower!r}, upper={bootstrap_upper!r}"
        )
    failure_rates = {
        "baseline": baseline_failure_rate,
        "candidate": candidate_failure_rate,
    }
    baseline_failure = _required_finite_metric(
        failure_rates,
        "baseline",
        context="failure rate",
        minimum=0.0,
        maximum=1.0,
    )
    candidate_failure = _required_finite_metric(
        failure_rates,
        "candidate",
        context="failure rate",
        minimum=0.0,
        maximum=1.0,
    )

    generator_slices = slices.get("generator")
    if not isinstance(generator_slices, Mapping):
        raise ValueError("slices must contain a generator mapping")
    positive_generator_count = 0
    for name, generator_slice in generator_slices.items():
        if not isinstance(generator_slice, Mapping):
            raise ValueError(f"invalid generator slice {name!r}")
        delta = generator_slice.get("candidate_minus_baseline")
        if not isinstance(delta, Mapping):
            raise ValueError(f"missing candidate delta for generator {name!r}")
        macro_f1_delta = _required_finite_metric(
            delta,
            "macro_f1",
            context=f"generator {name!r} delta",
            minimum=-1.0,
            maximum=1.0,
        )
        positive_generator_count += macro_f1_delta > 0

    macro_f1_delta_decimal = Decimal(str(candidate_macro_f1)) - Decimal(
        str(baseline_macro_f1)
    )
    failure_rate_increase_decimal = Decimal(str(candidate_failure)) - Decimal(
        str(baseline_failure)
    )
    macro_f1_delta = float(macro_f1_delta_decimal)
    failure_rate_increase = float(failure_rate_increase_decimal)
    gates = {
        "macro_f1_delta": {
            "value": macro_f1_delta,
            "threshold": 0.05,
            "operator": ">=",
            "pass": macro_f1_delta_decimal >= Decimal("0.05"),
        },
        "bootstrap_lower": {
            "value": bootstrap_lower,
            "threshold": 0.0,
            "operator": ">",
            "pass": bootstrap_lower > 0.0,
        },
        "candidate_nonzero_recalls": {
            "value": {
                "physical_recall": physical_recall,
                "violation_recall": violation_recall,
            },
            "threshold": {
                "physical_recall": 0.0,
                "violation_recall": 0.0,
            },
            "operator": ">",
            "pass": physical_recall > 0.0 and violation_recall > 0.0,
        },
        "failure_rate_increase": {
            "value": failure_rate_increase,
            "threshold": 0.01,
            "operator": "<=",
            "pass": failure_rate_increase_decimal <= Decimal("0.01"),
        },
        "positive_generator_count": {
            "value": positive_generator_count,
            "threshold": 2,
            "operator": ">=",
            "pass": positive_generator_count >= 2,
        },
    }
    return {
        "gates": gates,
        "videophy2_support": all(gate["pass"] for gate in gates.values()),
        "ood_status": "deferred",
        "overall_verdict": "not_evaluable_ood_deferred",
    }


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parse_prediction(
    line: str,
    *,
    source: Path,
    line_number: int,
) -> BenchmarkPrediction:
    try:
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError("prediction must be a JSON object")
        if raw.get("semantic_label") not in {
            "adherent",
            "not_adherent",
            "unknown",
        }:
            raise ValueError(f"invalid semantic_label: {raw.get('semantic_label')!r}")
        if raw.get("physics_label") not in {
            "physical",
            "violation",
            "unknown",
        }:
            raise ValueError(f"invalid physics_label: {raw.get('physics_label')!r}")
        for field in ("semantic_score", "physics_score"):
            _validate_numeric_field(
                raw,
                field,
                allow_none=True,
                minimum=1.0,
                maximum=5.0,
            )
        for field in ("confidence", "coverage"):
            _validate_numeric_field(
                raw,
                field,
                minimum=0.0,
                maximum=1.0,
            )
        _validate_numeric_field(raw, "latency_sec", minimum=0.0)
        visible_frame_count = raw.get("visible_frame_count")
        if isinstance(visible_frame_count, bool) or not isinstance(
            visible_frame_count, int
        ):
            raise ValueError(
                "invalid integer visible_frame_count: "
                f"{visible_frame_count!r}"
            )
        if visible_frame_count < 0:
            raise ValueError(
                "visible_frame_count must be non-negative: "
                f"{visible_frame_count!r}"
            )
        return BenchmarkPrediction.from_dict(raw)
    except (
        json.JSONDecodeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"malformed prediction record in {source} at line {line_number}: {exc}"
        ) from exc


def _validate_numeric_field(
    raw: dict[str, Any],
    field: str,
    *,
    allow_none: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    value = raw.get(field)
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid numeric {field}: {value!r}")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError) as exc:
        raise ValueError(f"invalid numeric {field}: {value!r}") from exc
    if not finite:
        raise ValueError(f"non-finite {field}: {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} is below {minimum}: {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} is above {maximum}: {value!r}")


def _serialize_predictions(
    predictions: Sequence[BenchmarkPrediction],
) -> bytes:
    try:
        lines = []
        for prediction in predictions:
            lines.append(
                json.dumps(
                    prediction.to_dict(),
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    except ValueError as exc:
        raise ValueError(
            "prediction record contains a non-finite JSON number"
        ) from exc
    return "".join(lines).encode("utf-8")


def _stage_bytes(path: Path, content: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    staged = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _restore_path(path: Path, *, existed: bool, content: bytes | None) -> None:
    if not existed:
        path.unlink(missing_ok=True)
        return
    if content is None:
        raise RuntimeError(f"missing rollback content for {path}")
    staged = _stage_bytes(path, content)
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _write_artifact_pair(
    output_path: Path,
    output_content: bytes,
    audit_path: Path,
    audit_content: bytes,
) -> None:
    """Replace output and audit together without destroying verified output."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_existed = output_path.exists()
    original_output = output_path.read_bytes() if output_existed else None
    staged_output: Path | None = None
    staged_audit: Path | None = None
    output_replaced = False
    try:
        staged_output = _stage_bytes(output_path, output_content)
        staged_audit = _stage_bytes(audit_path, audit_content)
        os.replace(staged_output, output_path)
        output_replaced = True
        os.replace(staged_audit, audit_path)
    except BaseException:
        if output_replaced:
            _restore_path(
                output_path,
                existed=output_existed,
                content=original_output,
            )
        raise
    finally:
        if staged_output is not None:
            staged_output.unlink(missing_ok=True)
        if staged_audit is not None:
            staged_audit.unlink(missing_ok=True)


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        return os.path.samefile(first, second)
    return False


def _canonical_path_order(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def merge_prediction_shards(
    samples: Sequence[BenchmarkSample],
    prediction_paths: Sequence[str | Path],
    method_ids: tuple[str, ...],
    output_path: str | Path,
    *,
    audit_path: str | Path | None = None,
) -> PredictionMergeResult:
    """Validate exact sample/method coverage and write one stable JSONL file.

    Every parsed prediction is terminal, including records whose ``failure``
    field is populated. Inputs are never rewritten. Unless ``audit_path`` is
    supplied, the audit is written to ``artifact_audit.json`` beside the
    merged output.
    """

    if not samples:
        raise ValueError("at least one benchmark sample is required")
    if not prediction_paths:
        raise ValueError("at least one prediction path is required")
    if not method_ids or len(set(method_ids)) != len(method_ids):
        raise ValueError("method_ids must be a non-empty tuple of unique IDs")

    sources = sorted(
        (Path(raw_path) for raw_path in prediction_paths),
        key=_canonical_path_order,
    )
    for source in sources:
        if not source.is_file():
            raise ValueError(f"prediction input does not exist: {source}")

    destination = Path(output_path)
    audit_destination = (
        Path(audit_path)
        if audit_path is not None
        else destination.with_name("artifact_audit.json")
    )
    if _paths_alias(destination, audit_destination):
        raise ValueError("merged output and audit destination must not alias")
    for source in sources:
        if _paths_alias(destination, source) or _paths_alias(
            audit_destination, source
        ):
            raise ValueError(
                "output or audit destination aliases a prediction input: "
                f"{source}"
            )

    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("benchmark samples contain duplicate sample_id values")

    known_samples = set(sample_ids)
    known_methods = set(method_ids)
    expected_keys = {
        (sample_id, method_id)
        for sample_id in sample_ids
        for method_id in method_ids
    }
    predictions_by_key: dict[tuple[str, str], BenchmarkPrediction] = {}
    locations_by_key: dict[tuple[str, str], tuple[Path, int]] = {}
    input_audits: list[dict[str, Any]] = []

    for source in sources:
        content = source.read_bytes()
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"prediction input is not valid UTF-8: {source}") from exc

        method_counts = {method_id: 0 for method_id in method_ids}
        failure_count = 0
        for line_number, line in enumerate(lines, start=1):
            prediction = _parse_prediction(
                line,
                source=source,
                line_number=line_number,
            )
            if prediction.sample_id not in known_samples:
                raise ValueError(
                    f"unknown sample_id {prediction.sample_id!r} "
                    f"in {source} at line {line_number}"
                )
            if prediction.method_id not in known_methods:
                raise ValueError(
                    f"unknown method_id {prediction.method_id!r} "
                    f"in {source} at line {line_number}"
                )

            key = (prediction.sample_id, prediction.method_id)
            if key in predictions_by_key:
                first_path, first_line = locations_by_key[key]
                raise ValueError(
                    f"duplicate prediction key {key!r}: first seen in "
                    f"{first_path} at line {first_line}, repeated in "
                    f"{source} at line {line_number}"
                )
            predictions_by_key[key] = prediction
            locations_by_key[key] = (source, line_number)
            method_counts[prediction.method_id] += 1
            failure_count += prediction.failure is not None

        input_audits.append(
            {
                "path": str(source),
                "name": source.name,
                "sha256": _sha256(content),
                "line_count": len(lines),
                "method_counts": method_counts,
                "terminal_count": len(lines),
                "failure_count": failure_count,
            }
        )

    missing_keys = expected_keys - predictions_by_key.keys()
    if missing_keys:
        preview = sorted(missing_keys)[:5]
        noun = "key" if len(missing_keys) == 1 else "keys"
        raise ValueError(
            f"missing {len(missing_keys)} expected prediction {noun}: {preview!r}"
        )

    method_order = {method_id: index for index, method_id in enumerate(method_ids)}
    predictions = tuple(
        sorted(
            predictions_by_key.values(),
            key=lambda prediction: (
                prediction.sample_id,
                method_order[prediction.method_id],
            ),
        )
    )
    output_content = _serialize_predictions(predictions)
    artifact_audit = {
        "methods": list(method_ids),
        "inputs": input_audits,
        "expected_count": len(expected_keys),
        "merged_count": len(predictions),
        "duplicate_count": 0,
        "extra_count": 0,
        "missing_count": 0,
        "merged_output_path": str(destination),
        "merged_output_name": destination.name,
        "merged_output_sha256": _sha256(output_content),
        "merged_output_line_count": len(predictions),
        "artifact_audit_path": str(audit_destination),
    }
    audit_content = (
        json.dumps(
            artifact_audit,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_artifact_pair(
        destination,
        output_content,
        audit_destination,
        audit_content,
    )
    return PredictionMergeResult(
        predictions=predictions,
        artifact_audit=artifact_audit,
    )
