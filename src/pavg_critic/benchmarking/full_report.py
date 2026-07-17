"""Strict, deterministic prediction merging for full benchmark reports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample


@dataclass(frozen=True)
class PredictionMergeResult:
    """Validated predictions and the audit of their source/output artifacts."""

    predictions: tuple[BenchmarkPrediction, ...]
    artifact_audit: dict[str, Any]


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
