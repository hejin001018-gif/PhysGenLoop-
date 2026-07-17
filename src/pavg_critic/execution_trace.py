"""Safe, structured execution tracing for the Physics Critic pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import FusionConfig
from .schemas import CriticReport, EVIDENCE_FAMILIES


TRACE_SCHEMA_VERSION = "pavg-critic-trace/v1"
TRACE_STATUSES = frozenset({"completed", "skipped", "degraded", "error"})
FORBIDDEN_TRACE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "headers",
        "raw_response",
        "image",
        "image_bytes",
        "mask",
        "masks",
        "base64",
    }
)
_COLLECTION_PREVIEW_LIMIT = 20
_STRING_LIMIT = 2_000
_ERROR_MESSAGE_LIMIT = 300
_FIXED_STAGE_IDS = (
    "request",
    "physics_planner",
    "question_graph",
    "video_observation",
    "trajectory",
    "event_detection",
    "mechanics",
    "rule_engine",
    "temporal_localization",
    "visual_evidence",
    "checklist",
    "keyframe_selection",
    "pqsg_execution",
    "vlm_verification",
    "candidate_fusion",
    "question_scoring",
    "evidence_fusion",
    "final_report",
)


class TraceSafetyError(ValueError):
    """Raised when trace data could expose a secret or binary payload."""


@dataclass(frozen=True)
class TraceValidationPolicy:
    """Optional strict requirements for one trace validation run."""

    require_sam2: bool = False
    require_model_planner: bool = False
    fail_on_provider_fallback: bool = False


@dataclass(frozen=True)
class TraceValidationCheck:
    """One independently evaluated trace invariant."""

    code: str
    level: str
    passed: bool
    message: str


@dataclass(frozen=True)
class TraceValidationReport:
    """Complete validator output with stable check codes."""

    checks: tuple[TraceValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(
            check.passed for check in self.checks if check.level == "error"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


@dataclass(frozen=True)
class TraceNodeRecord:
    """One terminal, JSON-safe pipeline-node observation."""

    sequence: int
    node_id: str
    label: str
    status: str
    source_nodes: tuple[str, ...]
    elapsed_ms: float
    inputs: Mapping[str, object]
    outputs: Mapping[str, object]
    parent_id: str | None = None
    warnings: tuple[str, ...] = ()
    error: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "node_id": self.node_id,
            "label": self.label,
            "parent_id": self.parent_id,
            "source_nodes": list(self.source_nodes),
            "status": self.status,
            "elapsed_ms": self.elapsed_ms,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "warnings": list(self.warnings),
            "error": None if self.error is None else dict(self.error),
        }


class _TraceNodeContext(AbstractContextManager["_TraceNodeContext"]):
    def __init__(
        self,
        recorder: "TraceRecorder",
        node_id: str,
        *,
        label: str,
        source_nodes: tuple[str, ...],
        inputs: Mapping[str, object],
        parent_id: str | None,
    ) -> None:
        self._recorder = recorder
        self._node_id = node_id
        self._label = label
        self._source_nodes = source_nodes
        self._inputs = inputs
        self._parent_id = parent_id
        self._started = 0.0
        self._status = "completed"
        self._outputs: Mapping[str, object] = {}
        self._warnings: tuple[str, ...] = ()

    def __enter__(self) -> "_TraceNodeContext":
        self._started = perf_counter()
        return self

    def complete(self, *, outputs: Mapping[str, object]) -> None:
        self._status = "completed"
        self._outputs = outputs
        self._warnings = ()

    def degrade(
        self,
        *,
        outputs: Mapping[str, object],
        warnings: Sequence[str],
    ) -> None:
        self._status = "degraded"
        self._outputs = outputs
        self._warnings = tuple(str(item)[:_ERROR_MESSAGE_LIMIT] for item in warnings)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        elapsed_ms = max(0.0, (perf_counter() - self._started) * 1_000.0)
        if exc is not None:
            self._recorder.record_error(
                self._node_id,
                label=self._label,
                source_nodes=self._source_nodes,
                inputs=self._inputs,
                error=exc,
                elapsed_ms=elapsed_ms,
                parent_id=self._parent_id,
            )
            return False
        if self._status == "degraded":
            self._recorder.record_degraded(
                self._node_id,
                label=self._label,
                source_nodes=self._source_nodes,
                inputs=self._inputs,
                outputs=self._outputs,
                warnings=self._warnings,
                elapsed_ms=elapsed_ms,
                parent_id=self._parent_id,
            )
        else:
            self._recorder.record_completed(
                self._node_id,
                label=self._label,
                source_nodes=self._source_nodes,
                inputs=self._inputs,
                outputs=self._outputs,
                elapsed_ms=elapsed_ms,
                parent_id=self._parent_id,
            )
        return False


class TraceRecorder:
    """Own ordered trace records and optionally stream completed records."""

    def __init__(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
        on_record: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._metadata = _sanitize_mapping(metadata or {})
        self._on_record = on_record
        self._records: list[TraceNodeRecord] = []
        self._node_ids: set[str] = set()
        self._outcome: dict[str, object] = {"status": "running"}
        self._warnings: list[str] = []

    def update_metadata(self, **values: object) -> None:
        self._metadata.update(_sanitize_mapping(values))

    def record_completed(
        self,
        node_id: str,
        *,
        label: str,
        source_nodes: Sequence[str],
        inputs: Mapping[str, object],
        outputs: Mapping[str, object],
        elapsed_ms: float,
        parent_id: str | None = None,
    ) -> None:
        self._append(
            node_id,
            label=label,
            source_nodes=source_nodes,
            inputs=inputs,
            outputs=outputs,
            status="completed",
            elapsed_ms=elapsed_ms,
            parent_id=parent_id,
        )

    def record_degraded(
        self,
        node_id: str,
        *,
        label: str,
        source_nodes: Sequence[str],
        inputs: Mapping[str, object],
        outputs: Mapping[str, object],
        warnings: Sequence[str],
        elapsed_ms: float,
        parent_id: str | None = None,
    ) -> None:
        self._append(
            node_id,
            label=label,
            source_nodes=source_nodes,
            inputs=inputs,
            outputs=outputs,
            status="degraded",
            elapsed_ms=elapsed_ms,
            parent_id=parent_id,
            warnings=warnings,
        )

    def record_skipped(
        self,
        node_id: str,
        *,
        label: str,
        source_nodes: Sequence[str],
        inputs: Mapping[str, object],
        reason: str,
        parent_id: str | None = None,
    ) -> None:
        self._append(
            node_id,
            label=label,
            source_nodes=source_nodes,
            inputs=inputs,
            outputs={"reason": reason},
            status="skipped",
            elapsed_ms=0.0,
            parent_id=parent_id,
        )

    def record_error(
        self,
        node_id: str,
        *,
        label: str,
        source_nodes: Sequence[str],
        inputs: Mapping[str, object],
        error: BaseException,
        elapsed_ms: float,
        parent_id: str | None = None,
    ) -> None:
        self._append(
            node_id,
            label=label,
            source_nodes=source_nodes,
            inputs=inputs,
            outputs={},
            status="error",
            elapsed_ms=elapsed_ms,
            parent_id=parent_id,
            error={
                "type": type(error).__name__,
                "message": str(error)[:_ERROR_MESSAGE_LIMIT],
            },
        )

    def node(
        self,
        node_id: str,
        *,
        label: str,
        source_nodes: Sequence[str],
        inputs: Mapping[str, object],
        parent_id: str | None = None,
    ) -> _TraceNodeContext:
        return _TraceNodeContext(
            self,
            node_id,
            label=label,
            source_nodes=tuple(source_nodes),
            inputs=inputs,
            parent_id=parent_id,
        )

    def set_outcome(self, outcome: Mapping[str, object]) -> None:
        self._outcome = _sanitize_mapping(outcome)

    def add_warning(self, warning: str) -> None:
        self._warnings.append(str(warning)[:_ERROR_MESSAGE_LIMIT])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "metadata": dict(self._metadata),
            "nodes": [record.to_dict() for record in self._records],
            "outcome": dict(self._outcome),
            "warnings": list(self._warnings),
        }

    def _append(
        self,
        node_id: str,
        *,
        label: str,
        source_nodes: Sequence[str],
        inputs: Mapping[str, object],
        outputs: Mapping[str, object],
        status: str,
        elapsed_ms: float,
        parent_id: str | None,
        warnings: Sequence[str] = (),
        error: Mapping[str, str] | None = None,
    ) -> None:
        if not node_id or not node_id.strip():
            raise ValueError("trace node_id must not be empty")
        if node_id in self._node_ids:
            raise ValueError(f"duplicate trace node_id: {node_id}")
        if status not in TRACE_STATUSES:
            raise ValueError(f"invalid trace status: {status}")
        if not math.isfinite(float(elapsed_ms)) or elapsed_ms < 0:
            raise ValueError("trace elapsed_ms must be finite and non-negative")
        record = TraceNodeRecord(
            sequence=len(self._records) + 1,
            node_id=node_id,
            label=str(label),
            parent_id=parent_id,
            source_nodes=tuple(str(item) for item in source_nodes),
            status=status,
            elapsed_ms=round(float(elapsed_ms), 6),
            inputs=_sanitize_mapping(inputs),
            outputs=_sanitize_mapping(outputs),
            warnings=tuple(str(item)[:_ERROR_MESSAGE_LIMIT] for item in warnings),
            error=None if error is None else _sanitize_error(error),
        )
        self._node_ids.add(node_id)
        self._records.append(record)
        if self._on_record is not None:
            try:
                self._on_record(record.to_dict())
            except Exception as exc:
                self._warnings.append(
                    f"trace callback {type(exc).__name__}: {exc}"[
                        :_ERROR_MESSAGE_LIMIT
                    ]
                )


def _sanitize_error(error: Mapping[str, str]) -> dict[str, str]:
    return {
        "type": str(error.get("type", "Error"))[:100],
        "message": str(error.get("message", ""))[:_ERROR_MESSAGE_LIMIT],
    }


def _sanitize_mapping(value: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, item in value.items():
        name = str(key)
        if name.lower() in FORBIDDEN_TRACE_KEYS:
            raise TraceSafetyError(f"forbidden trace key: {name}")
        sanitized[name] = _sanitize_trace_value(item)
    return sanitized


def _sanitize_trace_value(value: Any) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TraceSafetyError("trace numbers must be finite")
        return value
    if isinstance(value, str):
        return value[:_STRING_LIMIT]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TraceSafetyError("binary payloads are not allowed in traces")
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize_mapping(asdict(value))
    if isinstance(value, (set, frozenset)):
        return _sanitize_collection(sorted(value, key=repr))
    if isinstance(value, Sequence):
        return _sanitize_collection(list(value))
    return {"type": type(value).__name__, "omitted": True}


def _sanitize_collection(values: Sequence[object]) -> object:
    sanitized = [_sanitize_trace_value(item) for item in values]
    if len(sanitized) <= _COLLECTION_PREVIEW_LIMIT:
        return sanitized
    canonical = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "count": len(sanitized),
        "preview": sanitized[:_COLLECTION_PREVIEW_LIMIT],
        "sha256": sha256(canonical).hexdigest(),
        "truncated": True,
    }


def write_trace_atomic(
    path: str | Path,
    document: Mapping[str, object],
) -> None:
    """Persist one UTF-8 trace through an atomic same-directory replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def summarize_request(request: object) -> dict[str, object]:
    """Return the prompt-conditioned request fields needed for an audit."""

    prompt = str(getattr(request, "prompt", ""))
    plan = getattr(request, "physics_plan", None)
    return {
        "video_path": str(getattr(request, "video_path", "")),
        "prompt": prompt,
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "explicit_plan": summarize_plan(plan),
    }


def summarize_plan(plan: object | None) -> dict[str, object]:
    if plan is None:
        return {
            "objects": [],
            "expected_events": [],
            "relation_count": 0,
            "constraint_count": 0,
            "source": "none",
            "confidence": 0.0,
            "fallback_used": False,
            "model": None,
        }
    metadata = getattr(plan, "planner_metadata", None)
    return {
        "objects": list(getattr(plan, "objects", ())),
        "expected_events": list(getattr(plan, "expected_events", ())),
        "relation_count": len(getattr(plan, "relations", ())),
        "constraint_count": len(getattr(plan, "physics_constraints", ())),
        "source": str(getattr(metadata, "source", "unknown")),
        "confidence": float(getattr(metadata, "confidence", 0.0)),
        "fallback_used": bool(getattr(metadata, "fallback_used", False)),
        "model": getattr(metadata, "model", None),
    }


def summarize_graph(graph: object | None) -> dict[str, object]:
    if graph is None:
        return {"source": None, "node_count": 0, "nodes": []}
    nodes = tuple(getattr(graph, "nodes", ()))
    return {
        "source": str(getattr(graph, "source", "unknown")),
        "node_count": len(nodes),
        "nodes": [summarize_question_node(node) for node in nodes],
    }


def summarize_question_node(node: object) -> dict[str, object]:
    return {
        "id": str(getattr(node, "id", "")),
        "category": str(getattr(node, "category", "")),
        "question": str(getattr(node, "question", "")),
        "parent_ids": list(getattr(node, "parent_ids", ())),
        "target_objects": list(getattr(node, "target_objects", ())),
        "expected_events": list(getattr(node, "expected_events", ())),
        "rule_ids": list(getattr(node, "rule_ids", ())),
    }


def summarize_states(states: Sequence[object]) -> dict[str, object]:
    frames = sorted({int(getattr(state, "frame", 0)) for state in states})
    objects = sorted({str(getattr(state, "object", "")) for state in states})
    return {
        "state_count": len(states),
        "frame_count": len(frames),
        "frame_range": [] if not frames else [frames[0], frames[-1]],
        "objects": objects,
        "visible_state_count": sum(bool(getattr(state, "visible", False)) for state in states),
    }


def summarize_tracks(tracks: Sequence[object]) -> dict[str, object]:
    return {
        "track_count": len(tracks),
        "tracks": [
            {
                "track_id": str(getattr(track, "track_id", "")),
                "object": str(getattr(track, "object", "")),
                "state_count": len(getattr(track, "states", ())),
                "visible_state_count": sum(
                    bool(getattr(state, "visible", False))
                    for state in getattr(track, "states", ())
                ),
            }
            for track in tracks
        ],
    }


def summarize_events(events: Sequence[object]) -> dict[str, object]:
    return {
        "event_count": len(events),
        "events": [
            {
                "event_type": str(getattr(event, "event_type", "")),
                "object": str(getattr(event, "object", "")),
                "track_id": str(getattr(event, "track_id", "")),
                "start_frame": int(getattr(event, "start_frame", 0)),
                "peak_frame": int(getattr(event, "peak_frame", 0)),
                "end_frame": int(getattr(event, "end_frame", 0)),
                "confidence": float(getattr(event, "confidence", 0.0)),
            }
            for event in events
        ],
    }


def summarize_candidates(candidates: Sequence[object]) -> dict[str, object]:
    return {
        "candidate_count": len(candidates),
        "candidates": [
            {
                "index": index,
                "object": str(getattr(candidate, "object", "")),
                "track_id": str(getattr(candidate, "track_id", "")),
                "category": str(getattr(candidate, "category", "")),
                "start_frame": int(getattr(candidate, "start_frame", 0)),
                "peak_frame": int(getattr(candidate, "peak_frame", 0)),
                "end_frame": int(getattr(candidate, "end_frame", 0)),
                "detector_score": float(getattr(candidate, "detector_score", 0.0)),
                "evidence_frames": list(getattr(candidate, "evidence_frames", ())),
            }
            for index, candidate in enumerate(candidates)
        ],
    }


def summarize_mechanics(
    results: Sequence[object], summary: object | None
) -> dict[str, object]:
    return {
        "result_count": len(results),
        "results": [
            {
                "evaluator": str(getattr(result, "evaluator", "")),
                "applicability": str(getattr(result, "applicability", "")),
                "score": getattr(result, "score", None),
                "is_plausible": getattr(result, "is_plausible", None),
                "reason": str(getattr(result, "reason", "")),
            }
            for result in results
        ],
        "summary": None
        if summary is None
        else {
            "score": getattr(summary, "score", None),
            "coverage": float(getattr(summary, "coverage", 0.0)),
            "applicable": int(getattr(summary, "applicable", 0)),
            "not_applicable": int(getattr(summary, "not_applicable", 0)),
            "failed": int(getattr(summary, "failed", 0)),
        },
    }


def summarize_checklist(
    results: Sequence[object], summary: object | None
) -> dict[str, object]:
    return {
        "dimension_count": len(results),
        "dimensions": [
            {
                "dimension": str(getattr(result, "dimension", "")),
                "status": str(getattr(result, "status", "")),
                "score": getattr(result, "score", None),
                "confidence": float(getattr(result, "confidence", 0.0)),
                "reason": str(getattr(result, "reason", "")),
                "critical_frames": list(getattr(result, "critical_frames", ())),
            }
            for result in results
        ],
        "summary": None
        if summary is None
        else {
            "score": getattr(summary, "score", None),
            "coverage": float(getattr(summary, "coverage", 0.0)),
            "passed": int(getattr(summary, "passed", 0)),
            "failed": int(getattr(summary, "failed", 0)),
            "unknown": int(getattr(summary, "unknown", 0)),
        },
    }


def summarize_keyframes(keyframes: Mapping[int, Sequence[int]]) -> dict[str, object]:
    return {
        "candidate_count": len(keyframes),
        "candidate_keyframes": [
            {"index": int(index), "frames": [int(frame) for frame in frames]}
            for index, frames in sorted(keyframes.items())
        ],
    }


def summarize_node_result(result: object) -> dict[str, object]:
    return {
        "node_id": str(getattr(result, "node_id", "")),
        "category": str(getattr(result, "category", "")),
        "status": str(getattr(result, "status", "")),
        "direct_score": getattr(result, "direct_score", None),
        "confidence": float(getattr(result, "confidence", 0.0)),
        "reason": str(getattr(result, "reason", "")),
        "verifier": str(getattr(result, "verifier", "")),
        "critical_frames": list(getattr(result, "critical_frames", ())),
        "blocked_by": list(getattr(result, "blocked_by", ())),
    }


def summarize_reviews(
    candidates: Sequence[object], reviews: Mapping[int, object | None]
) -> dict[str, object]:
    rows = []
    for index, candidate in enumerate(candidates):
        review = reviews.get(index)
        rows.append(
            {
                "index": index,
                "category": str(getattr(candidate, "category", "")),
                "key": f"{getattr(candidate, 'track_id', '')}:{index}",
                "review_status": "unavailable"
                if review is None
                else str(getattr(review, "claim_status", "uncertain")),
                "score": None if review is None else float(getattr(review, "score", 0.0)),
                "model": None if review is None else str(getattr(review, "model", "")),
                "reason": None if review is None else str(getattr(review, "reason", "")),
            }
        )
    return {"review_count": sum(reviews.get(i) is not None for i in range(len(candidates))), "reviews": rows}


def summarize_report(report: object) -> dict[str, object]:
    return {
        "decision": str(getattr(report, "decision", "unknown")),
        "physics_score": float(getattr(report, "physics_score", 0.0)),
        "confidence": float(getattr(report, "confidence", 0.0)),
        "coverage": float(getattr(report, "coverage", 0.0)),
        "violations": [
            {
                "object": str(getattr(item, "object", "")),
                "category": str(getattr(item, "category", "")),
                "start_frame": int(getattr(item, "start_frame", 0)),
                "peak_frame": int(getattr(item, "peak_frame", 0)),
                "end_frame": int(getattr(item, "end_frame", 0)),
            }
            for item in getattr(report, "violations", ())
        ],
    }


def build_fusion_audit(
    config: FusionConfig,
    report: CriticReport,
) -> dict[str, object]:
    """Expose and independently recompute every family-fusion arithmetic term."""

    weights = {
        "rules": config.rule_family_weight,
        "pqsg": config.pqsg_family_weight,
        "checklist": config.checklist_family_weight,
        "mechanics": config.mechanics_family_weight,
        "vlm": config.vlm_family_weight,
    }
    bundles = {bundle.family: bundle for bundle in report.evidence_bundles}
    families: list[dict[str, object]] = []
    total_effective = 0.0
    total_contribution = 0.0
    total_configured = sum(weights.values())
    coverage_numerator = 0.0
    available_count = 0
    for family in EVIDENCE_FAMILIES:
        configured_weight = weights[family]
        bundle = bundles.get(family)
        if bundle is None:
            source = None
            status = "unknown"
            score = None
            coverage = 0.0
            confidence = 0.0
        else:
            source = bundle.source
            status = bundle.status
            score = bundle.score
            coverage = bundle.coverage
            confidence = bundle.confidence
        coverage_numerator += configured_weight * coverage
        if status == "available" and score is not None and configured_weight > 0:
            effective_weight = configured_weight * coverage * confidence
            weighted_contribution = score * effective_weight
            available_count += 1
        else:
            effective_weight = 0.0
            weighted_contribution = 0.0
        total_effective += effective_weight
        total_contribution += weighted_contribution
        families.append(
            {
                "family": family,
                "source": source,
                "status": status,
                "score": score,
                "configured_weight": configured_weight,
                "coverage": coverage,
                "confidence": confidence,
                "effective_weight": effective_weight,
                "weighted_contribution": weighted_contribution,
            }
        )
    score_before_hard = (
        total_contribution / total_effective if total_effective else 0.5
    )
    weighted_coverage = (
        coverage_numerator / total_configured if total_configured else 0.0
    )
    confidence_before_hard = (
        total_effective / total_configured if total_configured else 0.0
    )
    if available_count == 0 or weighted_coverage < config.minimum_coverage:
        decision_before_hard = "unknown"
    elif score_before_hard < config.physical_score_threshold:
        decision_before_hard = "violation"
    else:
        decision_before_hard = "physical"
    hard_violation = bool(report.violations)
    return {
        "families": families,
        "total_configured_weight": total_configured,
        "total_effective_weight": total_effective,
        "total_weighted_contribution": total_contribution,
        "score_before_hard_violation": score_before_hard,
        "confidence_before_hard_violation": confidence_before_hard,
        "weighted_coverage": weighted_coverage,
        "physical_score_threshold": config.physical_score_threshold,
        "minimum_coverage": config.minimum_coverage,
        "decision_before_hard_violation": decision_before_hard,
        "hard_violation": hard_violation,
        "hard_violation_count": len(report.violations),
        "hard_violation_score_cap": report.physics_score if hard_violation else None,
        "final_score": report.physics_score,
        "final_confidence": report.confidence,
        "final_coverage": report.coverage,
        "final_decision": report.decision,
    }


def validate_trace(
    document: Mapping[str, object],
    *,
    policy: TraceValidationPolicy = TraceValidationPolicy(),
    tolerance: float = 1e-6,
) -> TraceValidationReport:
    """Independently validate trace structure, privacy and fusion arithmetic."""

    checks: list[TraceValidationCheck] = []

    def add(code: str, passed: bool, message: str, *, level: str = "error") -> None:
        checks.append(
            TraceValidationCheck(
                code=code,
                level=level,
                passed=bool(passed),
                message=message,
            )
        )

    add(
        "schema.version",
        document.get("schema_version") == TRACE_SCHEMA_VERSION,
        f"schema_version must be {TRACE_SCHEMA_VERSION}",
    )
    nodes_value = document.get("nodes")
    nodes = nodes_value if isinstance(nodes_value, list) else []
    add("schema.nodes", isinstance(nodes_value, list), "nodes must be an array")
    node_maps = [node for node in nodes if isinstance(node, Mapping)]
    add(
        "schema.node_types",
        len(node_maps) == len(nodes),
        "every node must be an object",
    )
    node_ids = [str(node.get("node_id", "")) for node in node_maps]
    sequences = [node.get("sequence") for node in node_maps]
    add(
        "graph.unique_nodes",
        len(node_ids) == len(set(node_ids)) and all(node_ids),
        "node IDs must be non-empty and unique",
    )
    add(
        "graph.sequence",
        sequences == list(range(1, len(sequences) + 1)),
        "node sequence numbers must be contiguous and monotonic",
    )
    fixed = [node_id for node_id in node_ids if not node_id.startswith("pqsg_node.")]
    add(
        "graph.required_stages",
        fixed == list(_FIXED_STAGE_IDS),
        "fixed pipeline stages must be present in execution order",
    )
    position = {node_id: index for index, node_id in enumerate(node_ids)}
    legal_references = True
    for index, node in enumerate(node_maps):
        references = list(node.get("source_nodes", []))
        parent_id = node.get("parent_id")
        if parent_id is not None:
            references.append(parent_id)
        if any(
            str(reference) not in position or position[str(reference)] >= index
            for reference in references
        ):
            legal_references = False
            break
    add(
        "graph.references",
        legal_references,
        "parent and source nodes must reference earlier records",
    )
    skipped_reasons = all(
        node.get("status") != "skipped"
        or (
            isinstance(node.get("outputs"), Mapping)
            and bool(node["outputs"].get("reason"))
        )
        for node in node_maps
    )
    add(
        "graph.skipped_reason",
        skipped_reasons,
        "every skipped node must declare a reason",
    )
    first_error = next(
        (index for index, node in enumerate(node_maps) if node.get("status") == "error"),
        None,
    )
    no_completion_after_error = first_error is None or not any(
        node.get("status") == "completed" for node in node_maps[first_error + 1 :]
    )
    add(
        "graph.error_terminal",
        no_completion_after_error,
        "no completed stage may follow an error stage",
    )
    question_graph_node = next(
        (node for node in node_maps if node.get("node_id") == "question_graph"),
        None,
    )
    expected_pqsg_ids: set[str] = set()
    if question_graph_node is not None and isinstance(
        question_graph_node.get("outputs"), Mapping
    ):
        graph_nodes = question_graph_node["outputs"].get("nodes", [])
        if isinstance(graph_nodes, list):
            expected_pqsg_ids = {
                f"pqsg_node.{item.get('id')}"
                for item in graph_nodes
                if isinstance(item, Mapping) and item.get("id")
            }
    actual_pqsg_ids = {
        node_id for node_id in node_ids if node_id.startswith("pqsg_node.")
    }
    add(
        "pqsg.node_coverage",
        expected_pqsg_ids == actual_pqsg_ids,
        "every generated question-graph node must have exactly one child trace",
    )
    sensitive_path = _find_sensitive_path(document)
    add(
        "privacy.forbidden_keys",
        sensitive_path is None,
        "trace contains no forbidden secret or raw-payload keys"
        if sensitive_path is None
        else f"forbidden trace field at {sensitive_path}",
    )

    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    detector = metadata.get("detector")
    detector = detector if isinstance(detector, Mapping) else {}
    planner = metadata.get("planner")
    planner = planner if isinstance(planner, Mapping) else {}
    if policy.require_sam2:
        add(
            "policy.sam2",
            detector.get("sam2_used") is True,
            "strict policy requires SAM2",
        )
    if policy.require_model_planner:
        add(
            "policy.model_planner",
            planner.get("source") == "model",
            "strict policy requires a model Planner",
        )
    degraded = [node for node in node_maps if node.get("status") == "degraded"]
    fallback_count = metadata.get("provider_fallback_count", 0)
    if policy.fail_on_provider_fallback:
        add(
            "policy.provider_fallback",
            not degraded and fallback_count == 0,
            "strict policy forbids provider fallback",
        )

    fusion_node = next(
        (node for node in node_maps if node.get("node_id") == "evidence_fusion"),
        None,
    )
    final_node = next(
        (node for node in node_maps if node.get("node_id") == "final_report"),
        None,
    )
    if fusion_node is None or not isinstance(fusion_node.get("outputs"), Mapping):
        add("fusion.present", False, "evidence_fusion outputs are required")
    else:
        add("fusion.present", True, "evidence_fusion outputs are present")
        _validate_fusion_outputs(
            fusion_node["outputs"],
            final_node.get("outputs") if final_node is not None else None,
            tolerance=tolerance,
            add=add,
        )

    candidate_node = next(
        (node for node in node_maps if node.get("node_id") == "candidate_fusion"),
        None,
    )
    filtered_ok = True
    if candidate_node is not None and isinstance(candidate_node.get("outputs"), Mapping):
        candidates = candidate_node["outputs"].get("candidates", [])
        if isinstance(candidates, list):
            filtered_ok = not any(
                isinstance(candidate, Mapping)
                and candidate.get("review_status") in {"rejected", "uncertain"}
                and candidate.get("retained") is True
                for candidate in candidates
            )
    add(
        "fusion.review_filter",
        filtered_ok,
        "rejected and uncertain VLM candidates must not be retained",
    )
    return TraceValidationReport(checks=tuple(checks))


def _validate_fusion_outputs(
    outputs: Mapping[str, object],
    final_outputs: object,
    *,
    tolerance: float,
    add: Callable[..., None],
) -> None:
    family_value = outputs.get("families")
    families = family_value if isinstance(family_value, list) else []
    add(
        "fusion.families",
        len(families) == len(EVIDENCE_FAMILIES),
        "fusion must contain all five evidence families",
    )
    effective_ok = True
    contribution_ok = True
    calculated_effective = 0.0
    calculated_contribution = 0.0
    coverage_numerator = 0.0
    total_configured = 0.0
    available_count = 0
    for row in families:
        if not isinstance(row, Mapping):
            effective_ok = contribution_ok = False
            continue
        try:
            configured = float(row["configured_weight"])
            coverage = float(row["coverage"])
            confidence = float(row["confidence"])
            stored_effective = float(row["effective_weight"])
            stored_contribution = float(row["weighted_contribution"])
        except (KeyError, TypeError, ValueError):
            effective_ok = contribution_ok = False
            continue
        total_configured += configured
        coverage_numerator += configured * coverage
        expected_effective = (
            configured * coverage * confidence
            if row.get("status") == "available" and row.get("score") is not None
            else 0.0
        )
        if row.get("status") == "available" and row.get("score") is not None:
            available_count += 1
        score = 0.0 if row.get("score") is None else float(row["score"])
        expected_contribution = score * expected_effective
        effective_ok &= _close(stored_effective, expected_effective, tolerance)
        contribution_ok &= _close(
            stored_contribution, expected_contribution, tolerance
        )
        calculated_effective += expected_effective
        calculated_contribution += expected_contribution
    add(
        "fusion.effective_weight",
        effective_ok,
        "effective weights equal configured_weight × coverage × confidence",
    )
    add(
        "fusion.weighted_contribution",
        contribution_ok,
        "weighted contributions equal score × effective_weight",
    )
    expected_score = (
        calculated_contribution / calculated_effective
        if calculated_effective
        else 0.5
    )
    expected_coverage = (
        coverage_numerator / total_configured if total_configured else 0.0
    )
    aggregate_ok = all(
        (
            _close(outputs.get("total_effective_weight"), calculated_effective, tolerance),
            _close(
                outputs.get("total_weighted_contribution"),
                calculated_contribution,
                tolerance,
            ),
            _close(outputs.get("score_before_hard_violation"), expected_score, tolerance),
            _close(outputs.get("weighted_coverage"), expected_coverage, tolerance),
        )
    )
    add(
        "fusion.aggregates",
        aggregate_ok,
        "fusion aggregate totals and pre-hard score are reproducible",
    )
    try:
        minimum_coverage = float(outputs["minimum_coverage"])
        physical_threshold = float(outputs["physical_score_threshold"])
    except (KeyError, TypeError, ValueError):
        minimum_coverage = math.inf
        physical_threshold = math.inf
    if available_count == 0 or expected_coverage < minimum_coverage:
        expected_pre_decision = "unknown"
    elif expected_score < physical_threshold:
        expected_pre_decision = "violation"
    else:
        expected_pre_decision = "physical"
    expected_final_decision = (
        "violation" if outputs.get("hard_violation") is True else expected_pre_decision
    )
    decision_ok = (
        outputs.get("decision_before_hard_violation") == expected_pre_decision
        and outputs.get("final_decision") == expected_final_decision
    )
    add(
        "fusion.decision",
        decision_ok,
        "final decision must be independently implied by coverage, score threshold and hard violations",
    )
    if isinstance(final_outputs, Mapping):
        final_ok = (
            outputs.get("final_decision") == final_outputs.get("decision")
            and _close(
                outputs.get("final_score"), final_outputs.get("physics_score"), tolerance
            )
            and _close(
                outputs.get("final_confidence"), final_outputs.get("confidence"), tolerance
            )
            and _close(
                outputs.get("final_coverage"), final_outputs.get("coverage"), tolerance
            )
        )
    else:
        final_ok = False
    add(
        "fusion.final_report",
        final_ok,
        "fusion final values match the public report",
    )


def _close(left: object, right: object, tolerance: float) -> bool:
    try:
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
    except (TypeError, ValueError):
        return False


def _find_sensitive_path(value: object, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            child_path = f"{path}.{name}"
            if name.lower() in FORBIDDEN_TRACE_KEYS:
                return child_path
            found = _find_sensitive_path(item, child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_sensitive_path(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None
