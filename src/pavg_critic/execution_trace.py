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


class TraceSafetyError(ValueError):
    """Raised when trace data could expose a secret or binary payload."""


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
            self._on_record(record.to_dict())


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

