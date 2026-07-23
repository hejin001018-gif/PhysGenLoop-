"""Atomic run/sample/attempt artifacts for the single V2 runtime."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARTIFACTS_SCHEMA_VERSION = "v2-artifacts/2.0"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_STATES = (
    "CREATED",
    "PREFLIGHT_FAILED",
    "GENERATING",
    "CRITIC_RUNNING",
    "DECISION_READY",
    "EXECUTING",
    "RE_EVALUATING",
    "ACCEPTED",
    "REJECTED",
    "MAX_ROUNDS",
    "EVALUATION_FAILED",
    "EXECUTION_FAILED",
)
TERMINAL_STATES = frozenset(
    {
        "ACCEPTED",
        "REJECTED",
        "MAX_ROUNDS",
        "EVALUATION_FAILED",
        "EXECUTION_FAILED",
        "PREFLIGHT_FAILED",
    }
)
RETRYABLE_STATES = frozenset(
    {"EVALUATION_FAILED", "EXECUTION_FAILED", "PREFLIGHT_FAILED"}
)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, payload: Any) -> None:
    _atomic_write(
        Path(path),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def append_jsonl(path: str | Path, record: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_schema(record: dict[str, Any], schema_name: str) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("jsonschema is required for strict artifact validation") from exc
    schema_path = _PROJECT_ROOT / "schemas" / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(record, schema)


@dataclass(frozen=True)
class SampleStatus:
    sample_id: str
    state: str
    round_index: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    schema_version: str = ARTIFACTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.state not in SAMPLE_STATES:
            raise ValueError(f"invalid sample state: {self.state!r}")

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "state": self.state,
            "round_index": self.round_index,
            "detail": dict(self.detail),
        }


class RunArtifacts:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._active_attempts: dict[str, str] = {}

    def create_run_manifest(self, payload: dict[str, Any]) -> None:
        destination = self.run_dir / "run_manifest.json"
        if destination.exists():
            raise FileExistsError(f"run manifest already exists: {destination}")
        write_json(destination, payload)

    def write_run_manifest(self, payload: dict[str, Any]) -> None:
        self.create_run_manifest(payload)

    def read_run_manifest(self) -> dict[str, Any]:
        return json.loads((self.run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    def write_run_status(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "run_status.json", payload)

    def write_summary(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "summary.json", payload)

    def sample_root(self, sample_id: str) -> Path:
        root = self.run_dir / sample_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def start_attempt(self, sample_id: str, *, reason: str) -> tuple[str, Path]:
        root = self.sample_root(sample_id)
        attempts = root / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        indices = []
        for path in attempts.glob("attempt_*"):
            try:
                indices.append(int(path.name.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        attempt_id = f"attempt_{(max(indices, default=0) + 1):04d}"
        destination = attempts / attempt_id
        destination.mkdir(parents=False, exist_ok=False)
        self._active_attempts[sample_id] = attempt_id
        write_json(
            root / "active_attempt.json",
            {"attempt_id": attempt_id, "reason": reason},
        )
        return attempt_id, destination

    def active_attempt_id(self, sample_id: str) -> str | None:
        if sample_id in self._active_attempts:
            return self._active_attempts[sample_id]
        path = self.run_dir / sample_id / "active_attempt.json"
        if not path.exists():
            return None
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))["attempt_id"])
        except Exception:  # noqa: BLE001
            return None

    def sample_dir(self, sample_id: str) -> Path:
        attempt_id = self.active_attempt_id(sample_id)
        if attempt_id is None:
            return self.sample_root(sample_id)
        return self.sample_root(sample_id) / "attempts" / attempt_id

    def set_status(self, status: SampleStatus) -> None:
        root = self.sample_root(status.sample_id)
        payload = status.to_dict()
        attempt_id = self.active_attempt_id(status.sample_id)
        if attempt_id is not None:
            payload["active_attempt"] = attempt_id
            if status.terminal:
                payload["authoritative_attempt"] = attempt_id
        write_json(root / "sample_status.json", payload)
        append_jsonl(root / "sample_status_history.jsonl", payload)

    def read_status(self, sample_id: str) -> SampleStatus | None:
        path = self.run_dir / sample_id / "sample_status.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return SampleStatus(
                sample_id=str(raw["sample_id"]),
                state=str(raw["state"]),
                round_index=int(raw.get("round_index", 0)),
                detail=dict(raw.get("detail", {})),
            )
        except Exception:  # noqa: BLE001
            return None

    def candidate_dir(self, sample_id: str, candidate_id: str) -> Path:
        destination = self.sample_dir(sample_id) / candidate_id
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def write_critic_report(self, sample_id: str, candidate_id: str, payload: dict[str, Any]) -> None:
        write_json(self.candidate_dir(sample_id, candidate_id) / "critic_report.json", payload)

    def write_raw_payload(self, sample_id: str, candidate_id: str, raw: dict[str, Any], error: str) -> None:
        destination = self.candidate_dir(sample_id, candidate_id)
        write_json(destination / "critic_payload_raw.json", raw)
        write_json(destination / "critic_roundtrip_error.json", {"error": error})

    def write_mask_manifest(self, sample_id: str, candidate_id: str, payload: dict[str, Any]) -> Path:
        destination = self.candidate_dir(sample_id, candidate_id) / "mask_manifest.json"
        write_json(destination, payload)
        return destination

    def write_decision(self, sample_id: str, candidate_id: str, payload: dict[str, Any]) -> None:
        write_json(self.candidate_dir(sample_id, candidate_id) / "repair_decision.json", payload)

    def append_repair_trace(self, sample_id: str, record: dict[str, Any]) -> None:
        _validate_schema(record, "loop_trace_v2.schema.json")
        append_jsonl(self.sample_dir(sample_id) / "repair_trace.jsonl", record)

    def write_loop_result(self, sample_id: str, payload: dict[str, Any]) -> None:
        write_json(self.sample_dir(sample_id) / "loop_result.json", payload)

    def append_trial(self, sample_id: str, record: dict[str, Any]) -> None:
        _validate_schema(record, "wan_repair_trial_v3.schema.json")
        append_jsonl(self.sample_dir(sample_id) / "trials.jsonl", record)

    def append_resource_metrics(self, sample_id: str, record: dict[str, Any]) -> None:
        append_jsonl(self.sample_dir(sample_id) / "resource_metrics.jsonl", record)

    def write_owner(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "vllm.owner.json", payload)


def pending_samples(
    run_dir: str | Path,
    sample_ids: list[str],
    *,
    retry_failed: bool = False,
) -> list[str]:
    artifacts = RunArtifacts(run_dir)
    pending: list[str] = []
    for sample_id in sample_ids:
        status = artifacts.read_status(sample_id)
        if status is None or not status.terminal:
            pending.append(sample_id)
        elif retry_failed and status.state in RETRYABLE_STATES:
            pending.append(sample_id)
    return pending


def rebuild_summary(run_dir: str | Path, sample_ids: list[str]) -> dict[str, Any]:
    root = Path(run_dir)
    counts = {state.lower(): 0 for state in TERMINAL_STATES}
    pending: list[str] = []
    results: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        status_path = root / sample_id / "sample_status.json"
        if not status_path.exists():
            pending.append(sample_id)
            continue
        raw_status = json.loads(status_path.read_text(encoding="utf-8"))
        state = str(raw_status.get("state", ""))
        if state in TERMINAL_STATES:
            counts[state.lower()] += 1
        else:
            pending.append(sample_id)
        attempt_id = raw_status.get("authoritative_attempt")
        if attempt_id:
            result_path = root / sample_id / "attempts" / str(attempt_id) / "loop_result.json"
            if result_path.exists():
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
    return {
        "total_samples": len(sample_ids),
        **counts,
        "pending": len(pending),
        "pending_sample_ids": pending,
        "results": results,
    }
