"""Canonical, versioned contracts for the Learning Repair Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .base_contracts import (
    ACTION_ORDER,
    PolicyPrediction,
    RepairAction,
    RepairContext,
    RepairExample,
    LegacyRepairDecision,
    REPAIR_SCHEMA_VERSION,
)


DECISION_SCHEMA_VERSION = "learning-repair-decision/2.0"
TRIAL_SCHEMA_VERSION = "repair-trial/1.0"
TARGET_SCHEMA_VERSION = "repair-target/1.0"
RUN_SCHEMA_VERSION = "learning-repair-run/1.0"
DOMAINS = frozenset({"blender", "hunyuan", "fake"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping or expose to_dict()")
    return dict(value)


def _score(value: float, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def _action_float_map(
    values: Mapping[RepairAction | str, float],
    *,
    name: str,
    normalize: bool,
) -> dict[RepairAction, float]:
    converted = {RepairAction(key): float(value) for key, value in values.items()}
    if set(converted) != set(ACTION_ORDER):
        raise ValueError(f"{name} must contain every repair action")
    if normalize:
        if any(value < 0.0 for value in converted.values()):
            raise ValueError(f"{name} values must be non-negative")
        total = sum(converted.values())
        if total <= 0.0:
            raise ValueError(f"{name} must have a positive sum")
        converted = {action: value / total for action, value in converted.items()}
    return converted


@dataclass(frozen=True)
class CandidateRecord:
    """Serializable snapshot of a candidate without changing GeneratedCandidate."""

    candidate_id: str
    video_path: str
    prompt: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.video_path.strip():
            raise ValueError("candidate_id and video_path must not be empty")
        if not isinstance(self.metadata, dict):
            raise ValueError("candidate metadata must be an object")

    @classmethod
    def from_candidate(cls, candidate: Any) -> "CandidateRecord":
        return cls(
            candidate_id=str(candidate.candidate_id),
            video_path=str(candidate.video_path),
            prompt=str(candidate.prompt),
            seed=int(candidate.seed),
            metadata=dict(getattr(candidate, "metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "video_path": self.video_path,
            "prompt": self.prompt,
            "seed": self.seed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidateRecord":
        return cls(
            candidate_id=str(raw["candidate_id"]),
            video_path=str(raw["video_path"]),
            prompt=str(raw.get("prompt", "")),
            seed=int(raw.get("seed", 0)),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class LocalEditTarget:
    """Object/time/mask coordinates required by a local editing backend."""

    parent_candidate_id: str
    objects: tuple[str, ...] = ()
    start_frame: int | None = None
    end_frame: int | None = None
    critical_frames: tuple[int, ...] = ()
    mask_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.parent_candidate_id.strip():
            raise ValueError("parent_candidate_id must not be empty")
        if (self.start_frame is None) != (self.end_frame is None):
            raise ValueError("start_frame and end_frame must be provided together")
        if self.start_frame is not None:
            if self.start_frame < 0 or self.end_frame is None or self.end_frame < self.start_frame:
                raise ValueError("invalid local edit frame interval")
        if any(frame < 0 for frame in self.critical_frames):
            raise ValueError("critical frames must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_candidate_id": self.parent_candidate_id,
            "objects": list(self.objects),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "critical_frames": list(self.critical_frames),
            "mask_uri": self.mask_uri,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "LocalEditTarget | None":
        if raw is None:
            return None
        return cls(
            parent_candidate_id=str(raw["parent_candidate_id"]),
            objects=tuple(str(item) for item in raw.get("objects", ())),
            start_frame=None if raw.get("start_frame") is None else int(raw["start_frame"]),
            end_frame=None if raw.get("end_frame") is None else int(raw["end_frame"]),
            critical_frames=tuple(int(item) for item in raw.get("critical_frames", ())),
            mask_uri=None if raw.get("mask_uri") is None else str(raw["mask_uri"]),
        )


@dataclass(frozen=True)
class RepairDecision:
    """Executor-facing decision with action-specific values and provenance."""

    action: RepairAction
    confidence: float
    instruction: str
    action_probabilities: dict[RepairAction, float]
    per_action_values: dict[RepairAction, float]
    parameters: dict[str, Any] = field(default_factory=dict)
    local_target: LocalEditTarget | None = None
    source: str = "unknown"
    abstained: bool = False
    fallback_reason: str | None = None
    compatibility_id: str = "unknown"
    schema_version: str = DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_SCHEMA_VERSION:
            raise ValueError(f"unsupported decision schema: {self.schema_version!r}")
        object.__setattr__(self, "action", RepairAction(self.action))
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        object.__setattr__(
            self,
            "action_probabilities",
            _action_float_map(
                self.action_probabilities,
                name="action_probabilities",
                normalize=True,
            ),
        )
        object.__setattr__(
            self,
            "per_action_values",
            _action_float_map(
                self.per_action_values,
                name="per_action_values",
                normalize=False,
            ),
        )
        if self.action is RepairAction.LOCAL_EDITING and self.local_target is None:
            raise ValueError("local_editing decisions require local_target")
        if not isinstance(self.parameters, dict):
            raise ValueError("decision parameters must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action": self.action.value,
            "confidence": self.confidence,
            "instruction": self.instruction,
            "action_probabilities": {
                action.value: self.action_probabilities[action] for action in ACTION_ORDER
            },
            "per_action_values": {
                action.value: self.per_action_values[action] for action in ACTION_ORDER
            },
            "parameters": self.parameters,
            "local_target": None if self.local_target is None else self.local_target.to_dict(),
            "source": self.source,
            "abstained": self.abstained,
            "fallback_reason": self.fallback_reason,
            "compatibility_id": self.compatibility_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RepairDecision":
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            action=RepairAction(str(raw["action"])),
            confidence=float(raw["confidence"]),
            instruction=str(raw.get("instruction", "")),
            action_probabilities=dict(raw["action_probabilities"]),
            per_action_values=dict(raw["per_action_values"]),
            parameters=dict(raw.get("parameters", {})),
            local_target=LocalEditTarget.from_dict(raw.get("local_target")),
            source=str(raw.get("source", "unknown")),
            abstained=bool(raw.get("abstained", False)),
            fallback_reason=(
                None if raw.get("fallback_reason") is None else str(raw["fallback_reason"])
            ),
            compatibility_id=str(raw.get("compatibility_id", "unknown")),
        )


# Compatibility name for artifacts and callers that froze the v1 schema before
# the two Learning Repair namespaces were consolidated.
RepairDecisionV1 = RepairDecision


@dataclass(frozen=True)
class ExecutionRequest:
    """One immutable action request consumed by exactly one executor."""

    decision: RepairDecision
    candidate: Any
    critic_report: Any
    prompt: str
    seed: int
    history: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    action: RepairAction
    status: str
    backend_id: str
    candidate: Any | None = None
    next_prompt: str | None = None
    cost: float = 0.0
    latency_seconds: float = 0.0
    terminal: bool = False
    failure_reason: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", RepairAction(self.action))
        if self.status not in {"succeeded", "failed", "rejected"}:
            raise ValueError("execution status must be succeeded, failed, or rejected")
        if self.cost < 0 or self.latency_seconds < 0:
            raise ValueError("execution cost and latency must be non-negative")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed execution requires failure_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "status": self.status,
            "backend_id": self.backend_id,
            "candidate": (
                None
                if self.candidate is None
                else CandidateRecord.from_candidate(self.candidate).to_dict()
            ),
            "next_prompt": self.next_prompt,
            "cost": self.cost,
            "latency_seconds": self.latency_seconds,
            "terminal": self.terminal,
            "failure_reason": self.failure_reason,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ScoreBundle:
    physics: float
    semantic: float | None = None
    original_prompt_semantic: float | None = None
    quality: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "physics", _score(self.physics, "physics"))
        if self.semantic is not None:
            object.__setattr__(self, "semantic", _score(self.semantic, "semantic"))
        if self.original_prompt_semantic is not None:
            object.__setattr__(
                self,
                "original_prompt_semantic",
                _score(
                    self.original_prompt_semantic,
                    "original_prompt_semantic",
                ),
            )
        if self.quality is not None:
            object.__setattr__(self, "quality", _score(self.quality, "quality"))

    def to_dict(self) -> dict[str, float | None]:
        return {
            "physics": self.physics,
            "semantic": self.semantic,
            "original_prompt_semantic": self.original_prompt_semantic,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScoreBundle":
        return cls(
            physics=float(raw["physics"]),
            semantic=None if raw.get("semantic") is None else float(raw["semantic"]),
            original_prompt_semantic=(
                None
                if raw.get("original_prompt_semantic") is None
                else float(raw["original_prompt_semantic"])
            ),
            quality=None if raw.get("quality") is None else float(raw["quality"]),
        )


@dataclass(frozen=True)
class RepairTrialV1:
    """Authoritative before/action/after record used for labels and memory."""

    trial_id: str
    group_id: str
    domain: str
    source_candidate: CandidateRecord
    prompt: str
    critic_before: dict[str, Any]
    decision: RepairDecision
    execution: dict[str, Any]
    before_scores: ScoreBundle
    critic_after: dict[str, Any] | None = None
    after_scores: ScoreBundle | None = None
    successful: bool = False
    failure_reason: str | None = None
    compatibility: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    schema_version: str = TRIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRIAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported trial schema: {self.schema_version!r}")
        if not self.trial_id.strip() or not self.group_id.strip():
            raise ValueError("trial_id and group_id must not be empty")
        if self.domain not in DOMAINS:
            raise ValueError(f"unsupported trial domain: {self.domain!r}")
        if self.successful and self.after_scores is None:
            raise ValueError("successful trial requires after_scores")
        if not self.successful and not self.failure_reason:
            raise ValueError("unsuccessful trial requires failure_reason")

    @property
    def physics_gain(self) -> float | None:
        if self.after_scores is None:
            return None
        return self.after_scores.physics - self.before_scores.physics

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "group_id": self.group_id,
            "domain": self.domain,
            "created_at": self.created_at,
            "source_candidate": self.source_candidate.to_dict(),
            "prompt": self.prompt,
            "critic_before": self.critic_before,
            "decision": self.decision.to_dict(),
            "execution": self.execution,
            "critic_after": self.critic_after,
            "before_scores": self.before_scores.to_dict(),
            "after_scores": None if self.after_scores is None else self.after_scores.to_dict(),
            "physics_gain": self.physics_gain,
            "successful": self.successful,
            "failure_reason": self.failure_reason,
            "compatibility": self.compatibility,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RepairTrialV1":
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            trial_id=str(raw["trial_id"]),
            group_id=str(raw["group_id"]),
            domain=str(raw["domain"]),
            created_at=str(raw.get("created_at", utc_now())),
            source_candidate=CandidateRecord.from_dict(raw["source_candidate"]),
            prompt=str(raw.get("prompt", "")),
            critic_before=_mapping(raw["critic_before"], "critic_before"),
            decision=RepairDecision.from_dict(raw["decision"]),
            execution=dict(raw["execution"]),
            critic_after=(
                None
                if raw.get("critic_after") is None
                else _mapping(raw["critic_after"], "critic_after")
            ),
            before_scores=ScoreBundle.from_dict(raw["before_scores"]),
            after_scores=(
                None
                if raw.get("after_scores") is None
                else ScoreBundle.from_dict(raw["after_scores"])
            ),
            successful=bool(raw.get("successful", False)),
            failure_reason=(
                None if raw.get("failure_reason") is None else str(raw["failure_reason"])
            ),
            compatibility={str(k): str(v) for k, v in dict(raw.get("compatibility", {})).items()},
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class LearningTargetV1:
    """One group-safe policy target derived from actual action trials."""

    sample_id: str
    group_id: str
    domain: str
    critic_report: dict[str, Any]
    context: RepairContext
    target_action: RepairAction
    action_rewards: dict[RepairAction, float | None]
    available_actions: dict[RepairAction, bool]
    source_trial_ids: tuple[str, ...]
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_SCHEMA_VERSION:
            raise ValueError(f"unsupported target schema: {self.schema_version!r}")
        if self.domain not in DOMAINS:
            raise ValueError(f"unsupported target domain: {self.domain!r}")
        object.__setattr__(self, "target_action", RepairAction(self.target_action))
        rewards = {RepairAction(key): (None if value is None else float(value)) for key, value in self.action_rewards.items()}
        availability = {RepairAction(key): bool(value) for key, value in self.available_actions.items()}
        if set(rewards) != set(ACTION_ORDER) or set(availability) != set(ACTION_ORDER):
            raise ValueError("action_rewards and available_actions must cover every action")
        object.__setattr__(self, "action_rewards", rewards)
        object.__setattr__(self, "available_actions", availability)
        if not availability[self.target_action]:
            raise ValueError("target action must be available")
        if self.split not in {None, "train", "validation", "test", "calibration"}:
            raise ValueError("invalid target split")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "domain": self.domain,
            "split": self.split,
            "critic_report": self.critic_report,
            "context": self.context.to_dict(),
            "target_action": self.target_action.value,
            "action_rewards": {action.value: self.action_rewards[action] for action in ACTION_ORDER},
            "available_actions": {action.value: self.available_actions[action] for action in ACTION_ORDER},
            "source_trial_ids": list(self.source_trial_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LearningTargetV1":
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            sample_id=str(raw["sample_id"]),
            group_id=str(raw["group_id"]),
            domain=str(raw["domain"]),
            split=None if raw.get("split") is None else str(raw["split"]),
            critic_report=_mapping(raw["critic_report"], "critic_report"),
            context=RepairContext.from_dict(raw.get("context")),
            target_action=RepairAction(str(raw["target_action"])),
            action_rewards=dict(raw["action_rewards"]),
            available_actions=dict(raw["available_actions"]),
            source_trial_ids=tuple(str(item) for item in raw.get("source_trial_ids", ())),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class RepairRunResult:
    final_candidate: CandidateRecord
    final_report: dict[str, Any]
    stop_reason: str
    trials: tuple[RepairTrialV1, ...]
    candidate_history: tuple[CandidateRecord, ...]
    schema_version: str = RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "final_candidate": self.final_candidate.to_dict(),
            "final_report": self.final_report,
            "stop_reason": self.stop_reason,
            "trials": [trial.to_dict() for trial in self.trials],
            "candidate_history": [item.to_dict() for item in self.candidate_history],
        }
