"""Learning Repair Agent 的稳定数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


REPAIR_SCHEMA_VERSION = "2.0"


class RepairAction(str, Enum):
    """Repair Policy 可以选择的互斥动作。"""

    PROMPT_REPAIR = "prompt_repair"
    LOCAL_EDITING = "local_editing"
    REJECT = "reject"


ACTION_ORDER = tuple(RepairAction)


def _score(value: float, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        report = report.to_dict()
    if not isinstance(report, Mapping):
        raise ValueError("critic_report must be a mapping or expose to_dict()")
    return dict(report)


@dataclass(frozen=True)
class RepairContext:
    """动作选择所需的闭环状态与后端可用性。"""

    attempt_index: int = 0
    max_attempts: int = 3
    prompt_repair_available: bool = True
    local_editor_available: bool = True
    semantic_score: float | None = None
    original_prompt_semantic_score: float | None = None
    quality_score: float | None = None
    previous_actions: tuple[RepairAction, ...] = ()

    def __post_init__(self) -> None:
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if self.max_attempts < 1 or self.attempt_index > self.max_attempts:
            raise ValueError("max_attempts must be positive and cover attempt_index")
        if self.semantic_score is not None:
            object.__setattr__(
                self, "semantic_score", _score(self.semantic_score, "semantic_score")
            )
        if self.original_prompt_semantic_score is not None:
            object.__setattr__(
                self,
                "original_prompt_semantic_score",
                _score(
                    self.original_prompt_semantic_score,
                    "original_prompt_semantic_score",
                ),
            )
        if self.quality_score is not None:
            object.__setattr__(
                self, "quality_score", _score(self.quality_score, "quality_score")
            )
        object.__setattr__(
            self,
            "previous_actions",
            tuple(RepairAction(item) for item in self.previous_actions),
        )

    def action_available(self, action: RepairAction) -> bool:
        return {
            RepairAction.PROMPT_REPAIR: self.prompt_repair_available,
            RepairAction.LOCAL_EDITING: self.local_editor_available,
            RepairAction.REJECT: True,
        }[action]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "max_attempts": self.max_attempts,
            "prompt_repair_available": self.prompt_repair_available,
            "local_editor_available": self.local_editor_available,
            "semantic_score": self.semantic_score,
            "original_prompt_semantic_score": self.original_prompt_semantic_score,
            "quality_score": self.quality_score,
            "previous_actions": [item.value for item in self.previous_actions],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "RepairContext":
        raw = raw or {}
        return cls(
            attempt_index=int(raw.get("attempt_index", 0)),
            max_attempts=int(raw.get("max_attempts", 3)),
            prompt_repair_available=_boolean(
                raw.get("prompt_repair_available", True), "prompt_repair_available"
            ),
            local_editor_available=_boolean(
                raw.get("local_editor_available", True), "local_editor_available"
            ),
            semantic_score=(
                None if raw.get("semantic_score") is None else float(raw["semantic_score"])
            ),
            original_prompt_semantic_score=(
                None
                if raw.get("original_prompt_semantic_score") is None
                else float(raw["original_prompt_semantic_score"])
            ),
            quality_score=(
                None if raw.get("quality_score") is None else float(raw["quality_score"])
            ),
            previous_actions=tuple(
                RepairAction(str(item)) for item in raw.get("previous_actions", ())
            ),
        )


@dataclass(frozen=True)
class RepairExample:
    """一条由 Blender/真实闭环产出的监督样本。

    ``group_id`` 表示同一基础场景或同一源视频。训练、验证和测试必须按 group
    切分，避免同一场景的正常/异常变体跨集合泄漏。
    """

    sample_id: str
    group_id: str
    prompt: str
    critic_report: dict[str, Any]
    target_action: RepairAction
    strategy: str
    before_score: float
    after_score: float
    successful: bool
    context: RepairContext = field(default_factory=RepairContext)
    semantic_score: float | None = None
    quality_score: float | None = None
    repair_cost: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = REPAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPAIR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported repair schema {self.schema_version!r}; "
                f"expected {REPAIR_SCHEMA_VERSION!r}"
            )
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not self.group_id.strip():
            raise ValueError("group_id must not be empty")
        if not isinstance(self.target_action, RepairAction):
            object.__setattr__(self, "target_action", RepairAction(self.target_action))
        if not isinstance(self.context, RepairContext):
            object.__setattr__(self, "context", RepairContext.from_dict(self.context))
        object.__setattr__(self, "critic_report", _report_dict(self.critic_report))
        object.__setattr__(self, "before_score", _score(self.before_score, "before_score"))
        object.__setattr__(self, "after_score", _score(self.after_score, "after_score"))
        if self.semantic_score is not None:
            object.__setattr__(
                self, "semantic_score", _score(self.semantic_score, "semantic_score")
            )
        if self.quality_score is not None:
            object.__setattr__(
                self, "quality_score", _score(self.quality_score, "quality_score")
            )
        if self.repair_cost is not None and self.repair_cost < 0:
            raise ValueError("repair_cost must be non-negative")
        if self.split not in (None, "train", "validation", "test"):
            raise ValueError("split must be train, validation, test, or null")
        if not isinstance(self.artifacts, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.artifacts.items()
        ):
            raise ValueError("artifacts must be a string-to-string mapping")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    @property
    def score_gain(self) -> float:
        return self.after_score - self.before_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "split": self.split,
            "prompt": self.prompt,
            "critic_report": self.critic_report,
            "context": self.context.to_dict(),
            "target": {
                "action": self.target_action.value,
                "strategy": self.strategy,
            },
            "outcome": {
                "before_score": self.before_score,
                "after_score": self.after_score,
                "successful": self.successful,
                "semantic_score": self.semantic_score,
                "quality_score": self.quality_score,
                "repair_cost": self.repair_cost,
            },
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RepairExample":
        target = raw.get("target")
        outcome = raw.get("outcome")
        if not isinstance(target, Mapping):
            raise ValueError("repair example target must be an object")
        if not isinstance(outcome, Mapping):
            raise ValueError("repair example outcome must be an object")
        required = ("sample_id", "group_id", "critic_report")
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"repair example missing fields: {missing}")
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            sample_id=str(raw["sample_id"]),
            group_id=str(raw["group_id"]),
            split=None if raw.get("split") is None else str(raw["split"]),
            prompt=str(raw.get("prompt", "")),
            critic_report=_report_dict(raw["critic_report"]),
            context=RepairContext.from_dict(raw.get("context")),
            target_action=RepairAction(str(target.get("action", ""))),
            strategy=str(target.get("strategy", "")),
            before_score=float(outcome.get("before_score", -1.0)),
            after_score=float(outcome.get("after_score", -1.0)),
            successful=_boolean(outcome.get("successful"), "outcome.successful"),
            semantic_score=(
                None
                if outcome.get("semantic_score") is None
                else float(outcome["semantic_score"])
            ),
            quality_score=(
                None
                if outcome.get("quality_score") is None
                else float(outcome["quality_score"])
            ),
            repair_cost=(
                None
                if outcome.get("repair_cost") is None
                else float(outcome["repair_cost"])
            ),
            artifacts={str(k): str(v) for k, v in dict(raw.get("artifacts", {})).items()},
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class PolicyPrediction:
    """策略模型的原始概率与预期物理分提升。"""

    action_probabilities: dict[RepairAction, float]
    expected_gain: float = 0.0
    model_id: str = "unknown"

    def __post_init__(self) -> None:
        probabilities = {
            RepairAction(action): float(value)
            for action, value in self.action_probabilities.items()
        }
        if set(probabilities) != set(ACTION_ORDER):
            raise ValueError("action_probabilities must contain every RepairAction")
        if any(value < 0.0 for value in probabilities.values()):
            raise ValueError("action probabilities must be non-negative")
        total = sum(probabilities.values())
        if total <= 0.0:
            raise ValueError("action probabilities must have a positive sum")
        object.__setattr__(
            self,
            "action_probabilities",
            {action: value / total for action, value in probabilities.items()},
        )
        object.__setattr__(
            self, "expected_gain", max(-1.0, min(1.0, float(self.expected_gain)))
        )

    @property
    def action(self) -> RepairAction:
        return max(
            ACTION_ORDER,
            key=lambda action: self.action_probabilities[action],
        )

    @property
    def confidence(self) -> float:
        return self.action_probabilities[self.action]


@dataclass(frozen=True)
class LegacyRepairDecision:
    """Legacy classification decision retained only for checkpoint adaptation."""

    action: RepairAction
    confidence: float
    instruction: str
    expected_gain: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    memory_ids: tuple[str, ...] = ()
    action_probabilities: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, RepairAction):
            object.__setattr__(self, "action", RepairAction(self.action))
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        object.__setattr__(
            self, "expected_gain", max(-1.0, min(1.0, float(self.expected_gain)))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "instruction": self.instruction,
            "expected_gain": self.expected_gain,
            "parameters": self.parameters,
            "source": self.source,
            "memory_ids": list(self.memory_ids),
            "action_probabilities": self.action_probabilities,
        }
