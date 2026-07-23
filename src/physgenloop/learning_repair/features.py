"""将 CriticReport 编码为版本化、确定性的策略特征。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
from typing import Any, Mapping, Sequence

from .contracts import ACTION_ORDER, RepairContext


DEFAULT_CATEGORIES = (
    "gravity_violation",
    "collision_violation",
    "friction_violation",
    "trajectory_violation",
    "continuity_violation",
    "contact_violation",
    "appearance_violation",
    "unknown_violation",
)
DEFAULT_EVIDENCE_FAMILIES = ("rules", "pqsg", "checklist", "mechanics", "vlm")


_CATEGORY_ALIASES = {
    "gravity": "gravity_violation",
    "gravity_violation": "gravity_violation",
    "reverse_gravity": "gravity_violation",
    "anti_gravity": "gravity_violation",
    "collision": "collision_violation",
    "collision_violation": "collision_violation",
    "surface_penetration": "collision_violation",
    "premature_rebound": "contact_violation",
    "contact": "contact_violation",
    "contact_violation": "contact_violation",
    "friction": "friction_violation",
    "friction_violation": "friction_violation",
    "trajectory": "trajectory_violation",
    "trajectory_violation": "trajectory_violation",
    "teleportation": "trajectory_violation",
    "midair_hover": "trajectory_violation",
    "object_disappearance": "continuity_violation",
    "continuity": "continuity_violation",
    "continuity_violation": "continuity_violation",
    "appearance": "appearance_violation",
}


def normalize_category(value: str) -> str:
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    return _CATEGORY_ALIASES.get(key, "unknown_violation")


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return converted
    if is_dataclass(value):
        return asdict(value)
    return {}


def _items(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _unit(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class FeatureConfig:
    """固定输入维度；必须随 checkpoint 一起保存。"""

    version: str = "1.0"
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    evidence_families: tuple[str, ...] = DEFAULT_EVIDENCE_FAMILIES
    object_hash_buckets: int = 8
    frame_span_scale: float = 120.0

    def __post_init__(self) -> None:
        if self.version != "1.0":
            raise ValueError(f"unsupported feature version: {self.version}")
        if not self.categories or len(self.categories) != len(set(self.categories)):
            raise ValueError("categories must be non-empty and unique")
        if self.object_hash_buckets < 1:
            raise ValueError("object_hash_buckets must be positive")
        if self.frame_span_scale <= 0:
            raise ValueError("frame_span_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "categories": list(self.categories),
            "evidence_families": list(self.evidence_families),
            "object_hash_buckets": self.object_hash_buckets,
            "frame_span_scale": self.frame_span_scale,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FeatureConfig":
        return cls(
            version=str(raw.get("version", "1.0")),
            categories=tuple(str(item) for item in raw.get("categories", DEFAULT_CATEGORIES)),
            evidence_families=tuple(
                str(item)
                for item in raw.get("evidence_families", DEFAULT_EVIDENCE_FAMILIES)
            ),
            object_hash_buckets=int(raw.get("object_hash_buckets", 8)),
            frame_span_scale=float(raw.get("frame_span_scale", 120.0)),
        )


class ReportFeatureEncoder:
    """只使用结构化诊断，不读取视频像素。"""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.feature_names = self._feature_names()

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def _feature_names(self) -> tuple[str, ...]:
        names = [
            "decision.physical",
            "decision.violation",
            "decision.unknown",
            "physics_score",
            "confidence",
            "coverage",
            "risk_score",
            "violation_count",
            "max_temporal_span",
            "mean_temporal_span",
            "repair_instruction_coverage",
            "context.attempt_fraction",
            "context.remaining_budget",
            "context.semantic_score",
            "context.original_prompt_semantic_score",
            "context.quality_score",
            "context.prompt_repair_available",
            "context.local_editor_available",
        ]
        names.extend(f"category.{name}" for name in self.config.categories)
        for family in self.config.evidence_families:
            names.extend(
                (
                    f"evidence.{family}.available",
                    f"evidence.{family}.score",
                    f"evidence.{family}.coverage",
                )
            )
        names.extend(
            f"object_hash.{index}" for index in range(self.config.object_hash_buckets)
        )
        names.extend(f"previous_action.{item.value}" for item in ACTION_ORDER)
        return tuple(names)

    def encode(
        self, report: Any, context: RepairContext | Mapping[str, Any] | None = None
    ) -> tuple[float, ...]:
        raw = _mapping(report)
        if not isinstance(context, RepairContext):
            context = RepairContext.from_dict(context)
        decision = str(raw.get("decision", "unknown"))
        physics_score = _unit(raw.get("physics_score"), 0.5)
        confidence = _unit(raw.get("confidence"), 0.0)
        coverage = _unit(raw.get("coverage"), 0.0)
        violations = tuple(_mapping(item) for item in _items(raw.get("violations", ())))

        spans: list[float] = []
        instruction_count = 0
        categories: set[str] = set()
        object_buckets = [0.0] * self.config.object_hash_buckets
        for violation in violations:
            start = int(violation.get("start_frame", 0) or 0)
            end = int(violation.get("end_frame", start) or start)
            spans.append(max(0, end - start + 1) / self.config.frame_span_scale)
            if str(violation.get("repair_instruction", "")).strip():
                instruction_count += 1
            categories.add(normalize_category(str(violation.get("category", ""))))
            object_name = str(violation.get("object", "unknown"))
            digest = hashlib.sha256(object_name.encode("utf-8")).digest()
            object_buckets[int.from_bytes(digest[:4], "big") % len(object_buckets)] = 1.0

        values = [
            1.0 if decision == "physical" else 0.0,
            1.0 if decision == "violation" else 0.0,
            1.0 if decision == "unknown" else 0.0,
            physics_score,
            confidence,
            coverage,
            1.0 - physics_score,
            min(1.0, len(violations) / 8.0),
            min(1.0, max(spans, default=0.0)),
            min(1.0, sum(spans) / len(spans)) if spans else 0.0,
            instruction_count / len(violations) if violations else 0.0,
            context.attempt_index / context.max_attempts,
            (context.max_attempts - context.attempt_index) / context.max_attempts,
            0.5 if context.semantic_score is None else context.semantic_score,
            (
                0.5
                if context.original_prompt_semantic_score is None
                else context.original_prompt_semantic_score
            ),
            0.5 if context.quality_score is None else context.quality_score,
            1.0 if context.prompt_repair_available else 0.0,
            1.0 if context.local_editor_available else 0.0,
        ]
        values.extend(1.0 if name in categories else 0.0 for name in self.config.categories)

        bundles = {
            str(item.get("family", "")): item
            for item in (_mapping(value) for value in _items(raw.get("evidence_bundles", ())))
        }
        for family in self.config.evidence_families:
            bundle = bundles.get(family, {})
            available = str(bundle.get("status", "")) == "available"
            values.extend(
                (
                    1.0 if available else 0.0,
                    _unit(bundle.get("score"), 0.5) if available else 0.0,
                    _unit(bundle.get("coverage"), 0.0),
                )
            )
        values.extend(object_buckets)
        values.extend(
            min(1.0, context.previous_actions.count(action) / context.max_attempts)
            for action in ACTION_ORDER
        )
        return tuple(float(value) for value in values)

    def primary_category(self, report: Any) -> str:
        raw = _mapping(report)
        violations = _items(raw.get("violations", ()))
        if not violations:
            return "unknown_violation"
        return normalize_category(str(_mapping(violations[0]).get("category", "")))
