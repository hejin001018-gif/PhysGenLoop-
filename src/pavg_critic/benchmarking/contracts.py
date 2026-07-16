"""Canonical inputs and outputs shared by benchmark methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


SemanticLabel = Literal["adherent", "not_adherent", "unknown"]
PhysicsLabel = Literal["physical", "violation", "unknown"]


@dataclass(frozen=True)
class BenchmarkSample:
    """One normalized, locally materialized benchmark sample."""

    sample_id: str
    benchmark: str
    split: str
    prompt: str
    video_path: str
    prompt_group_id: str
    generator: str
    semantic_label: SemanticLabel
    physics_label: PhysicsLabel
    semantic_score: float | None = None
    physics_score: float | None = None
    physical_rules: tuple[str, ...] = ()
    raw_labels: Mapping[str, Any] = field(default_factory=dict)
    source_url: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.prompt_group_id.strip():
            raise ValueError("sample_id and prompt_group_id must not be empty")
        if not Path(self.video_path).is_file():
            raise ValueError(f"video does not exist: {self.video_path}")
        if self.semantic_label not in {"adherent", "not_adherent", "unknown"}:
            raise ValueError(f"invalid semantic label: {self.semantic_label}")
        if self.physics_label not in {"physical", "violation", "unknown"}:
            raise ValueError(f"invalid physics label: {self.physics_label}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BenchmarkSample":
        data = dict(raw)
        data["physical_rules"] = tuple(data.get("physical_rules", ()))
        return cls(**data)


@dataclass(frozen=True)
class BenchmarkPrediction:
    """One method's auditable prediction for one benchmark sample."""

    sample_id: str
    method_id: str
    model_id: str | None
    semantic_score: float | None
    physics_score: float | None
    semantic_label: SemanticLabel
    physics_label: PhysicsLabel
    confidence: float
    coverage: float
    latency_sec: float
    visible_frame_count: int
    violation_categories: tuple[str, ...] = ()
    evidence_frames: tuple[int, ...] = ()
    repair_instruction: str | None = None
    failure: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.confidence, "confidence"),
            (self.coverage, "coverage"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.visible_frame_count < 0 or self.latency_sec < 0:
            raise ValueError("frame count and latency must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BenchmarkPrediction":
        data = dict(raw)
        data["violation_categories"] = tuple(data.get("violation_categories", ()))
        data["evidence_frames"] = tuple(data.get("evidence_frames", ()))
        return cls(**data)
