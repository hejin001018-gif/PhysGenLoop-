"""PhysGenLoop 跨组件共享的不可变数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pavg_critic.schemas import CriticReport, PhysicsPlan


@dataclass(frozen=True)
class LoopConfig:
    """控制 Best-of-K 和最大反馈轮数的最小配置。"""

    max_rounds: int = 3
    candidates_per_round: int = 2
    acceptance_score: float = 0.8
    base_seed: int = 42
    error_scope_threshold: float = 0.4
    default_total_frames: int | None = None

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if self.candidates_per_round < 1:
            raise ValueError("candidates_per_round must be at least 1")
        if not 0.0 <= self.acceptance_score <= 1.0:
            raise ValueError("acceptance_score must be within [0, 1]")
        if not 0.0 <= self.error_scope_threshold <= 1.0:
            raise ValueError("error_scope_threshold must be within [0, 1]")
        if self.default_total_frames is not None and self.default_total_frames < 1:
            raise ValueError("default_total_frames must be positive when provided")


@dataclass(frozen=True)
class GeneratedCandidate:
    """生成器返回的候选视频引用；不要求视频已加载到内存。"""

    candidate_id: str
    video_path: str
    prompt: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if not self.video_path.strip():
            raise ValueError("video_path must not be empty")


@dataclass(frozen=True)
class CandidateEvaluation:
    """一个候选与其结构化 Critic 报告。"""

    candidate: GeneratedCandidate
    report: CriticReport


@dataclass(frozen=True)
class LoopRound:
    """一次生成—评价—选择的完整审计记录。"""

    round_index: int
    prompt: str
    evaluations: tuple[CandidateEvaluation, ...]
    selected_candidate_id: str


@dataclass(frozen=True)
class LoopResult:
    """有界循环的最终结果与全部历史。"""

    best: CandidateEvaluation
    history: tuple[LoopRound, ...]
    stop_reason: str
    resolved_plan: PhysicsPlan
