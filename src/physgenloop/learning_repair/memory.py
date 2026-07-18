"""轻量、可审计的 Repair Memory 检索实现。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from .contracts import ACTION_ORDER, RepairAction, RepairContext, RepairExample
from .dataset import load_repair_manifest, write_repair_manifest
from .features import ReportFeatureEncoder


@dataclass(frozen=True)
class MemoryMatch:
    example: RepairExample
    similarity: float
    utility: float


def _cosine(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return numerator / (first_norm * second_norm)


class RepairMemory:
    """用结构化 Critic 特征做余弦检索，不依赖外部向量数据库。"""

    def __init__(
        self,
        examples: Iterable[RepairExample] = (),
        *,
        encoder: ReportFeatureEncoder | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.encoder = encoder or ReportFeatureEncoder()
        self.path = None if path is None else Path(path)
        self._examples = list(examples)
        self._vectors = [
            self.encoder.encode(item.critic_report, item.context) for item in self._examples
        ]

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        encoder: ReportFeatureEncoder | None = None,
    ) -> "RepairMemory":
        return cls(load_repair_manifest(path), encoder=encoder, path=path)

    def __len__(self) -> int:
        return len(self._examples)

    def add(self, example: RepairExample, *, persist: bool = False) -> None:
        if any(item.sample_id == example.sample_id for item in self._examples):
            raise ValueError(f"duplicate memory sample_id: {example.sample_id}")
        self._examples.append(example)
        self._vectors.append(self.encoder.encode(example.critic_report, example.context))
        if persist:
            if self.path is None:
                raise ValueError("memory path is required for persistence")
            # Rewrite a small auditable JSONL memory atomically at the logical level.
            # Production multi-writer deployments should replace this storage adapter.
            write_repair_manifest(self._examples, self.path)

    def retrieve(
        self,
        critic_report: Any,
        *,
        context: RepairContext | None = None,
        k: int = 5,
        successful_only: bool = True,
        minimum_similarity: float = 0.0,
    ) -> tuple[MemoryMatch, ...]:
        if k < 1:
            raise ValueError("k must be positive")
        query = self.encoder.encode(critic_report, context)
        matches: list[MemoryMatch] = []
        for example, vector in zip(self._examples, self._vectors):
            if successful_only and not example.successful:
                continue
            similarity = max(0.0, _cosine(query, vector))
            if similarity < minimum_similarity:
                continue
            # Positive score gain makes a similar experience more trustworthy. A
            # successful zero-gain sample still keeps half of its similarity weight.
            gain_factor = 0.5 + 0.5 * max(0.0, example.score_gain)
            utility = similarity * gain_factor
            matches.append(MemoryMatch(example, similarity, utility))
        matches.sort(
            key=lambda item: (-item.utility, -item.similarity, item.example.sample_id)
        )
        return tuple(matches[:k])

    @staticmethod
    def action_distribution(
        matches: Iterable[MemoryMatch],
    ) -> dict[RepairAction, float]:
        weights = {action: 0.0 for action in ACTION_ORDER}
        for match in matches:
            weights[match.example.target_action] += match.utility
        total = sum(weights.values())
        if total <= 0.0:
            return {action: 1.0 / len(ACTION_ORDER) for action in ACTION_ORDER}
        return {action: value / total for action, value in weights.items()}
