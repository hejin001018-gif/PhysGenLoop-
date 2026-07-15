"""候选视频的稳定选择策略。"""

from __future__ import annotations

from .contracts import CandidateEvaluation


_DECISION_RANK = {"violation": 0, "unknown": 1, "physical": 2}


class EvidenceAwareSelector:
    """先比较判定，再比较物理分数和证据置信度。"""

    def select(
        self, evaluations: tuple[CandidateEvaluation, ...]
    ) -> CandidateEvaluation:
        if not evaluations:
            raise ValueError("evaluations must not be empty")
        return max(
            evaluations,
            key=lambda item: (
                _DECISION_RANK[item.report.decision],
                item.report.physics_score,
                item.report.confidence,
            ),
        )
