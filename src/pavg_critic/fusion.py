"""融合规则与可选 VLM 证据，生成公开报告。

``detector_score`` 在当前基线中表示规则/视觉证据联合给出的违规置信度，而
``physics_score`` 定义为最强违规分数的补数。所有原始分数与模型标识都会写入
``evidence``，避免融合后的单一分数掩盖证据冲突。
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .config import FusionConfig
from .schemas import CriticReport, Violation, ViolationCandidate, VLMReview


class ResultFusion:
    """执行候选过滤、加权融合、排序和全局物理性判定。"""

    def __init__(self, config: FusionConfig) -> None:
        """保存融合权重与报告阈值。"""

        self.config = config

    def fuse(
        self,
        candidates: Iterable[ViolationCandidate],
        keyframes: Mapping[int, Sequence[int]],
        reviews: Mapping[int, VLMReview | None],
    ) -> CriticReport:
        """融合候选及其一一对应的关键帧和 VLM 复核。

        ``keyframes`` 与 ``reviews`` 使用候选索引作为键，避免不同对象出现同类别规则
        时发生字典键冲突。
        """

        violations: list[tuple[float, Violation]] = []
        for index, candidate in enumerate(candidates):
            review = reviews.get(index)
            score = self._score(candidate.detector_score, review)
            # 低于阈值的候选仍可在上游调试 artifacts 中看到，但不会污染公开报告。
            if score < self.config.violation_threshold:
                continue
            # 保留规则的原始扩展证据，再追加通用审计字段。
            evidence = dict(candidate.evidence)
            evidence.update(
                {
                    "rules": list(candidate.rules),
                    "detector_score": candidate.detector_score,
                    "fused_score": score,
                }
            )
            if review is not None:
                evidence.update({"vlm_score": review.score, "vlm_model": review.model})
            violations.append(
                (
                    score,
                    Violation(
                        object=candidate.object,
                        category=candidate.category,
                        start_frame=candidate.start_frame,
                        peak_frame=candidate.peak_frame,
                        end_frame=candidate.end_frame,
                        critical_frames=tuple(keyframes.get(index, ())),
                        reason=(review.reason if review and review.reason else candidate.reason),
                        repair_instruction=(
                            review.repair_instruction
                            if review and review.repair_instruction
                            else candidate.repair_instruction
                        ),
                        evidence=evidence,
                    ),
                )
            )

        # 最可信异常优先；相同分数下按时间和类别排序以保持跨运行确定性。
        violations.sort(key=lambda item: (-item[0], item[1].start_frame, item[1].category))
        if violations:
            strongest = violations[0][0]
            # 当前全局得分采用“最强违规”聚合，避免大量弱重复候选过度累加惩罚。
            physics_score = round(max(0.0, min(1.0, 1.0 - strongest)), 6)
            confidence = round(strongest, 6)
        else:
            # 无候选不等同于绝对正确，使用可配置 clean_confidence 表达规则覆盖有限。
            physics_score = 1.0
            confidence = self.config.clean_confidence
        return CriticReport(
            is_physical=physics_score >= self.config.physical_score_threshold,
            physics_score=physics_score,
            confidence=confidence,
            violations=tuple(item[1] for item in violations),
        )

    def _score(self, detector_score: float, review: VLMReview | None) -> float:
        """计算可用证据的归一化加权平均。

        VLM 缺席时直接返回规则分数，不把 VLM 权重当作零分加入分母。
        """

        if review is None:
            return detector_score
        total = self.config.detector_weight + self.config.vlm_weight
        return (
            self.config.detector_weight * detector_score
            + self.config.vlm_weight * review.score
        ) / total
