"""覆盖率与置信度校准的多证据家族融合。

旧候选融合仍负责生成逐条 violation；本层在其后统一规则、PQSG、检查表、力学和
VLM 的物理可信度。所有权重都乘以 coverage 与 confidence，防止“未运行”被解释为
满分；低覆盖时返回 unknown，而不是用缺失证据证明视频正确。
"""

from __future__ import annotations

from dataclasses import replace
from statistics import fmean
from typing import Mapping, Sequence

from .config import FusionConfig
from .physics_rules import RULE_ID_TO_CATEGORY
from .schemas import (
    ChecklistSummary,
    CriticReport,
    EvidenceBundle,
    MechanicsSummary,
    TrackSequence,
    VLMReview,
    ViolationCandidate,
)


class CoverageAwareEvidenceFusion:
    """构造五类证据包并生成 physical/violation/unknown 三态报告。"""

    def __init__(self, config: FusionConfig, *, enabled_rules: Sequence[str]) -> None:
        self.config = config
        self.enabled_rules = frozenset(enabled_rules)
        self.weights = {
            "rules": config.rule_family_weight,
            "pqsg": config.pqsg_family_weight,
            "checklist": config.checklist_family_weight,
            "mechanics": config.mechanics_family_weight,
            "vlm": config.vlm_family_weight,
        }

    def enrich(
        self,
        report: CriticReport,
        *,
        tracks: tuple[TrackSequence, ...],
        candidates: tuple[ViolationCandidate, ...],
        reviews: Mapping[int, VLMReview | None],
        checklist_summary: ChecklistSummary | None,
        mechanics_summary: MechanicsSummary | None,
    ) -> CriticReport:
        bundles = (
            self._rules(report, tracks, candidates),
            self._pqsg(report),
            self._checklist(checklist_summary),
            self._mechanics(mechanics_summary),
            self._vlm(candidates, reviews),
        )
        total_family_weight = sum(self.weights.values())
        coverage = sum(
            self.weights[item.family] * item.coverage for item in bundles
        ) / total_family_weight
        available = [
            item
            for item in bundles
            if item.status == "available"
            and item.score is not None
            and self.weights[item.family] > 0
        ]
        effective_weights = [
            self.weights[item.family] * item.coverage * item.confidence
            for item in available
        ]
        effective_total = sum(effective_weights)
        if effective_total:
            physics_score = sum(
                item.score * weight
                for item, weight in zip(available, effective_weights)
            ) / effective_total
        else:
            # 0.5 是无证据的中性先验，避免旧实现将空结果写成 1.0。
            physics_score = 0.5
        hard_violation = bool(report.violations)
        if hard_violation:
            decision = "violation"
            physics_score = min(physics_score, report.physics_score)
        elif coverage < self.config.minimum_coverage or not available:
            decision = "unknown"
        elif physics_score < self.config.physical_score_threshold:
            decision = "violation"
        else:
            decision = "physical"
        confidence = effective_total / total_family_weight
        if hard_violation:
            confidence = max(confidence, report.confidence)

        score_breakdown = dict(report.score_breakdown)
        score_breakdown.update(
            {item.family: item.score for item in available if item.score is not None}
        )
        return replace(
            report,
            decision=decision,
            is_physical=decision == "physical",
            physics_score=round(max(0.0, min(1.0, physics_score)), 6),
            confidence=round(max(0.0, min(1.0, confidence)), 6),
            coverage=round(max(0.0, min(1.0, coverage)), 6),
            score_breakdown=score_breakdown,
            evidence_bundles=bundles,
        )

    def _rules(self, report, tracks, candidates) -> EvidenceBundle:
        visible_count = sum(state.visible for track in tracks for state in track.states)
        enabled_fraction = len(self.enabled_rules) / max(
            len(set(RULE_ID_TO_CATEGORY.values())), 1
        )
        coverage = min(1.0, visible_count / 3.0) * min(1.0, enabled_fraction)
        if candidates:
            coverage = max(coverage, 0.6)
        if not visible_count and not candidates:
            return _unavailable("rules", "deterministic_rules", "unknown")
        frames = tuple(
            dict.fromkeys(frame for item in candidates for frame in item.evidence_frames)
        )
        return EvidenceBundle(
            family="rules",
            source="deterministic_rules",
            status="available",
            score=report.physics_score,
            confidence=report.confidence,
            coverage=coverage,
            critical_frames=frames,
            details={"candidate_count": len(candidates), "visible_states": visible_count},
        )

    def _pqsg(self, report) -> EvidenceBundle:
        summary = report.graph_evaluation
        if summary is None or summary.physics_plausibility_score is None:
            return _unavailable("pqsg", "hybrid_question_graph", "not_applicable")
        return EvidenceBundle(
            family="pqsg",
            source="hybrid_question_graph",
            status="available",
            score=summary.physics_plausibility_score,
            confidence=max(0.5, summary.physics_coverage),
            coverage=summary.physics_coverage,
            critical_frames=tuple(
                dict.fromkeys(
                    frame for result in report.node_results for frame in result.critical_frames
                )
            ),
            details={"prompt_fulfillment": summary.prompt_fulfillment_score},
        )

    def _checklist(self, summary) -> EvidenceBundle:
        if summary is None or summary.score is None:
            return _unavailable("checklist", "video_science_checklist", "unknown")
        return EvidenceBundle(
            family="checklist",
            source="video_science_checklist",
            status="available",
            score=summary.score,
            confidence=max(0.5, summary.coverage),
            coverage=summary.coverage,
            details={
                "passed": summary.passed,
                "failed": summary.failed,
                "unknown": summary.unknown,
            },
        )

    def _mechanics(self, summary) -> EvidenceBundle:
        if summary is None or summary.score is None:
            status = "failed" if summary is not None and summary.failed else "not_applicable"
            return _unavailable("mechanics", "morpheus_mechanics", status)
        return EvidenceBundle(
            family="mechanics",
            source="morpheus_mechanics",
            status="available",
            score=summary.score,
            confidence=0.8,
            coverage=summary.coverage,
            details={"applicable_evaluators": summary.applicable},
        )

    def _vlm(self, candidates, reviews) -> EvidenceBundle:
        if not candidates:
            return _unavailable("vlm", "candidate_vlm", "not_applicable")
        available = [review for review in reviews.values() if review is not None]
        if not available:
            return _unavailable("vlm", "candidate_vlm", "unknown")
        violation_score = fmean(review.score for review in available)
        confidence = fmean(max(0.5, abs(review.score - 0.5) * 2.0) for review in available)
        return EvidenceBundle(
            family="vlm",
            source="candidate_vlm",
            status="available",
            score=1.0 - violation_score,
            confidence=confidence,
            coverage=len(available) / len(candidates),
            details={"review_count": len(available)},
        )


def _unavailable(family: str, source: str, status: str) -> EvidenceBundle:
    return EvidenceBundle(
        family=family,
        source=source,
        status=status,
        score=None,
        confidence=0.0,
        coverage=0.0,
    )
