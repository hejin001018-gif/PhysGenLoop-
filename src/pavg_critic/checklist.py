"""VideoScience-Bench 启发的五维检查表评估。

评估器不重复解码视频：它消费 PAVG 共享感知层已经产生的轨迹、事件和规则候选。
每个维度都显式返回 pass/fail/unknown，unknown 不会被当成零分，从而让覆盖率和质量
分数分开解释。未来的光流、分割或语义模型可把额外证据写入相同结果结构。
"""

from __future__ import annotations

from dataclasses import replace

from .config import ChecklistConfig
from .schemas import (
    CHECKLIST_DIMENSIONS,
    ChecklistResult,
    ChecklistSummary,
    CriticRequest,
    Event,
    TrackSequence,
    ViolationCandidate,
    VisualEvidence,
)


class VideoScienceChecklistEvaluator:
    """以五个固定维度审计现象、运动、连续性、身份和交互。"""

    def __init__(self, config: ChecklistConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        request: CriticRequest,
        tracks: tuple[TrackSequence, ...],
        events: tuple[Event, ...],
        candidates: tuple[ViolationCandidate, ...],
        external_evidence: tuple[VisualEvidence, ...] = (),
    ) -> tuple[tuple[ChecklistResult, ...], ChecklistSummary]:
        event_types = {event.event_type for event in events}
        candidate_categories = {candidate.category for candidate in candidates}
        results = (
            self._phenomenon(request, events, event_types),
            self._dynamism(request, events, candidates, event_types, candidate_categories),
            self._continuity(tracks, events, candidates, candidate_categories),
            self._immutability(tracks, events, candidates, candidate_categories),
            self._interaction(request, events, candidates, event_types, candidate_categories),
        )
        results = tuple(
            self._apply_external_evidence(item, external_evidence) for item in results
        )
        answered = [item for item in results if item.score is not None]
        summary = ChecklistSummary(
            score=(
                sum(item.score or 0.0 for item in answered) / len(answered)
                if answered
                else None
            ),
            coverage=len(answered) / len(CHECKLIST_DIMENSIONS),
            passed=sum(item.status == "pass" for item in results),
            failed=sum(item.status == "fail" for item in results),
            unknown=sum(item.status == "unknown" for item in results),
        )
        return results, summary

    def _apply_external_evidence(
        self,
        result: ChecklistResult,
        evidence: tuple[VisualEvidence, ...],
    ) -> ChecklistResult:
        """用外部 CV 证据补齐 unknown，并为已有判定附加来源而不覆盖强规则失败。"""

        matching = tuple(item for item in evidence if item.dimension == result.dimension)
        if not matching:
            return result
        sources = tuple(dict.fromkeys(result.evidence_sources + tuple(
            item.source for item in matching
        )))
        frames = tuple(dict.fromkeys(result.critical_frames + tuple(
            frame for item in matching for frame in item.critical_frames
        )))
        external_payload = [
            {
                "source": item.source,
                "score": item.score,
                "confidence": item.confidence,
                "measurements": item.measurements,
            }
            for item in matching
        ]
        merged_evidence = dict(result.evidence)
        merged_evidence["external_cv"] = external_payload
        if result.status != "unknown":
            return replace(
                result,
                evidence_sources=sources,
                critical_frames=frames,
                evidence=merged_evidence,
            )

        total_confidence = sum(item.confidence for item in matching)
        score = (
            sum(item.score * item.confidence for item in matching) / total_confidence
            if total_confidence
            else sum(item.score for item in matching) / len(matching)
        )
        return ChecklistResult(
            dimension=result.dimension,
            status="pass" if score >= 0.5 else "fail",
            score=score,
            confidence=max(item.confidence for item in matching),
            reason="External CV evidence resolved a dimension without deterministic evidence.",
            critical_frames=frames,
            evidence_sources=sources,
            evidence=merged_evidence,
        )

    def _phenomenon(self, request, events, event_types) -> ChecklistResult:
        expected = tuple(dict.fromkeys(request.physics_plan.expected_events))
        if not expected:
            return _unknown("phenomenon_congruency", "No expected events were declared.")
        missing = [name for name in expected if not _event_observed(name, event_types)]
        if missing:
            return _failed(
                "phenomenon_congruency",
                f"Expected events were not observed: {missing}.",
                events,
                evidence={"expected": list(expected), "missing": missing},
            )
        return _passed(
            "phenomenon_congruency",
            "All declared phenomena have matching detected events.",
            self.config.pass_confidence,
            events,
            evidence={"expected": list(expected)},
        )

    def _dynamism(
        self, request, events, candidates, event_types, candidate_categories
    ) -> ChecklistResult:
        bad = {"premature_rebound", "reverse_gravity"}.intersection(candidate_categories)
        if bad:
            return _failed(
                "correct_dynamism",
                f"Dynamic rule violations were detected: {sorted(bad)}.",
                events,
                candidates,
            )
        expected = set(request.physics_plan.expected_events)
        dynamic = expected.intersection({"leave_support", "fall", "rebound"})
        if not dynamic:
            return _unknown("correct_dynamism", "No dynamic event is declared in the plan.")
        if all(_event_observed(name, event_types) for name in dynamic):
            return _passed(
                "correct_dynamism",
                "Expected dynamic phases are present without a rule-level dynamic violation.",
                self.config.pass_confidence,
                events,
            )
        return _failed(
            "correct_dynamism",
            "One or more expected dynamic phases are missing.",
            events,
        )

    def _continuity(
        self, tracks, events, candidates, candidate_categories
    ) -> ChecklistResult:
        bad = {"teleportation", "object_disappearance"}.intersection(candidate_categories)
        if bad:
            return _failed(
                "spatiotemporal_continuity",
                f"Continuity violations were detected: {sorted(bad)}.",
                events,
                candidates,
            )
        if tracks or events:
            return _passed(
                "spatiotemporal_continuity",
                "Available trajectories contain no configured continuity violation.",
                self.config.pass_confidence,
                events,
            )
        return _unknown("spatiotemporal_continuity", "No trajectory evidence is available.")

    def _immutability(
        self, tracks, events, candidates, candidate_categories
    ) -> ChecklistResult:
        if "object_disappearance" in candidate_categories:
            return _failed(
                "immutability",
                "The tracked object identity is not preserved across the clip.",
                events,
                candidates,
            )
        visible_count = sum(state.visible for track in tracks for state in track.states)
        if visible_count:
            return _passed(
                "immutability",
                "The tracker preserves a stable object identity in visible observations.",
                self.config.pass_confidence,
                events,
                evidence={"visible_observation_count": visible_count},
            )
        return _unknown("immutability", "No visible tracked identity is available.")

    def _interaction(
        self, request, events, candidates, event_types, candidate_categories
    ) -> ChecklistResult:
        bad = {"surface_penetration", "premature_rebound"}.intersection(
            candidate_categories
        )
        if bad:
            return _failed(
                "interaction_realism",
                f"Interaction violations were detected: {sorted(bad)}.",
                events,
                candidates,
            )
        expected = set(request.physics_plan.expected_events)
        interaction = expected.intersection({"floor_contact", "rebound", "collision"})
        if not interaction:
            return _unknown("interaction_realism", "No interaction is declared in the plan.")
        if all(_event_observed(name, event_types) for name in interaction):
            return _passed(
                "interaction_realism",
                "Expected contact phases occur without a configured interaction violation.",
                self.config.pass_confidence,
                events,
            )
        return _failed(
            "interaction_realism",
            "The declared interaction lacks matching contact evidence.",
            events,
        )


def _event_observed(expected: str, observed: set[str]) -> bool:
    aliases = {"leave_support": "fall"}
    return aliases.get(expected, expected) in observed


def _frames(
    events: tuple[Event, ...], candidates: tuple[ViolationCandidate, ...] = ()
) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            [event.peak_frame for event in events]
            + [frame for candidate in candidates for frame in candidate.evidence_frames]
        )
    )


def _passed(
    dimension: str,
    reason: str,
    confidence: float,
    events: tuple[Event, ...],
    *,
    evidence: dict | None = None,
) -> ChecklistResult:
    return ChecklistResult(
        dimension=dimension,
        status="pass",
        score=1.0,
        confidence=confidence,
        reason=reason,
        critical_frames=_frames(events),
        evidence_sources=("event_detector",),
        evidence=evidence or {},
    )


def _failed(
    dimension: str,
    reason: str,
    events: tuple[Event, ...],
    candidates: tuple[ViolationCandidate, ...] = (),
    *,
    evidence: dict | None = None,
) -> ChecklistResult:
    sources = ["event_detector"] if events else []
    if candidates:
        sources.append("rule_candidate")
    return ChecklistResult(
        dimension=dimension,
        status="fail",
        score=0.0,
        confidence=max(
            [item.confidence for item in events]
            + [item.detector_score for item in candidates]
            + [0.5]
        ),
        reason=reason,
        critical_frames=_frames(events, candidates),
        evidence_sources=tuple(sources),
        evidence=evidence or {},
    )


def _unknown(dimension: str, reason: str) -> ChecklistResult:
    return ChecklistResult(
        dimension=dimension,
        status="unknown",
        score=None,
        confidence=0.0,
        reason=reason,
    )
