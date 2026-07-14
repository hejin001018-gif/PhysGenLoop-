"""VideoScience 风格五维检查表。"""

from __future__ import annotations

from pavg_critic.checklist import VideoScienceChecklistEvaluator
from pavg_critic import FrameState, PhysicsCritic
from pavg_critic.config import ChecklistConfig
from pavg_critic.schemas import (
    CriticRequest,
    VisualEvidence,
    Event,
    PhysicsPlan,
    TrackSequence,
    ViolationCandidate,
)


def _event(event_type: str, frame: int = 1) -> Event:
    return Event(
        event_type=event_type,
        object="red_ball",
        track_id="ball-1",
        start_frame=frame,
        peak_frame=frame,
        end_frame=frame,
        confidence=0.9,
    )


def _candidate(category: str, frame: int = 2) -> ViolationCandidate:
    return ViolationCandidate(
        object="red_ball",
        track_id="ball-1",
        category=category,
        start_frame=frame,
        peak_frame=frame,
        end_frame=frame,
        reason="synthetic violation",
        repair_instruction="repair it",
        detector_score=0.9,
        rules=("synthetic_rule",),
        evidence_frames=(frame,),
    )


def test_clean_expected_interaction_passes_relevant_dimensions():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(
            objects=("red_ball",), expected_events=("fall", "floor_contact", "rebound")
        ),
    )
    evaluator = VideoScienceChecklistEvaluator(ChecklistConfig())

    results, summary = evaluator.evaluate(
        request=request,
        tracks=(),
        events=(_event("fall"), _event("floor_contact", 2), _event("rebound", 3)),
        candidates=(),
    )

    by_dimension = {item.dimension: item for item in results}
    assert by_dimension["phenomenon_congruency"].status == "pass"
    assert by_dimension["correct_dynamism"].status == "pass"
    assert by_dimension["interaction_realism"].status == "pass"
    assert summary.score == 1.0


def test_teleport_candidate_fails_spatiotemporal_continuity_with_frames():
    evaluator = VideoScienceChecklistEvaluator(ChecklistConfig())

    results, _ = evaluator.evaluate(
        request=CriticRequest(video_path="unused.mp4"),
        tracks=(),
        events=(_event("teleport", 8),),
        candidates=(_candidate("teleportation", 8),),
    )

    continuity = next(
        item for item in results if item.dimension == "spatiotemporal_continuity"
    )
    assert continuity.status == "fail"
    assert continuity.critical_frames == (8,)
    assert "rule_candidate" in continuity.evidence_sources


def test_non_applicable_dimensions_are_unknown_not_zero():
    evaluator = VideoScienceChecklistEvaluator(ChecklistConfig())

    results, summary = evaluator.evaluate(
        request=CriticRequest(video_path="unused.mp4"),
        tracks=(),
        events=(),
        candidates=(),
    )

    assert any(item.status == "unknown" for item in results)
    assert summary.coverage < 1.0


def test_pipeline_attaches_checklist_results_to_artifacts_and_report():
    observation = FrameState(
        frame=0,
        timestamp_sec=0,
        object="red_ball",
        center=(10, 10),
        bbox=(5, 5, 15, 15),
    )

    artifacts = PhysicsCritic().analyze_detailed(
        CriticRequest(video_path="unused.mp4"),
        observations=(observation,),
        floor_y=100,
    )

    assert len(artifacts.checklist_results) == 5
    assert "video_science" in artifacts.report.diagnostics


def test_external_cv_evidence_can_resolve_an_unknown_dimension():
    class ShapeEvidenceExtractor:
        def extract(self, request, tracks, events):
            return (
                VisualEvidence(
                    dimension="immutability",
                    source="shape_embedding",
                    score=0.9,
                    confidence=0.8,
                    critical_frames=(0, 3),
                    measurements={"cosine_similarity": 0.9},
                ),
            )

    artifacts = PhysicsCritic(
        visual_evidence_extractors=(ShapeEvidenceExtractor(),)
    ).analyze_detailed(
        CriticRequest(video_path="unused.mp4"),
        observations=(),
        floor_y=100,
    )

    immutability = next(
        item for item in artifacts.checklist_results if item.dimension == "immutability"
    )
    assert immutability.status == "pass"
    assert "shape_embedding" in immutability.evidence_sources
    assert len(artifacts.visual_evidence) == 1
