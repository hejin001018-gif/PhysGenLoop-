"""统一证据包、覆盖率校准和三态决策。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from pavg_critic import CriticRequest, FrameState, PhysicsCritic
from pavg_critic.config import CriticConfig, FusionConfig, QuestionGraphConfig, TrajectoryConfig
from pavg_critic.evidence_fusion import hard_violation_override_applied
from pavg_critic.fusion import ResultFusion
from pavg_critic.schemas import (
    CriticReport,
    EvidenceBundle,
    PhysicsPlan,
    Violation,
    ViolationCandidate,
    VLMReview,
)


def _state(frame, y, velocity, bottom=None):
    bottom = y + 5 if bottom is None else bottom
    return FrameState(
        frame=frame,
        timestamp_sec=frame / 10,
        object="red_ball",
        track_id="ball-1",
        center=(50, y),
        bbox=(45, bottom - 10, 55, bottom),
        velocity=velocity,
    )


def test_no_observations_is_unknown_instead_of_perfectly_physical():
    report = PhysicsCritic().analyze(
        CriticRequest(video_path="unused.mp4"),
        observations=(),
        floor_y=100,
    )

    assert report.decision == "unknown"
    assert report.is_physical is False
    assert report.coverage == 0.0
    assert report.physics_score == 0.5


def test_well_observed_clean_contact_is_physical_with_evidence_bundles():
    states = (
        _state(0, 85, (0, 100), 90),
        _state(1, 95, (0, 100), 100),
        _state(2, 92, (0, -100), 100),
        _state(3, 80, (0, -100), 85),
    )
    config = CriticConfig(
        trajectory=TrajectoryConfig(smoothing_window=1),
        question_graph=QuestionGraphConfig(enabled=False),
    )
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(
            objects=("red_ball",), expected_events=("fall", "floor_contact", "rebound")
        ),
    )

    report = PhysicsCritic(config).analyze(request, observations=states, floor_y=100)

    assert report.decision == "physical"
    assert report.coverage >= config.fusion.minimum_coverage
    assert {bundle.family for bundle in report.evidence_bundles} == {
        "rules",
        "pqsg",
        "checklist",
        "mechanics",
        "vlm",
    }


def test_strong_existing_rule_violation_remains_violation_after_fusion():
    states = (
        _state(0, 40, (0, 100)),
        _state(1, 55, (0, 100)),
        _state(2, 48, (0, -100)),
        _state(3, 38, (0, -100)),
    )
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=("fall",)),
    )

    report = PhysicsCritic().analyze(request, observations=states, floor_y=100)

    assert report.decision == "violation"
    assert any(item.category == "premature_rebound" for item in report.violations)


def test_fused_report_still_validates_schema_2():
    report = PhysicsCritic().analyze(
        CriticRequest(video_path="unused.mp4"), observations=(), floor_y=100
    )
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "critic_output.schema.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(report.to_dict(), schema)


def test_fusion_records_vlm_claim_status_in_violation_evidence():
    candidate = ViolationCandidate(
        object="red_ball",
        track_id="ball-1",
        category="premature_rebound",
        start_frame=1,
        peak_frame=2,
        end_frame=3,
        reason="reversal before contact",
        repair_instruction="continue falling",
        detector_score=0.9,
        rules=("velocity_reversal_before_contact",),
    )
    report = ResultFusion(FusionConfig(detector_weight=0.7, vlm_weight=0.3)).fuse(
        (candidate,),
        {0: (1, 2, 3)},
        {0: VLMReview(score=0.8, claim_status="confirmed")},
    )

    assert report.violations[0].evidence["vlm_claim_status"] == "confirmed"
    assert report.violations[0].evidence["candidate_index"] == 0


def test_vlm_rejected_candidate_is_removed_even_when_rule_score_is_high():
    candidate = ViolationCandidate(
        object="boulder",
        track_id="boulder-1",
        category="object_disappearance",
        start_frame=1,
        peak_frame=8,
        end_frame=16,
        reason="Sparse tracking suggests disappearance.",
        repair_instruction="Keep the object visible.",
        detector_score=0.95,
        rules=("object_persistence",),
    )
    review = VLMReview(
        score=0.0,
        reason="The boulder remains visible; the tracking claim is rejected.",
        repair_instruction="Correct the tracker.",
        claim_status="rejected",
    )

    report = ResultFusion(FusionConfig(detector_weight=0.7, vlm_weight=0.3)).fuse(
        (candidate,), {0: (1, 8, 16)}, {0: review}
    )

    assert report.violations == ()
    assert report.decision == "physical"


def test_vlm_uncertain_candidate_is_not_published_as_hard_violation():
    candidate = ViolationCandidate(
        object="background_door",
        track_id="door-1",
        category="object_disappearance",
        start_frame=10,
        peak_frame=11,
        end_frame=12,
        reason="The tracker loses the door.",
        repair_instruction="Inspect the track.",
        detector_score=0.95,
        rules=("object_persistence",),
    )
    review = VLMReview(
        score=0.9,
        reason="Occlusion prevents a visual decision.",
        repair_instruction="Provide unobstructed frames.",
        claim_status="uncertain",
    )

    report = ResultFusion(FusionConfig()).fuse(
        (candidate,), {0: (10, 11, 12)}, {0: review}
    )

    assert report.violations == ()


def _hard_violation_report(*, supporting_score: float) -> CriticReport:
    violation = Violation(
        object="ball",
        category="object_disappearance",
        start_frame=1,
        peak_frame=2,
        end_frame=3,
        critical_frames=(1, 2, 3),
        reason="The object disappears.",
        repair_instruction="Keep the object visible.",
        evidence={},
    )
    bundles = tuple(
        EvidenceBundle(
            family=family,
            source=f"test_{family}",
            status="available",
            score=supporting_score,
            confidence=1.0,
            coverage=1.0,
        )
        for family in ("pqsg", "checklist", "mechanics")
    )
    return CriticReport(
        is_physical=False,
        physics_score=0.2,
        confidence=0.9,
        violations=(violation,),
        decision="violation",
        evidence_bundles=bundles,
    )


def test_hard_override_requires_counterfactual_physical_support():
    config = FusionConfig()

    assert hard_violation_override_applied(
        _hard_violation_report(supporting_score=0.9), config
    )
    assert not hard_violation_override_applied(
        _hard_violation_report(supporting_score=0.2), config
    )
