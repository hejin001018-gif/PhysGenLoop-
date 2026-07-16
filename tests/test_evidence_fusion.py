"""统一证据包、覆盖率校准和三态决策。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from pavg_critic import CriticRequest, FrameState, PhysicsCritic
from pavg_critic.config import CriticConfig, FusionConfig, QuestionGraphConfig, TrajectoryConfig
from pavg_critic.fusion import ResultFusion
from pavg_critic.schemas import PhysicsPlan, ViolationCandidate, VLMReview


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
