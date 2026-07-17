"""Audit-only artifacts exposed by the complete critic pipeline."""

from __future__ import annotations

import json

from pavg_critic.config import CriticConfig, QuestionGraphConfig, TrajectoryConfig
from pavg_critic.pipeline import PhysicsCritic
from pavg_critic.schemas import CriticRequest, FrameState, PhysicsPlan, VLMReview


class FixedVerifier:
    def verify_many(self, request, candidates, keyframes):
        return {
            index: VLMReview(
                score=0.9,
                reason="The disappearance is visible in the selected frames.",
                repair_instruction="Keep the object visible.",
                model="fixed-verifier",
                claim_status="confirmed",
            )
            for index in range(len(candidates))
        }


def _state(frame: int, *, visible: bool) -> FrameState:
    return FrameState(
        frame=frame,
        timestamp_sec=frame / 10,
        object="ball",
        track_id="ball-1",
        center=(50.0, 50.0),
        bbox=(45.0, 45.0, 55.0, 55.0),
        visible=visible,
    )


def test_detailed_pipeline_exposes_fusion_keyframes_and_reviews():
    critic = PhysicsCritic(
        CriticConfig(
            trajectory=TrajectoryConfig(smoothing_window=1),
            question_graph=QuestionGraphConfig(enabled=False),
        ),
        vlm_verifier=FixedVerifier(),
    )
    request = CriticRequest(
        video_path="unused.mp4",
        prompt="A ball remains visible.",
        physics_plan=PhysicsPlan(objects=("ball",)),
    )
    states = (
        _state(0, visible=True),
        _state(1, visible=False),
        _state(2, visible=False),
        _state(3, visible=False),
    )

    artifacts = critic.analyze_detailed(request, observations=states, floor_y=100)

    assert artifacts.keyframes
    assert artifacts.reviews
    assert all(review is not None for review in artifacts.reviews.values())
    assert artifacts.report.diagnostics["pre_evidence_fusion"]["decision"] in {
        "physical",
        "violation",
        "unknown",
    }
    assert artifacts.report.diagnostics["hard_violation_override"] is bool(
        artifacts.report.violations
    )
    json.dumps(artifacts.to_dict(), allow_nan=False)
