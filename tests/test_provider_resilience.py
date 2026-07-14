"""可选模型失败时，确定性 Critic 主链必须继续返回可审计报告。"""

from __future__ import annotations

from pavg_critic import CriticRequest, FrameState, PhysicsCritic
from pavg_critic.schemas import PhysicsPlan


class TimeoutQuestionModel:
    def generate_json(self, **kwargs):
        raise TimeoutError("question API timeout")


class TimeoutVLMVerifier:
    def verify(self, request, candidate, critical_frames):
        raise TimeoutError("VLM API timeout")


def _state(frame, y, velocity):
    return FrameState(
        frame=frame,
        timestamp_sec=frame / 10,
        object="red_ball",
        track_id="ball-1",
        center=(50, y),
        bbox=(45, y - 5, 55, y + 5),
        velocity=velocity,
    )


def test_question_api_timeout_falls_back_to_template_graph():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=("fall",)),
    )

    artifacts = PhysicsCritic(question_model=TimeoutQuestionModel()).analyze_detailed(
        request,
        observations=(_state(0, 10, (0, 10)), _state(1, 20, (0, 10))),
        floor_y=100,
    )

    assert artifacts.question_graph is not None
    assert artifacts.question_graph.source == "physics_plan_template"
    assert artifacts.report.diagnostics["provider_failures"][0]["stage"] == "question_graph"


def test_vlm_timeout_keeps_rule_violation_and_marks_provider_failure():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=("fall",)),
    )
    states = (
        _state(0, 40, (0, 100)),
        _state(1, 55, (0, 100)),
        _state(2, 48, (0, -100)),
        _state(3, 38, (0, -100)),
    )

    report = PhysicsCritic(vlm_verifier=TimeoutVLMVerifier()).analyze(
        request,
        observations=states,
        floor_y=100,
    )

    assert report.decision == "violation"
    assert report.violations
    failures = report.diagnostics["provider_failures"]
    assert any(item["stage"] == "vlm_review" for item in failures)
