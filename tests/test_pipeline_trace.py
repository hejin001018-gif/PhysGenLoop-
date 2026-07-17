"""Integration contracts for real pipeline and PQSG trace boundaries."""

from __future__ import annotations

from copy import deepcopy

import pytest

import pavg_critic.pipeline as pipeline_module
from pavg_critic.config import (
    ChecklistConfig,
    CriticConfig,
    MechanicsConfig,
    QuestionGraphConfig,
    TrajectoryConfig,
)
from pavg_critic.execution_trace import TraceRecorder, validate_trace
from pavg_critic.pipeline import PhysicsCritic
from pavg_critic.question_executor import (
    QuestionExecutionContext,
    QuestionGraphExecutor,
)
from pavg_critic.schemas import (
    CriticRequest,
    Event,
    FrameState,
    PhysicsPlan,
    QuestionGraph,
    QuestionNode,
    SchemaError,
    TrackSequence,
    VLMReview,
)


def _question_context() -> QuestionExecutionContext:
    state = FrameState(
        frame=1,
        timestamp_sec=0.1,
        object="red_ball",
        track_id="ball-1",
        center=(10.0, 20.0),
        bbox=(5.0, 15.0, 15.0, 25.0),
    )
    return QuestionExecutionContext(
        tracks=(TrackSequence("ball-1", "red_ball", (state,)),),
        events=(Event("fall", "red_ball", "ball-1", 1, 1, 1, 0.9),),
        candidates=(),
        candidate_keyframes={},
    )


def _question_graph() -> QuestionGraph:
    return QuestionGraph(
        nodes=(
            QuestionNode(
                id="O1",
                category="object",
                question="Is the ball visible?",
                target_objects=("red_ball",),
            ),
            QuestionNode(
                id="A1",
                category="action",
                question="Does the ball fall?",
                parent_ids=("O1",),
                target_objects=("red_ball",),
                expected_events=("fall",),
            ),
        ),
        source="test",
    )


def test_question_executor_observer_receives_each_real_node_result():
    observed = []

    def observe(node, parent_results, result, error, elapsed_ms):
        observed.append((node, parent_results, result, error, elapsed_ms))

    results = QuestionGraphExecutor(
        enabled_rule_categories=("teleportation",),
        rule_pass_confidence=0.75,
    ).execute(_question_graph(), _question_context(), node_observer=observe)

    assert [item[0].id for item in observed] == ["O1", "A1"]
    assert observed[0][1] == {}
    assert observed[1][1] == {"O1": results[0]}
    assert [item[2] for item in observed] == list(results)
    assert all(item[3] is None for item in observed)
    assert all(item[4] >= 0.0 for item in observed)


def test_question_executor_observer_receives_failure_before_reraise():
    class FailingExecutor(QuestionGraphExecutor):
        def _verify(self, node, context):
            raise RuntimeError("node verifier failed")

    observed = []

    def observe(node, parent_results, result, error, elapsed_ms):
        observed.append((node, parent_results, result, error, elapsed_ms))

    executor = FailingExecutor(
        enabled_rule_categories=("teleportation",),
        rule_pass_confidence=0.75,
    )
    with pytest.raises(RuntimeError, match="node verifier failed"):
        executor.execute(_question_graph(), _question_context(), node_observer=observe)

    assert len(observed) == 1
    assert observed[0][0].id == "O1"
    assert observed[0][2] is None
    assert isinstance(observed[0][3], RuntimeError)
    assert observed[0][4] >= 0.0


class _ConfirmingVerifier:
    def verify_many(self, request, candidates, keyframes):
        return {
            index: VLMReview(
                score=0.9,
                reason="The candidate is visible in the evidence window.",
                repair_instruction="Keep the object visible.",
                model="fixed-verifier",
                claim_status="confirmed",
            )
            for index in range(len(candidates))
        }


def _pipeline_states() -> tuple[FrameState, ...]:
    return tuple(
        FrameState(
            frame=frame,
            timestamp_sec=frame / 10,
            object="ball",
            track_id="ball-1",
            center=(50.0, 50.0),
            bbox=(45.0, 45.0, 55.0, 55.0),
            visible=frame == 0,
        )
        for frame in range(5)
    )


def test_detailed_pipeline_trace_has_every_stage_and_preserves_report():
    config = CriticConfig(trajectory=TrajectoryConfig(smoothing_window=1))
    request = CriticRequest(
        video_path="unused.mp4",
        prompt="A ball falls and remains visible.",
        physics_plan=PhysicsPlan(objects=("ball",), expected_events=("fall",)),
    )
    untraced = PhysicsCritic(config, vlm_verifier=_ConfirmingVerifier())
    traced = PhysicsCritic(config, vlm_verifier=_ConfirmingVerifier())
    expected_report = untraced.analyze_detailed(
        request,
        observations=_pipeline_states(),
        floor_y=100.0,
    ).report.to_dict()
    recorder = TraceRecorder()

    artifacts = traced.analyze_detailed(
        request,
        observations=_pipeline_states(),
        floor_y=100.0,
        trace=recorder,
    )

    assert artifacts.report.to_dict() == expected_report
    trace = recorder.to_dict()
    fixed_nodes = [
        node["node_id"]
        for node in trace["nodes"]
        if not node["node_id"].startswith("pqsg_node.")
    ]
    assert fixed_nodes == [
        "request",
        "physics_planner",
        "question_graph",
        "video_observation",
        "trajectory",
        "event_detection",
        "mechanics",
        "rule_engine",
        "temporal_localization",
        "visual_evidence",
        "checklist",
        "keyframe_selection",
        "pqsg_execution",
        "vlm_verification",
        "candidate_fusion",
        "question_scoring",
        "evidence_fusion",
        "final_report",
    ]
    assert any(node["node_id"].startswith("pqsg_node.") for node in trace["nodes"])
    assert all("inputs" in node and "outputs" in node for node in trace["nodes"])
    fusion = next(
        node for node in trace["nodes"] if node["node_id"] == "evidence_fusion"
    )
    assert len(fusion["outputs"]["families"]) == 5
    assert validate_trace(trace).passed

    missing_child = deepcopy(trace)
    missing_child["nodes"] = [
        node
        for node in missing_child["nodes"]
        if node["node_id"] != next(
            item["node_id"]
            for item in missing_child["nodes"]
            if item["node_id"].startswith("pqsg_node.")
        )
    ]
    for sequence, node in enumerate(missing_child["nodes"], start=1):
        node["sequence"] = sequence

    validation = validate_trace(missing_child)

    assert not validation.passed
    assert any(
        check.code == "pqsg.node_coverage" and not check.passed
        for check in validation.checks
    )


def test_disabled_pipeline_modules_remain_visible_as_skipped_nodes():
    config = CriticConfig(
        trajectory=TrajectoryConfig(smoothing_window=1),
        question_graph=QuestionGraphConfig(enabled=False),
        checklist=ChecklistConfig(enabled=False),
        mechanics=MechanicsConfig(enabled=False),
    )
    recorder = TraceRecorder()

    PhysicsCritic(config, vlm_verifier=_ConfirmingVerifier()).analyze_detailed(
        CriticRequest(
            video_path="unused.mp4",
            prompt="A ball remains visible.",
            physics_plan=PhysicsPlan(
                objects=("ball",), expected_events=("fall",)
            ),
        ),
        observations=_pipeline_states(),
        floor_y=100.0,
        trace=recorder,
    )

    nodes = {node["node_id"]: node for node in recorder.to_dict()["nodes"]}
    for node_id in ("question_graph", "mechanics", "checklist", "pqsg_execution", "question_scoring"):
        assert nodes[node_id]["status"] == "skipped"
        assert nodes[node_id]["outputs"]["reason"]


class _FailingStructuredModel:
    model = "failing-model"

    def generate_json(self, *, system_prompt, user_prompt, schema):
        raise SchemaError("invalid structured response")


def test_model_planner_and_question_graph_fallbacks_are_degraded_nodes():
    recorder = TraceRecorder()
    critic = PhysicsCritic(
        CriticConfig(trajectory=TrajectoryConfig(smoothing_window=1)),
        planner_model=_FailingStructuredModel(),
        question_model=_FailingStructuredModel(),
        vlm_verifier=_ConfirmingVerifier(),
    )

    critic.analyze_detailed(
        CriticRequest(
            video_path="unused.mp4",
            prompt="A red ball falls under gravity.",
        ),
        observations=_pipeline_states(),
        floor_y=100.0,
        trace=recorder,
    )

    nodes = {node["node_id"]: node for node in recorder.to_dict()["nodes"]}
    assert nodes["physics_planner"]["status"] == "degraded"
    assert nodes["question_graph"]["status"] == "degraded"
    assert nodes["physics_planner"]["warnings"]
    assert nodes["question_graph"]["warnings"]


class _FailingVerifier:
    def verify_many(self, request, candidates, keyframes):
        raise SchemaError("review schema failed")


def test_vlm_provider_failure_is_a_degraded_node_not_a_silent_success():
    recorder = TraceRecorder()
    critic = PhysicsCritic(
        CriticConfig(trajectory=TrajectoryConfig(smoothing_window=1)),
        vlm_verifier=_FailingVerifier(),
    )

    artifacts = critic.analyze_detailed(
        CriticRequest(
            video_path="unused.mp4",
            prompt="A ball falls and remains visible.",
            physics_plan=PhysicsPlan(
                objects=("ball",), expected_events=("fall",)
            ),
        ),
        observations=_pipeline_states(),
        floor_y=100.0,
        trace=recorder,
    )

    vlm = next(
        node
        for node in recorder.to_dict()["nodes"]
        if node["node_id"] == "vlm_verification"
    )
    assert vlm["status"] == "degraded"
    assert vlm["warnings"]
    assert artifacts.report.diagnostics["provider_failures"]


def test_untraced_pipeline_does_not_build_trace_only_state_summaries(monkeypatch):
    def fail_if_called(states):
        raise AssertionError("trace summary must stay lazy when tracing is disabled")

    monkeypatch.setattr(pipeline_module, "summarize_states", fail_if_called)

    report = PhysicsCritic(
        CriticConfig(trajectory=TrajectoryConfig(smoothing_window=1)),
        vlm_verifier=_ConfirmingVerifier(),
    ).analyze_detailed(
        CriticRequest(
            video_path="unused.mp4",
            prompt="A ball falls.",
            physics_plan=PhysicsPlan(
                objects=("ball",), expected_events=("fall",)
            ),
        ),
        observations=_pipeline_states(),
        floor_y=100.0,
    ).report

    assert report.decision in {"physical", "violation", "unknown"}
