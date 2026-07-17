"""Integration contracts for real pipeline and PQSG trace boundaries."""

from __future__ import annotations

import pytest

from pavg_critic.question_executor import (
    QuestionExecutionContext,
    QuestionGraphExecutor,
)
from pavg_critic.schemas import (
    Event,
    FrameState,
    QuestionGraph,
    QuestionNode,
    TrackSequence,
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

