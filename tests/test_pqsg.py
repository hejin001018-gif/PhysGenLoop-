"""PQSG 融合层：图生成、混合和两阶段问答。"""

from __future__ import annotations

import json

from pavg_critic.config import QuestionGraphConfig
from pavg_critic import FrameState, PhysicsCritic
from pavg_critic.pqsg import (
    HybridQuestionGraphGenerator,
    PQSGQuestionGraphGenerator,
    TwoPassQuestionAnswerer,
)
from pavg_critic.question_executor import (
    QuestionExecutionContext,
    QuestionGraphExecutor,
)
from pavg_critic.question_generator import TemplateQuestionGraphGenerator
from pavg_critic.question_graph import QuestionGraphValidator
from pavg_critic.schemas import (
    CriticRequest,
    PhysicsConstraint,
    PhysicsPlan,
    PhysicsRelation,
    Event,
    QuestionGraph,
    QuestionNode,
    TrackSequence,
)


class FakeStructuredModel:
    """按顺序返回结构化响应，并保存调用以验证两阶段协议。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, *, system_prompt, user_prompt, schema):
        self.calls.append((system_prompt, user_prompt, schema))
        return self.responses.pop(0)


def test_validator_accepts_pqsg_object_to_physics_edge():
    graph = QuestionGraph(
        nodes=(
            QuestionNode(id="O1", category="object", question="Is the ball visible?"),
            QuestionNode(
                id="P1",
                category="physics",
                question="Does the ball obey gravity?",
                parent_ids=("O1",),
            ),
        ),
        source="pqsg",
    )

    QuestionGraphValidator().validate(graph)


def test_pqsg_generator_parses_and_validates_model_json():
    model = FakeStructuredModel(
        {
            "nodes": [
                {
                    "id": "O1",
                    "category": "object",
                    "question": "Is the red ball present?",
                    "target_objects": ["red_ball"],
                },
                {
                    "id": "P1",
                    "category": "physics",
                    "question": "Does the red ball accelerate downward?",
                    "parent_ids": ["O1"],
                    "physics_domain": "gravity",
                },
            ]
        }
    )
    request = CriticRequest(video_path="unused.mp4", prompt="A red ball falls.")

    graph = PQSGQuestionGraphGenerator(model).generate(request)

    assert graph.source == "pqsg_model"
    assert [node.id for node in graph.nodes] == ["O1", "P1"]


def test_pqsg_generator_sanitizes_reverse_or_missing_parent_edges():
    model = FakeStructuredModel(
        {
            "nodes": [
                {
                    "id": "P1",
                    "category": "physics",
                    "question": "Does the rock obey downhill dynamics?",
                },
                {
                    "id": "A1",
                    "category": "action",
                    "question": "Does the rock start rolling?",
                    "parent_ids": ["P1", "missing", "P1"],
                },
            ]
        }
    )

    graph = PQSGQuestionGraphGenerator(model).generate(
        CriticRequest(video_path="unused.mp4", prompt="A rock rolls downhill.")
    )

    assert graph.source == "pqsg_model_sanitized"
    assert graph.nodes[1].parent_ids == ()
    QuestionGraphValidator().validate(graph)


def test_question_executor_matches_generic_plan_names_and_downhill_roll_alias():
    graph = QuestionGraph(
        nodes=(
            QuestionNode(
                id="O1",
                category="object",
                question="Is the rock visible?",
                target_objects=("rock",),
            ),
            QuestionNode(
                id="A1",
                category="action",
                question="Does it roll downhill?",
                parent_ids=("O1",),
                target_objects=("rock",),
                expected_events=("rock_starts_rolling_down_slope",),
            ),
            QuestionNode(
                id="P1",
                category="physics",
                question="Does it remain continuous?",
                parent_ids=("A1",),
                target_objects=("rock",),
                rule_ids=("bounded_interframe_displacement",),
            ),
        ),
        source="test",
    )
    state = FrameState(
        frame=1,
        timestamp_sec=0.1,
        object="large_gray_rock",
        track_id="sam2:4",
        center=(10, 20),
        bbox=(5, 15, 15, 25),
    )
    context = QuestionExecutionContext(
        tracks=(
            TrackSequence(
                track_id="sam2:4", object="large_gray_rock", states=(state,)
            ),
        ),
        events=(
            Event(
                "fall",
                "large_gray_rock",
                "sam2:4",
                1,
                1,
                1,
                0.9,
            ),
        ),
        candidates=(),
        candidate_keyframes={},
    )

    results = QuestionGraphExecutor(
        enabled_rule_categories=("teleportation",),
        rule_pass_confidence=0.75,
    ).execute(graph, context)

    assert [result.status for result in results] == ["yes", "yes", "yes"]


def test_pqsg_generator_receives_relations_and_constraints():
    model = FakeStructuredModel({"nodes": []})
    request = CriticRequest(
        video_path="unused.mp4",
        prompt="A ball hits the floor.",
        physics_plan=PhysicsPlan(
            objects=("ball", "floor"),
            expected_events=("floor_contact",),
            relations=(
                PhysicsRelation(
                    "R1", "ball", "expected_to_collide_with", "floor"
                ),
            ),
            physics_constraints=(
                PhysicsConstraint(
                    id="C1",
                    domain="contact",
                    subjects=("ball", "floor"),
                    condition="during_contact",
                    expectation="no_interpenetration",
                ),
            ),
        ),
    )

    PQSGQuestionGraphGenerator(model).generate(request)

    user_payload = json.loads(model.calls[0][1])
    assert user_payload["relations"][0]["relation"] == "expected_to_collide_with"
    assert user_payload["physics_constraints"][0]["domain"] == "contact"
    assert "planner_metadata" not in user_payload
    schema = model.calls[0][2]
    node_properties = schema["properties"]["nodes"]["items"]["properties"]
    assert "fall" in node_properties["expected_events"]["items"]["enum"]
    assert (
        "object_persistence" in node_properties["rule_ids"]["items"]["enum"]
    )


def test_hybrid_graph_keeps_template_and_pqsg_nodes_without_id_collisions():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=("fall",)),
    )
    template = TemplateQuestionGraphGenerator(QuestionGraphConfig())
    pqsg = PQSGQuestionGraphGenerator(
        FakeStructuredModel(
            {
                "nodes": [
                    {
                        "id": "O1",
                        "category": "object",
                        "question": "Does the ball retain its round shape?",
                        "target_objects": ["red_ball"],
                    }
                ]
            }
        )
    )

    graph = HybridQuestionGraphGenerator(template, pqsg).generate(request)

    assert graph.source == "pavg_hybrid_template_pqsg"
    assert len({node.id for node in graph.nodes}) == len(graph.nodes)
    assert any(node.id.startswith("Q_") for node in graph.nodes)
    assert any(node.id.startswith("O") for node in graph.nodes)


def test_two_pass_qa_separates_reasoning_from_forced_answer():
    model = FakeStructuredModel(
        {"reasoning": "The tracked center moves downward in every visible frame."},
        {"answer": "yes", "confidence": 0.85},
    )

    answer = TwoPassQuestionAnswerer(model).answer(
        question="Does the ball move downward?",
        evidence="centers_y=[10, 20, 30]",
    )

    assert answer.answer == "yes"
    assert answer.confidence == 0.85
    assert len(model.calls) == 2
    assert "reasoning" in model.calls[1][1]


def test_physics_critic_automatically_builds_hybrid_graph_when_model_is_injected():
    model = FakeStructuredModel(
        {
            "objects": ["red_ball"],
            "expected_events": ["fall"],
            "relations": [],
            "physics_constraints": [],
        },
        {
            "nodes": [
                {
                    "id": "O1",
                    "category": "object",
                    "question": "Is the ball present?",
                    "target_objects": ["red_ball"],
                }
            ]
        }
    )
    critic = PhysicsCritic(question_model=model)
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=()),
    )
    observation = FrameState(
        frame=0,
        timestamp_sec=0,
        object="red_ball",
        center=(10, 10),
        bbox=(5, 5, 15, 15),
    )

    artifacts = critic.analyze_detailed(request, observations=(observation,), floor_y=100)

    assert artifacts.question_graph is not None
    assert artifacts.question_graph.source == "pavg_hybrid_template_pqsg"
    assert any(result.node_id == "Q_O1" for result in artifacts.node_results)
    assert len(model.calls) == 2
