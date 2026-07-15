"""PQSG 融合层：图生成、混合和两阶段问答。"""

from __future__ import annotations

from pavg_critic.config import QuestionGraphConfig
from pavg_critic import FrameState, PhysicsCritic
from pavg_critic.pqsg import (
    HybridQuestionGraphGenerator,
    PQSGQuestionGraphGenerator,
    TwoPassQuestionAnswerer,
)
from pavg_critic.question_generator import TemplateQuestionGraphGenerator
from pavg_critic.question_graph import QuestionGraphValidator
from pavg_critic.schemas import CriticRequest, PhysicsPlan, QuestionGraph, QuestionNode


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
