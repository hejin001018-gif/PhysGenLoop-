"""Prompt -> PhysicsPlan 规划器的数据契约与行为测试。"""

from __future__ import annotations

import pytest

from pavg_critic.schemas import (
    CriticRequest,
    PhysicsConstraint,
    PhysicsPlan,
    PhysicsRelation,
    PlannerMetadata,
    SchemaError,
    QuestionGraph,
)


class FakePlanModel:
    model = "fake-plan-model"

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.payload


class SequencedStructuredModel:
    model = "shared-fake-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, **kwargs):
        properties = kwargs["schema"].get("properties", {})
        stage = "physics_plan" if "objects" in properties else "question_graph"
        self.calls.append(stage)
        return self.responses.pop(0)


def _model_plan_payload():
    return {
        "objects": ["red_ball", "floor"],
        "expected_events": ["fall", "floor_contact"],
        "relations": [
            {
                "id": "R1",
                "subject": "red_ball",
                "relation": "expected_to_collide_with",
                "object": "floor",
            }
        ],
        "physics_constraints": [
            {
                "id": "C1",
                "domain": "gravity",
                "subjects": ["red_ball"],
                "expectation": "downward_acceleration",
            }
        ],
    }


def test_model_planner_parses_valid_structured_plan():
    from pavg_critic.planner import ModelPhysicsPlanner

    model = FakePlanModel(payload=_model_plan_payload())
    plan = ModelPhysicsPlanner(model).generate("A red ball falls to the floor.")

    assert plan.objects == ("red_ball", "floor")
    assert plan.planner_metadata.source == "model"
    assert plan.planner_metadata.confidence == 0.8
    assert plan.planner_metadata.model == "fake-plan-model"
    assert model.calls[0]["schema"]["additionalProperties"] is False


def test_resolver_falls_back_after_timeout():
    from pavg_critic.planner import (
        ModelPhysicsPlanner,
        PhysicsPlanResolver,
        TemplatePhysicsPlanner,
    )

    model = FakePlanModel(error=TimeoutError("planner timeout"))
    resolver = PhysicsPlanResolver(
        planner=ModelPhysicsPlanner(model),
        fallback=TemplatePhysicsPlanner(),
        fallback_on_provider_error=True,
    )

    resolution = resolver.resolve(
        CriticRequest(video_path="unused.mp4", prompt="A red ball falls.")
    )

    assert resolution.plan.planner_metadata.source == "template_fallback"
    assert resolution.plan.planner_metadata.confidence == 0.4
    assert resolution.plan.planner_metadata.fallback_used is True
    assert resolution.provider_failure["stage"] == "physics_planner"
    assert resolution.provider_failure["error_type"] == "TimeoutError"


def test_resolver_skips_model_for_complete_explicit_core():
    from pavg_critic.planner import ModelPhysicsPlanner, PhysicsPlanResolver

    model = FakePlanModel(error=AssertionError("model must not be called"))
    resolver = PhysicsPlanResolver(ModelPhysicsPlanner(model))
    request = CriticRequest(
        video_path="unused.mp4",
        prompt="ignored",
        physics_plan=PhysicsPlan(
            objects=("custom_ball",), expected_events=("custom_motion",)
        ),
    )

    resolution = resolver.resolve(request)

    assert model.calls == []
    assert resolution.plan.objects == ("custom_ball",)
    assert resolution.plan.planner_metadata.source == "explicit"


def test_resolver_fills_only_empty_core_and_explicit_extension_id_wins():
    from pavg_critic.planner import ModelPhysicsPlanner, PhysicsPlanResolver

    payload = _model_plan_payload()
    payload["objects"] = ["custom_ball", "floor"]
    payload["relations"][0]["subject"] = "custom_ball"
    payload["physics_constraints"][0]["subjects"] = ["custom_ball"]
    model = FakePlanModel(payload=payload)
    explicit = PhysicsPlan(
        objects=("custom_ball", "floor"),
        relations=(
            PhysicsRelation("R1", "custom_ball", "initially_above", "floor"),
        ),
    )

    resolution = PhysicsPlanResolver(ModelPhysicsPlanner(model)).resolve(
        CriticRequest(
            video_path="unused.mp4",
            prompt="A red ball falls to the floor.",
            physics_plan=explicit,
        )
    )

    assert resolution.plan.objects == ("custom_ball", "floor")
    assert resolution.plan.expected_events == ("fall", "floor_contact")
    assert resolution.plan.relations[0].relation == "initially_above"
    assert resolution.plan.planner_metadata.source == "merged"


def test_resolver_filters_generated_extensions_for_discarded_objects():
    from pavg_critic.planner import ModelPhysicsPlanner, PhysicsPlanResolver

    resolution = PhysicsPlanResolver(
        ModelPhysicsPlanner(FakePlanModel(payload=_model_plan_payload()))
    ).resolve(
        CriticRequest(
            video_path="unused.mp4",
            prompt="A red ball falls.",
            physics_plan=PhysicsPlan(objects=("custom_ball", "floor")),
        )
    )

    assert resolution.plan.objects == ("custom_ball", "floor")
    assert resolution.plan.relations == ()
    assert resolution.plan.physics_constraints == ()


def test_invalid_model_references_trigger_template_fallback():
    from pavg_critic.planner import (
        ModelPhysicsPlanner,
        PhysicsPlanResolver,
        TemplatePhysicsPlanner,
    )

    payload = _model_plan_payload()
    payload["objects"] = ["red_ball"]
    resolver = PhysicsPlanResolver(
        ModelPhysicsPlanner(FakePlanModel(payload=payload)),
        fallback=TemplatePhysicsPlanner(),
        fallback_on_provider_error=True,
    )

    resolution = resolver.resolve(
        CriticRequest(video_path="unused.mp4", prompt="A red ball falls.")
    )

    assert resolution.plan.planner_metadata.source == "template_fallback"
    assert resolution.provider_failure["error_type"] == "SchemaError"


def test_null_model_arrays_trigger_template_fallback():
    from pavg_critic.planner import (
        ModelPhysicsPlanner,
        PhysicsPlanResolver,
        TemplatePhysicsPlanner,
    )

    payload = _model_plan_payload()
    payload["relations"] = None
    resolution = PhysicsPlanResolver(
        ModelPhysicsPlanner(FakePlanModel(payload=payload)),
        fallback=TemplatePhysicsPlanner(),
        fallback_on_provider_error=True,
    ).resolve(CriticRequest(video_path="unused.mp4", prompt="A ball falls."))

    assert resolution.plan.planner_metadata.source == "template_fallback"
    assert resolution.provider_failure["error_type"] == "SchemaError"


def test_invalid_explicit_plan_is_not_silently_repaired():
    invalid = PhysicsPlan(
        objects=("ball",),
        physics_constraints=(
            PhysicsConstraint(
                id="C1",
                domain="contact",
                subjects=("ball", "floor"),
                expectation="no_interpenetration",
            ),
        ),
    )

    with pytest.raises(SchemaError, match="floor"):
        CriticRequest(video_path="unused.mp4", physics_plan=invalid)


def test_pipeline_reuses_question_model_for_planner_when_planner_model_missing():
    from pavg_critic import PhysicsCritic

    model = SequencedStructuredModel(_model_plan_payload(), {"nodes": []})
    artifacts = PhysicsCritic(question_model=model).analyze_detailed(
        CriticRequest(video_path="unused.mp4", prompt="A red ball falls."),
        observations=(),
        floor_y=100,
    )

    assert model.calls == ["physics_plan", "question_graph"]
    assert artifacts.resolved_request.physics_plan.objects == ("red_ball", "floor")
    assert artifacts.report.diagnostics["planner"]["source"] == "model"


def test_pipeline_prefers_dedicated_planner_model():
    from pavg_critic import PhysicsCritic

    planner_model = FakePlanModel(payload=_model_plan_payload())
    question_model = SequencedStructuredModel({"nodes": []})
    PhysicsCritic(
        planner_model=planner_model,
        question_model=question_model,
    ).analyze_detailed(
        CriticRequest(video_path="unused.mp4", prompt="A red ball falls."),
        observations=(),
        floor_y=100,
    )

    assert len(planner_model.calls) == 1
    assert question_model.calls == ["question_graph"]


def test_pipeline_complete_explicit_plan_skips_dedicated_planner_model():
    from pavg_critic import PhysicsCritic

    planner_model = FakePlanModel(error=AssertionError("must not run"))
    artifacts = PhysicsCritic(planner_model=planner_model).analyze_detailed(
        CriticRequest(
            video_path="unused.mp4",
            physics_plan=PhysicsPlan(
                objects=("ball",), expected_events=("projectile",)
            ),
        ),
        observations=(),
        floor_y=100,
    )

    assert planner_model.calls == []
    assert artifacts.resolved_request.physics_plan.planner_metadata.source == "explicit"


def test_pipeline_passes_one_resolved_request_to_downstream_plugins():
    from pavg_critic import PhysicsCritic

    class RecordingGraphGenerator:
        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return QuestionGraph(source="recording_graph")

    class RecordingExtractor:
        def __init__(self):
            self.requests = []

        def extract(self, request, tracks, events):
            self.requests.append(request)
            return ()

    graph = RecordingGraphGenerator()
    extractor = RecordingExtractor()
    artifacts = PhysicsCritic(
        question_graph_generator=graph,
        visual_evidence_extractors=(extractor,),
    ).analyze_detailed(
        CriticRequest(video_path="unused.mp4", prompt="A ball falls."),
        observations=(),
        floor_y=100,
    )

    assert graph.requests[0] is artifacts.resolved_request
    assert extractor.requests[0] is artifacts.resolved_request
    assert artifacts.resolved_request.physics_plan.expected_events == (
        "leave_support",
        "fall",
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "A red ball falls from a table, hits the floor, and bounces once.",
        "一个红球从桌子上掉落，接触地面后反弹一次。",
    ],
)
def test_template_planner_builds_fall_contact_rebound_plan(prompt):
    from pavg_critic.planner import TemplatePhysicsPlanner

    plan = TemplatePhysicsPlanner().generate(prompt)

    assert plan.objects == ("red_ball", "table", "floor")
    assert plan.expected_events == (
        "leave_support",
        "fall",
        "floor_contact",
        "rebound",
    )
    assert {item.domain for item in plan.physics_constraints} == {
        "gravity",
        "contact",
        "rebound",
    }
    assert plan.planner_metadata.source == "template"
    assert plan.planner_metadata.confidence == 0.55


def test_template_planner_empty_prompt_returns_empty_plan():
    from pavg_critic.planner import TemplatePhysicsPlanner

    plan = TemplatePhysicsPlanner().generate("")

    assert plan.objects == ()
    assert plan.expected_events == ()
    assert plan.planner_metadata.source == "empty"


def test_template_planner_detects_projectile():
    from pavg_critic.planner import TemplatePhysicsPlanner

    plan = TemplatePhysicsPlanner().generate("A ball is thrown through the air.")

    assert "projectile" in plan.expected_events
    assert any(item.domain == "projectile" for item in plan.physics_constraints)


def test_template_planner_detects_collision():
    from pavg_critic.planner import TemplatePhysicsPlanner

    plan = TemplatePhysicsPlanner().generate("Two balls collide with each other.")

    assert "collision" in plan.expected_events
    assert any(item.domain == "collision" for item in plan.physics_constraints)


def test_template_planner_does_not_invent_numeric_parameters():
    from pavg_critic.planner import TemplatePhysicsPlanner

    payload = TemplatePhysicsPlanner().generate("A ball falls.").to_dict()

    assert "9.8" not in str(payload)
    assert "mass" not in str(payload)


def test_old_physics_plan_remains_valid():
    plan = PhysicsPlan.from_dict(
        {"objects": ["red_ball"], "expected_events": ["fall"]}
    )

    assert plan.objects == ("red_ball",)
    assert plan.expected_events == ("fall",)
    assert plan.relations == ()
    assert plan.physics_constraints == ()
    assert plan.planner_metadata.source == "empty"


def test_extended_physics_plan_parses_and_validates_references():
    plan = PhysicsPlan.from_dict(
        {
            "objects": ["red_ball", "floor"],
            "expected_events": ["fall", "floor_contact"],
            "relations": [
                {
                    "id": "R1",
                    "subject": "red_ball",
                    "relation": "expected_to_collide_with",
                    "object": "floor",
                }
            ],
            "physics_constraints": [
                {
                    "id": "C1",
                    "domain": "contact",
                    "subjects": ["red_ball", "floor"],
                    "condition": "during_contact",
                    "expectation": "no_interpenetration",
                }
            ],
            "planner_metadata": {
                "source": "model",
                "confidence": 0.8,
                "model": "fake-model",
            },
        }
    )

    plan.validate_references()
    assert plan.relations[0].id == "R1"
    assert plan.physics_constraints[0].domain == "contact"
    assert plan.planner_metadata == PlannerMetadata(
        source="model", confidence=0.8, model="fake-model"
    )


def test_plan_rejects_unknown_constraint_subject():
    plan = PhysicsPlan.from_dict(
        {
            "objects": ["red_ball"],
            "physics_constraints": [
                {
                    "id": "C1",
                    "domain": "contact",
                    "subjects": ["red_ball", "floor"],
                    "expectation": "no_interpenetration",
                }
            ],
        }
    )

    with pytest.raises(SchemaError, match="floor"):
        plan.validate_references()


def test_plan_rejects_duplicate_extension_ids():
    with pytest.raises(SchemaError, match="duplicate physics relation id"):
        PhysicsPlan(
            objects=("ball",),
            relations=(
                PhysicsRelation("R1", "ball", "near", "ball"),
                PhysicsRelation("R1", "ball", "above", "ball"),
            ),
        )


def test_planner_confidence_must_be_normalized():
    with pytest.raises(SchemaError, match="planner confidence"):
        PlannerMetadata(source="model", confidence=1.1)


def test_extended_plan_serializes_to_plain_json_data():
    plan = PhysicsPlan(
        objects=("ball",),
        physics_constraints=(
            PhysicsConstraint(
                id="C1",
                domain="gravity",
                subjects=("ball",),
                expectation="downward_acceleration",
            ),
        ),
    )

    payload = plan.to_dict()
    assert payload["physics_constraints"][0]["subjects"] == ["ball"]
    assert payload["planner_metadata"]["source"] == "empty"
