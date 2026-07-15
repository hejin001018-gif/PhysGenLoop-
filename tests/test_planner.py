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
