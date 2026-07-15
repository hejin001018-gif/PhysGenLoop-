"""Prompt -> PhysicsPlan 规划器的数据契约与行为测试。"""

from __future__ import annotations

import pytest

from pavg_critic.schemas import (
    PhysicsConstraint,
    PhysicsPlan,
    PhysicsRelation,
    PlannerMetadata,
    SchemaError,
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
