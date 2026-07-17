"""Frozen prompt-diagnostic adapters for the complete PAVG benchmark."""

from __future__ import annotations

import pytest

from pavg_critic.benchmarking.prompt_diagnostics import OracleRulePhysicsPlanner
from pavg_critic.schemas import (
    PhysicsConstraint,
    PhysicsPlan,
    PhysicsRelation,
    PlannerMetadata,
    SchemaError,
)


class FixedPlanner:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def generate(self, prompt, partial_plan=None):
        self.calls.append((prompt, partial_plan))
        return self.plan


def test_oracle_plan_preserves_model_content_and_appends_exact_rules():
    model_plan = PhysicsPlan(
        objects=("ball", "floor"),
        expected_events=("fall", "contact"),
        relations=(
            PhysicsRelation(
                id="above",
                relation="above",
                subject="ball",
                object="floor",
            ),
        ),
        physics_constraints=(
            PhysicsConstraint(
                id="model-rule",
                domain="gravity",
                subjects=("ball",),
                expectation="The ball falls.",
            ),
        ),
    )
    planner = FixedPlanner(model_plan)

    plan = OracleRulePhysicsPlanner(
        planner,
        ("rule a", "rule b"),
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    ).generate("prompt")

    assert plan.objects == model_plan.objects
    assert plan.expected_events == model_plan.expected_events
    assert plan.relations == model_plan.relations
    assert plan.physics_constraints[:1] == model_plan.physics_constraints
    assert [item.id for item in plan.physics_constraints[-2:]] == [
        "oracle-rule-0",
        "oracle-rule-1",
    ]
    assert [item.expectation for item in plan.physics_constraints[-2:]] == [
        "rule a",
        "rule b",
    ]
    assert all(
        item.domain == "oracle_natural_language"
        for item in plan.physics_constraints[-2:]
    )
    assert plan.planner_metadata == PlannerMetadata(
        source="explicit",
        confidence=1.0,
        fallback_used=False,
        model="Qwen/Qwen3-VL-8B-Instruct",
    )


def test_oracle_plan_uses_scene_only_when_model_has_no_objects():
    plan = OracleRulePhysicsPlanner(
        FixedPlanner(PhysicsPlan()),
        ("The scene remains coherent.",),
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    ).generate("prompt")

    assert plan.objects == ("scene",)
    assert plan.physics_constraints[0].subjects == ("scene",)


def test_oracle_plan_rejects_stable_id_collision():
    model_plan = PhysicsPlan(
        objects=("ball",),
        physics_constraints=(
            PhysicsConstraint(
                id="oracle-rule-0",
                domain="model",
                subjects=("ball",),
                expectation="Existing model constraint.",
            ),
        ),
    )

    with pytest.raises(SchemaError, match="ID collision"):
        OracleRulePhysicsPlanner(
            FixedPlanner(model_plan),
            ("oracle rule",),
            model_id="Qwen/Qwen3-VL-8B-Instruct",
        ).generate("prompt")
