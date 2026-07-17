"""Prompt-conditioning diagnostic adapters with explicit leakage boundaries."""

from __future__ import annotations

from pavg_critic.schemas import (
    PhysicsConstraint,
    PhysicsPlan,
    PlannerMetadata,
    SchemaError,
)


class OracleRulePhysicsPlanner:
    """Append human rules to a byte-cached normal model plan for diagnostics."""

    def __init__(self, model_planner, rules, *, model_id: str) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        self.model_planner = model_planner
        self.rules = tuple(str(rule) for rule in rules)
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        partial_plan: PhysicsPlan | None = None,
    ) -> PhysicsPlan:
        plan = self.model_planner.generate(prompt, partial_plan)
        existing_ids = {item.id for item in plan.physics_constraints}
        oracle_ids = {
            f"oracle-rule-{index}" for index in range(len(self.rules))
        }
        collisions = sorted(existing_ids & oracle_ids)
        if collisions:
            raise SchemaError(f"oracle constraint ID collision: {collisions}")
        objects = plan.objects or ("scene",)
        constraints = plan.physics_constraints + tuple(
            PhysicsConstraint(
                id=f"oracle-rule-{index}",
                domain="oracle_natural_language",
                subjects=objects,
                expectation=rule,
            )
            for index, rule in enumerate(self.rules)
        )
        result = PhysicsPlan(
            objects=objects,
            expected_events=plan.expected_events,
            relations=plan.relations,
            physics_constraints=constraints,
            planner_metadata=PlannerMetadata(
                source="explicit",
                confidence=1.0,
                fallback_used=False,
                model=self.model_id,
            ),
        )
        result.validate_references()
        return result
