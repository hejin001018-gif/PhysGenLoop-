import inspect

from generators.wanphysics.adapter import WanSubprocessGenerator
from generators.wanphysics.v2.runner import ActionAwareRunnerV2
from physgenloop.learning_repair.contracts import ExecutionRequest, ScoreBundle


def test_online_interfaces_do_not_expose_physics_plan():
    assert "physics_plan" not in inspect.signature(WanSubprocessGenerator.generate).parameters
    assert "physics_plan" not in inspect.signature(ActionAwareRunnerV2.run).parameters
    assert "physics_plan" not in ExecutionRequest.__dataclass_fields__


def test_score_bundle_has_dual_semantic_scores():
    bundle = ScoreBundle(
        physics=0.8,
        semantic=0.9,
        original_prompt_semantic=0.85,
        quality=0.8,
    )
    assert bundle.to_dict()["original_prompt_semantic"] == 0.85


def test_decision_schema_is_v2():
    from physgenloop.learning_repair.contracts import DECISION_SCHEMA_VERSION

    assert DECISION_SCHEMA_VERSION == "learning-repair-decision/2.0"
