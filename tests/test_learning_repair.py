import json
from collections import Counter

from pavg_critic.schemas import CriticReport, Violation

from physgenloop import (
    LearningRepairAgent,
    LearningRepairPromptAdapter,
)
from physgenloop.learning_repair import (
    ACTION_ORDER,
    AgentConfig,
    HeuristicRepairPolicy,
    PolicyPrediction,
    RepairAction,
    RepairExample,
    RepairMemory,
    RepairTrial,
    ReportFeatureEncoder,
    audit_dataset,
    build_labeled_example,
    grouped_split,
    load_repair_manifest,
    write_repair_manifest,
)
from physgenloop.learning_repair.cli import main as repair_cli
from Blender_video.scripts.finalize_repair_shard import (
    ACTION_BY_CATEGORY,
    context_cases,
)


def _report(category="gravity_violation", *, score=0.2, coverage=0.9):
    return CriticReport(
        is_physical=False,
        decision="violation",
        physics_score=score,
        confidence=0.9,
        coverage=coverage,
        violations=(
            Violation(
                object="ball",
                category=category,
                start_frame=10,
                peak_frame=15,
                end_frame=20,
                critical_frames=(10, 15, 20),
                reason="physical error",
                repair_instruction="Restore physically plausible motion.",
                evidence={},
            ),
        ),
    )


def _example(sample_id, group_id, action, *, report=None, successful=True):
    return RepairExample(
        sample_id=sample_id,
        group_id=group_id,
        prompt="A ball falls.",
        critic_report=(report or _report()).to_dict(),
        target_action=action,
        strategy="test strategy",
        before_score=0.2,
        after_score=0.9 if successful else 0.1,
        successful=successful,
    )


def test_feature_encoder_is_versioned_deterministic_and_normalizes_categories():
    encoder = ReportFeatureEncoder()
    first = encoder.encode(_report("reverse_gravity"))
    second = encoder.encode(_report("reverse_gravity"))
    assert first == second
    assert len(first) == encoder.dimension == len(encoder.feature_names)
    category_index = encoder.feature_names.index("category.gravity_violation")
    assert first[category_index] == 1.0


def test_manifest_round_trip_and_grouped_split_prevents_leakage(tmp_path):
    records = (
        _example("a-normal", "scene-a", RepairAction.PROMPT_REPAIR),
        _example("a-broken", "scene-a", RepairAction.LOCAL_EDITING),
        _example("b", "scene-b", RepairAction.GLOBAL_REGENERATION),
        _example("c", "scene-c", RepairAction.REJECT),
    )
    path = tmp_path / "repair.jsonl"
    write_repair_manifest(records, path)
    loaded = load_repair_manifest(path)
    assert loaded == records
    split = grouped_split(loaded, validation_fraction=0.25, test_fraction=0.25, seed=7)
    assert audit_dataset(split).group_leakage == {}
    scene_a_splits = {item.split for item in split if item.group_id == "scene-a"}
    assert len(scene_a_splits) == 1


def test_repair_memory_prioritizes_similar_successful_experience():
    matching = _example("match", "g1", RepairAction.PROMPT_REPAIR)
    failed = _example(
        "failed",
        "g2",
        RepairAction.LOCAL_EDITING,
        successful=False,
    )
    memory = RepairMemory((matching, failed))
    matches = memory.retrieve(_report(), k=3)
    assert [item.example.sample_id for item in matches] == ["match"]
    distribution = memory.action_distribution(matches)
    assert distribution[RepairAction.PROMPT_REPAIR] == 1.0


def test_trial_labeling_applies_quality_gates_before_reward():
    low_quality = RepairTrial(
        RepairAction.LOCAL_EDITING,
        "cheap but artifacted",
        after_score=0.99,
        semantic_score=0.95,
        quality_score=0.2,
        cost=0.1,
        artifacts={},
    )
    valid = RepairTrial(
        RepairAction.GLOBAL_REGENERATION,
        "stable regeneration",
        after_score=0.9,
        semantic_score=0.9,
        quality_score=0.85,
        cost=0.5,
        artifacts={"repaired_video": "repaired.mp4"},
    )
    example = build_labeled_example(
        sample_id="labeled",
        group_id="scene",
        prompt="p",
        critic_report=_report(),
        before_score=0.2,
        trials=(low_quality, valid),
    )
    assert example.target_action is RepairAction.GLOBAL_REGENERATION
    assert example.artifacts["repaired_video"] == "repaired.mp4"


def test_agent_combines_policy_and_memory_but_keeps_structured_action():
    class UniformPolicy:
        def predict(self, critic_report, *, context=None):
            return PolicyPrediction(
                {action: 0.25 for action in ACTION_ORDER},
                expected_gain=0.4,
                model_id="uniform",
            )

    memory = RepairMemory(
        (_example("local", "g", RepairAction.LOCAL_EDITING),)
    )
    agent = LearningRepairAgent(
        UniformPolicy(),
        memory=memory,
        config=AgentConfig(memory_weight=1.0, minimum_policy_confidence=0.2),
    )
    decision = agent.decide(critic_report=_report("surface_penetration"))
    assert decision.action is RepairAction.LOCAL_EDITING
    assert decision.memory_ids == ("local",)
    assert decision.parameters["execution_backend_required"] == "local_video_editor"


def test_heuristic_policy_and_prompt_adapter_preserve_legacy_prompt_integration():
    adapter = LearningRepairPromptAdapter(
        LearningRepairAgent(HeuristicRepairPolicy())
    )
    prompt, decision = adapter.repair_with_decision(
        prompt="A ball falls.", report=_report("gravity_violation")
    )
    assert decision.action is RepairAction.PROMPT_REPAIR
    assert prompt.startswith("A ball falls.\nPhysics correction:")


def test_repair_cli_validates_manifest(tmp_path, capsys):
    path = tmp_path / "repair.jsonl"
    write_repair_manifest(
        (_example("sample", "scene", RepairAction.PROMPT_REPAIR),), path
    )
    assert repair_cli(("validate", "--manifest", str(path))) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["sample_count"] == 1


def test_hard_profile_expands_context_cases_without_selecting_unavailable_action():
    categories = (
        "physical",
        "premature_rebound",
        "premature_rebound",
        "surface_penetration",
        "surface_penetration",
        "object_disappearance",
        "object_disappearance",
        "reverse_gravity",
        "reverse_gravity",
        "teleportation",
        "teleportation",
        "unknown",
        "multi_violation",
    )
    cases = [
        case
        for category in categories
        for case in context_cases(category, ACTION_BY_CATEGORY[category], "hard-v1")
    ]
    assert len(cases) == 35
    assert Counter(action.value for _name, action, _strategy, _context in cases) == {
        "prompt_repair": 4,
        "global_regeneration": 11,
        "local_editing": 6,
        "reject": 14,
    }
    assert all(context.action_available(action) for _name, action, _strategy, context in cases)
    assert [
        case[1]
        for category in categories
        for case in context_cases(category, ACTION_BY_CATEGORY[category], "hard-v1.1")
    ] == [case[1] for case in cases]
