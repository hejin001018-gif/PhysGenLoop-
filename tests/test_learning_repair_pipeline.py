import json

import pytest

from pavg_critic.schemas import CriticReport, PhysicsPlan, Violation
from physgenloop import EvidenceAwareSelector, InstructionPromptRepairer
from physgenloop.contracts import CandidateEvaluation, GeneratedCandidate
from physgenloop.learning_repair.contracts import (
    ACTION_ORDER,
    RepairAction,
    RepairContext,
    RepairExample,
)
from physgenloop.learning_repair.baselines import (
    CategoryOnlyPolicy,
    HeuristicDecisionPolicy,
)
from physgenloop.learning_repair.campaign import (
    ActualTrialCampaign,
    RewardSpec,
)
from physgenloop.learning_repair.cloud_campaign import (
    CloudBackendBundle,
    run_frozen_campaign,
)
from physgenloop.learning_repair.compatibility import (
    CompatibilityError,
    CompatibilityManifest,
    verify_proxy_baseline,
)
from physgenloop.learning_repair.contracts import (
    LearningTargetV1,
    LocalEditTarget,
    RepairDecision,
)
from physgenloop.learning_repair.evaluation import (
    audit_targets,
    closed_loop_metrics,
    compare_policies,
)
from physgenloop.learning_repair.executors import (
    ExecutorRegistry,
    GlobalRegenerationExecutor,
    LocalEditingExecutor,
    PromptRepairExecutor,
    RejectExecutor,
)
from physgenloop.learning_repair.manifests import (
    CampaignItem,
    FrozenCampaignManifest,
    validate_campaign_artifacts,
)
from physgenloop.learning_repair.memory_policy import (
    ActualTrialMemory,
    MemoryValuePredictor,
)
from physgenloop.learning_repair.recording import (
    JsonlTrialRecorder,
    VersionedMemoryWriter,
    read_trials,
)
from physgenloop.learning_repair.proxy_adapter import adapt_proxy_example
from physgenloop.learning_repair.runner import (
    LearningRepairLoopRunner,
    RunnerConfig,
)
from physgenloop.learning_repair.value_policy import (
    ActionValuePrediction,
    ActionValueDecisionPolicy,
)
from physgenloop.learning_repair.value_training import assign_group_splits


def test_proxy_adapter_preserves_partial_feedback_and_never_fakes_trials():
    example = RepairExample(
        sample_id="proxy-1",
        group_id="group-1",
        prompt="A ball falls.",
        critic_report=_report(category="surface_penetration").to_dict(),
        target_action=RepairAction.LOCAL_EDITING,
        strategy="repair local contact",
        before_score=0.2,
        after_score=0.9,
        successful=True,
        context=RepairContext(),
        semantic_score=1.0,
        quality_score=1.0,
        repair_cost=0.5,
    )
    target = adapt_proxy_example(example)
    assert target.metadata["proxy_label"] is True
    assert target.source_trial_ids == ()
    assert target.action_rewards[RepairAction.LOCAL_EDITING] is not None
    assert all(
        target.action_rewards[action] is None
        for action in ACTION_ORDER
        if action is not RepairAction.LOCAL_EDITING
    )


def test_proxy_selection_uses_masked_classification_not_unobserved_values():
    class ProxyPrediction:
        compatibility_id = "compat"
        selection_mode = "classification_proxy"

        def predict(self, critic_report, *, context=None):
            return ActionValuePrediction(
                action_probabilities={
                    RepairAction.PROMPT_REPAIR: 0.05,
                    RepairAction.GLOBAL_REGENERATION: 0.10,
                    RepairAction.LOCAL_EDITING: 0.80,
                    RepairAction.REJECT: 0.05,
                },
                per_action_values={
                    RepairAction.PROMPT_REPAIR: 100.0,
                    RepairAction.GLOBAL_REGENERATION: 50.0,
                    RepairAction.LOCAL_EDITING: -100.0,
                    RepairAction.REJECT: 25.0,
                },
                model_id="proxy",
            )

    class Candidate:
        candidate_id = "candidate"

    decision = ActionValueDecisionPolicy(
        ProxyPrediction(), minimum_confidence=0.0
    ).decide(
        critic_report=_report(category="surface_penetration"),
        candidate=Candidate(),
        prompt="p",
        context=RepairContext(),
    )
    assert decision.action is RepairAction.LOCAL_EDITING
    assert decision.parameters["selection"] == "classification_proxy"

    masked = ActionValueDecisionPolicy(
        ProxyPrediction(), minimum_confidence=0.0
    ).decide(
        critic_report=_report(category="surface_penetration"),
        candidate=Candidate(),
        prompt="p",
        context=RepairContext(local_editor_available=False),
    )
    assert masked.action is RepairAction.GLOBAL_REGENERATION


def test_deprecated_pipeline_namespace_redirects_to_canonical_contracts():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        from physgenloop.learning_repair_pipeline.contracts import (
            RepairDecision as CompatibilityDecision,
            RepairDecisionV1,
        )

    assert CompatibilityDecision is RepairDecision
    assert RepairDecisionV1 is RepairDecision


def _report(*, physical=False, score=None, category="gravity_violation"):
    value = (0.9 if physical else 0.2) if score is None else score
    violations = ()
    if not physical:
        violations = (
            Violation(
                object="ball",
                category=category,
                start_frame=2,
                peak_frame=4,
                end_frame=6,
                critical_frames=(2, 4, 6),
                reason="physics error",
                repair_instruction="Restore plausible motion.",
                evidence={"mask_uri": "mask://ball"},
            ),
        )
    return CriticReport(
        is_physical=physical,
        decision="physical" if physical else "violation",
        physics_score=value,
        confidence=0.9,
        coverage=0.9,
        violations=violations,
    )


class _Generator:
    def generate(self, *, prompt, physics_plan, seed):
        return GeneratedCandidate(
            candidate_id=f"candidate-{seed}",
            video_path=f"fake://candidate-{seed}.mp4",
            prompt=prompt,
            seed=seed,
            metadata={"backend": "test"},
        )


class _Critic:
    def evaluate(self, candidate, *, prompt, physics_plan):
        return _report(physical=candidate.seed > 1)


class _Editor:
    def edit(
        self,
        *,
        candidate,
        target,
        instruction,
        critic_report,
        physics_plan,
        seed,
    ):
        assert target.parent_candidate_id == candidate.candidate_id
        return GeneratedCandidate(
            candidate_id=f"local-{seed}",
            video_path=f"fake://local-{seed}.mp4",
            prompt=candidate.prompt,
            seed=seed,
            metadata={"backend": "test-local"},
        )


def _registry():
    generator = _Generator()
    selector = EvidenceAwareSelector()
    return ExecutorRegistry(
        (
            PromptRepairExecutor(
                prompt_rewriter=InstructionPromptRepairer(), generator=generator
            ),
            GlobalRegenerationExecutor(generator=generator),
            LocalEditingExecutor(editor=_Editor()),
            RejectExecutor(selector=selector),
        )
    )


def _compatibility():
    return CompatibilityManifest.load(
        "configs/learning_repair/critic_compatibility_v1.json"
    )


def test_milestone1_archives_proxy_and_fails_fast_on_hash_mismatch(tmp_path):
    result = verify_proxy_baseline(
        "configs/learning_repair/proxy_baseline_1200g_v3.json", root="."
    )
    assert result["valid"] is True
    manifest = _compatibility()
    manifest.verify_files(
        critic_config="configs/default.yaml",
        critic_schema="schemas/critic_output.schema.json",
        feature_schema="configs/learning_repair/feature_schema.json",
    )
    changed = tmp_path / "critic.yaml"
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(CompatibilityError, match="expected"):
        manifest.verify_files(
            critic_config=changed,
            critic_schema="schemas/critic_output.schema.json",
            feature_schema="configs/learning_repair/feature_schema.json",
        )


def test_private_decision_requires_local_coordinates_and_per_action_values():
    target = LocalEditTarget(
        parent_candidate_id="candidate",
        objects=("ball",),
        start_frame=2,
        end_frame=6,
        critical_frames=(4,),
        mask_uri="mask://ball",
    )
    decision = RepairDecision(
        action=RepairAction.LOCAL_EDITING,
        confidence=0.8,
        instruction="edit",
        action_probabilities={item: 0.25 for item in ACTION_ORDER},
        per_action_values={item: float(index) for index, item in enumerate(ACTION_ORDER)},
        local_target=target,
    )
    assert RepairDecision.from_dict(decision.to_dict()) == decision
    with pytest.raises(ValueError, match="local_target"):
        RepairDecision(
            action=RepairAction.LOCAL_EDITING,
            confidence=0.8,
            instruction="edit",
            action_probabilities={item: 0.25 for item in ACTION_ORDER},
            per_action_values={item: 0.0 for item in ACTION_ORDER},
        )


def test_milestone2_executor_registry_has_distinct_capabilities():
    registry = _registry()
    manifest = registry.capability_manifest()
    assert [item["action"] for item in manifest["capabilities"]] == [
        item.value for item in ACTION_ORDER
    ]
    context = registry.context(attempt_index=0, max_attempts=2)
    assert all(context.action_available(item) for item in ACTION_ORDER)


def test_milestone3_campaign_executes_all_actions_and_records_actual_trials(tmp_path):
    source = _Generator().generate(prompt="A ball falls.", physics_plan=PhysicsPlan(), seed=1)
    recorder = JsonlTrialRecorder(tmp_path / "trials.jsonl", fsync=False)
    campaign = ActualTrialCampaign(
        critic=_Critic(),
        executors=_registry(),
        semantic_scorer=lambda before, after, prompt: 0.9,
        quality_scorer=lambda before, after, prompt: 0.85,
        recorder=recorder,
        compatibility={"compatibility_id": _compatibility().compatibility_id},
    )
    result = campaign.run(
        sample_id="sample-1",
        group_id="group-1",
        domain="blender",
        source_evaluation=CandidateEvaluation(source, _report()),
        prompt="A ball falls.",
        physics_plan=PhysicsPlan(),
        base_seed=10,
    )
    assert [trial.decision.action for trial in result.trials] == list(ACTION_ORDER[:3])
    assert all(trial.successful and trial.failure_reason is None for trial in result.trials)
    assert result.target.target_action is RepairAction.PROMPT_REPAIR
    loaded = read_trials(tmp_path / "trials.jsonl")
    assert loaded == result.trials
    metrics = closed_loop_metrics(loaded)
    assert metrics["by_domain"]["blender"]["repair_success_rate"] == 1.0


def test_independent_runner_closes_loop_without_loop_controller(tmp_path):
    compatibility = _compatibility()
    runner = LearningRepairLoopRunner(
        generator=_Generator(),
        critic=_Critic(),
        selector=EvidenceAwareSelector(),
        policy=CategoryOnlyPolicy(compatibility_id=compatibility.compatibility_id),
        executors=_registry(),
        semantic_scorer=lambda before, after, prompt: 0.9,
        quality_scorer=lambda before, after, prompt: 0.85,
        config=RunnerConfig(max_attempts=2, base_seed=1, domain="fake"),
        recorder=JsonlTrialRecorder(tmp_path / "runner.jsonl", fsync=False),
        compatibility_manifest=compatibility,
    )
    result = runner.run(
        prompt="A ball falls.",
        physics_plan=PhysicsPlan(),
        group_id="group-runner",
        run_id="run-1",
    )
    assert result.stop_reason == "accepted"
    assert result.trials[0].decision.action is RepairAction.PROMPT_REPAIR
    assert result.final_report["decision"] == "physical"


def _target(sample_id, group_id, action, *, domain="blender", rewards=None):
    rewards = rewards or {
        RepairAction.PROMPT_REPAIR: 0.5,
        RepairAction.GLOBAL_REGENERATION: 0.4,
        RepairAction.LOCAL_EDITING: None,
        RepairAction.REJECT: 0.0,
    }
    return LearningTargetV1(
        sample_id=sample_id,
        group_id=group_id,
        domain=domain,
        critic_report=_report().to_dict(),
        context=RepairContext(),
        target_action=action,
        action_rewards=rewards,
        available_actions={item: True for item in ACTION_ORDER},
        source_trial_ids=(f"trial-{sample_id}",),
        metadata={"proxy_label": False},
    )


def test_group_split_and_memory_keep_failures_as_negative_utility():
    targets = tuple(
        _target(f"s{index}", f"g{index}", RepairAction.PROMPT_REPAIR)
        for index in range(8)
    )
    split = assign_group_splits(
        targets, validation_fraction=0.25, test_fraction=0.25, seed=42
    )
    assert audit_targets(split)["group_leakage"] == {}
    memory = ActualTrialMemory(split, failed_action_utility=-0.3)
    prediction = MemoryValuePredictor(
        memory, compatibility_id=_compatibility().compatibility_id
    ).predict(_report())
    assert prediction.per_action_values[RepairAction.PROMPT_REPAIR] > 0
    assert prediction.per_action_values[RepairAction.LOCAL_EDITING] < 0


def test_r0_r1_r3_evaluation_is_domain_separated():
    compatibility = _compatibility()
    targets = (
        _target("b", "gb", RepairAction.PROMPT_REPAIR),
        _target("h", "gh", RepairAction.PROMPT_REPAIR, domain="hunyuan"),
    )
    memory = ActualTrialMemory(targets)
    policies = {
        "R0": CategoryOnlyPolicy(compatibility_id=compatibility.compatibility_id),
        "R1": HeuristicDecisionPolicy(compatibility_id=compatibility.compatibility_id),
        "R3": ActionValueDecisionPolicy(
            MemoryValuePredictor(memory, compatibility_id=compatibility.compatibility_id),
            minimum_confidence=0.0,
        ),
    }
    report = compare_policies(policies, targets)
    assert set(report["by_domain"]) == {"blender", "hunyuan"}
    assert report["domain_warning"]


def test_hunyuan_manifest_freezes_splits_and_rejects_placeholders():
    manifest = FrozenCampaignManifest.load(
        "configs/learning_repair/hunyuan_campaign.example.json"
    )
    result = validate_campaign_artifacts(manifest, base_dir=".")
    assert result["valid"] is False
    assert result["split_counts"] == {"calibration": 1, "test": 1}
    assert any("placeholder" in item for item in result["artifact_failures"])


def test_canonical_cloud_campaign_bridge_writes_new_trial_and_target_manifests(tmp_path):
    compatibility = _compatibility()
    manifest = FrozenCampaignManifest(
        campaign_id="blender-smoke-v1",
        domain="blender",
        critic_model_id=compatibility.critic_model_id,
        generator_model_id="fake-generator-v1",
        executor_version="test-executors-v1",
        items=(
            CampaignItem(
                sample_id="cloud-smoke",
                group_id="cloud-group",
                prompt="A ball falls.",
                source_video="fake://source.mp4",
                seed=1,
                split="train",
            ),
        ),
    )
    bundle = CloudBackendBundle(
        critic=_Critic(),
        executors=_registry(),
        source_loader=lambda item: _Generator().generate(
            prompt=item.prompt, physics_plan=PhysicsPlan(), seed=1
        ),
        physics_plan_provider=lambda item: PhysicsPlan(),
        semantic_scorer=lambda before, after, prompt: 0.9,
        quality_scorer=lambda before, after, prompt: 0.9,
    )
    result = run_frozen_campaign(
        manifest,
        bundle,
        compatibility=compatibility,
        trials_output=tmp_path / "trials.jsonl",
        targets_output=tmp_path / "targets.jsonl",
    )
    assert result["sample_count"] == 1
    assert result["trial_count"] == 3
    assert (tmp_path / "trials.jsonl").is_file()
    assert (tmp_path / "targets.jsonl").is_file()


def test_versioned_memory_never_overwrites_an_existing_version(tmp_path):
    source = _Generator().generate(prompt="p", physics_plan=PhysicsPlan(), seed=1)
    campaign = ActualTrialCampaign(
        critic=_Critic(),
        executors=_registry(),
        semantic_scorer=lambda before, after, prompt: 0.9,
        quality_scorer=lambda before, after, prompt: 0.9,
    )
    trials = campaign.run(
        sample_id="memory-sample",
        group_id="memory-group",
        domain="fake",
        source_evaluation=CandidateEvaluation(source, _report()),
        prompt="p",
        physics_plan=PhysicsPlan(),
        base_seed=10,
    ).trials
    writer = VersionedMemoryWriter(tmp_path)
    paths = writer.publish(
        trials,
        version="v1",
        critic_id="critic",
        executor_manifest=_registry().capability_manifest(),
    )
    assert all(path.is_file() for path in paths)
    with pytest.raises(FileExistsError):
        writer.publish(
            trials,
            version="v1",
            critic_id="critic",
            executor_manifest=_registry().capability_manifest(),
        )
