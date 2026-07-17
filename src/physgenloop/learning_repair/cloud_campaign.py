"""Backend-factory bridge for running frozen campaigns on an isolated cloud host."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Callable

from physgenloop.contracts import CandidateEvaluation

from .campaign import ActualTrialCampaign, MetricScorer, RewardSpec
from .compatibility import CompatibilityError, CompatibilityManifest
from .executors import ExecutorRegistry
from .manifests import CampaignItem, FrozenCampaignManifest
from .recording import JsonlTrialRecorder, write_targets


@dataclass(frozen=True)
class CloudBackendBundle:
    """All mutable/model-specific cloud integrations stay behind this boundary."""

    critic: Any
    executors: ExecutorRegistry
    source_loader: Callable[[CampaignItem], Any]
    physics_plan_provider: Callable[[CampaignItem], Any]
    semantic_scorer: MetricScorer
    quality_scorer: MetricScorer


def load_backend_factory(
    specification: str,
    *,
    manifest: FrozenCampaignManifest,
    compatibility: CompatibilityManifest,
) -> CloudBackendBundle:
    if ":" not in specification:
        raise ValueError("backend factory must use module.path:function syntax")
    module_name, function_name = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    bundle = factory(manifest=manifest, compatibility=compatibility)
    if not isinstance(bundle, CloudBackendBundle):
        raise TypeError("backend factory must return CloudBackendBundle")
    return bundle


def run_frozen_campaign(
    manifest: FrozenCampaignManifest,
    bundle: CloudBackendBundle,
    *,
    compatibility: CompatibilityManifest,
    trials_output: str | Path,
    targets_output: str | Path,
    reward_spec: RewardSpec | None = None,
) -> dict[str, Any]:
    """Run without overwriting prior outputs; use a new path for every campaign."""

    trials_path = Path(trials_output)
    targets_path = Path(targets_output)
    if trials_path.exists() or targets_path.exists():
        raise FileExistsError("campaign outputs already exist; choose a new versioned path")
    if manifest.critic_model_id != compatibility.critic_model_id:
        raise CompatibilityError(
            "campaign critic_model_id does not match compatibility manifest"
        )
    recorder = JsonlTrialRecorder(trials_path)
    campaign = ActualTrialCampaign(
        critic=bundle.critic,
        executors=bundle.executors,
        semantic_scorer=bundle.semantic_scorer,
        quality_scorer=bundle.quality_scorer,
        reward_spec=reward_spec,
        recorder=recorder,
        compatibility={
            "compatibility_id": compatibility.compatibility_id,
            "critic_model_id": compatibility.critic_model_id,
            "campaign_manifest_sha256": manifest.fingerprint,
            "executor_version": manifest.executor_version,
        },
    )
    targets = []
    trial_count = 0
    for item in manifest.items:
        candidate = bundle.source_loader(item)
        physics_plan = bundle.physics_plan_provider(item)
        report = bundle.critic.evaluate(
            candidate,
            prompt=item.prompt,
            physics_plan=physics_plan,
        )
        compatibility.assert_report(report)
        result = campaign.run(
            sample_id=item.sample_id,
            group_id=item.group_id,
            domain=manifest.domain,
            source_evaluation=CandidateEvaluation(candidate, report),
            prompt=item.prompt,
            physics_plan=physics_plan,
            base_seed=item.seed + 1,
            split=item.split,
            metadata={
                **(item.metadata or {}),
                "campaign_id": manifest.campaign_id,
                "campaign_manifest_sha256": manifest.fingerprint,
                "generator_model_id": manifest.generator_model_id,
            },
        )
        targets.append(result.target)
        trial_count += len(result.trials)
    write_targets(targets, targets_path)
    return {
        "campaign_id": manifest.campaign_id,
        "domain": manifest.domain,
        "sample_count": len(targets),
        "trial_count": trial_count,
        "trials_output": str(trials_path),
        "targets_output": str(targets_path),
        "campaign_manifest_sha256": manifest.fingerprint,
        "compatibility_id": compatibility.compatibility_id,
        "executor_capabilities": bundle.executors.capability_manifest(),
    }
