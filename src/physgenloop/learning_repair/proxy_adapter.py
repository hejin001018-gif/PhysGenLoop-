"""Explicit adapter from legacy Blender proxy labels to V1 value targets.

The adapter never manufactures RepairTrialV1 records or unobserved action rewards.
Only the selected proxy action receives an observed utility; every alternative
remains ``None`` until a real Executor trial supplies evidence.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable

from physgenloop.learning_repair import (
    ACTION_ORDER,
    RepairAction,
    RepairExample,
    load_repair_manifest,
)

from .campaign import RewardSpec
from .contracts import LearningTargetV1, ScoreBundle


def _selected_proxy_reward(example: RepairExample, reward_spec: RewardSpec) -> float:
    if example.target_action is RepairAction.REJECT:
        return 0.0
    if not example.successful:
        raise ValueError(
            f"proxy target {example.sample_id!r} selects a failed repair action"
        )
    after = ScoreBundle(
        physics=example.after_score,
        semantic=example.semantic_score,
        quality=example.quality_score,
    )
    if not reward_spec.valid(after):
        raise ValueError(
            f"proxy target {example.sample_id!r} does not pass the frozen reward gates"
        )
    return reward_spec.reward(
        before=ScoreBundle(physics=example.before_score),
        after=after,
        cost=float(example.repair_cost or 0.0),
    )


def adapt_proxy_example(
    example: RepairExample,
    *,
    reward_spec: RewardSpec | None = None,
) -> LearningTargetV1:
    """Convert one proxy label while preserving partial-feedback provenance."""

    reward_spec = reward_spec or RewardSpec()
    available = {
        action: example.context.action_available(action) for action in ACTION_ORDER
    }
    available[RepairAction.REJECT] = True
    rewards = {action: None for action in ACTION_ORDER}
    rewards[example.target_action] = _selected_proxy_reward(example, reward_spec)
    metadata = dict(example.metadata)
    metadata.update(
        {
            "proxy_label": True,
            "proxy_source_schema": example.schema_version,
            "proxy_reward_observation": "selected_action_only",
            "proxy_reward_fingerprint": reward_spec.fingerprint,
            "prompt": example.prompt,
            "legacy_outcome": {
                "before_score": example.before_score,
                "after_score": example.after_score,
                "semantic_score": example.semantic_score,
                "quality_score": example.quality_score,
                "repair_cost": example.repair_cost,
            },
        }
    )
    return LearningTargetV1(
        sample_id=example.sample_id,
        group_id=example.group_id,
        domain="blender",
        critic_report=example.critic_report,
        context=example.context,
        target_action=example.target_action,
        action_rewards=rewards,
        available_actions=available,
        source_trial_ids=(),
        split=None,
        metadata=metadata,
    )


def adapt_proxy_manifests(
    manifests: Iterable[str | Path],
    *,
    reward_spec: RewardSpec | None = None,
) -> tuple[tuple[LearningTargetV1, ...], dict[str, object]]:
    """Merge legacy manifests and return V1 targets plus an audit report."""

    reward_spec = reward_spec or RewardSpec()
    targets = []
    sample_ids: set[str] = set()
    sources = []
    source_hashes = {}
    for raw_path in manifests:
        path = Path(raw_path).resolve()
        examples = load_repair_manifest(path)
        sources.append(str(path))
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for example in examples:
            if example.sample_id in sample_ids:
                raise ValueError(f"duplicate proxy sample_id: {example.sample_id}")
            sample_ids.add(example.sample_id)
            targets.append(adapt_proxy_example(example, reward_spec=reward_spec))
    report = {
        "schema_version": "proxy-target-adaptation/1.0",
        "label_type": "blender_proxy_selected_action_only",
        "sample_count": len(targets),
        "group_count": len({item.group_id for item in targets}),
        "action_counts": dict(Counter(item.target_action.value for item in targets)),
        "source_manifests": sources,
        "source_sha256": source_hashes,
        "reward_spec": reward_spec.to_dict(),
        "reward_fingerprint": reward_spec.fingerprint,
        "actual_trial_count": 0,
        "limitations": [
            "Targets originate from Blender proxy labels, not executed RepairTrialV1 records.",
            "Only the selected action reward is observed; alternative action rewards remain null.",
            "Metrics measure Blender proxy generalization and are not Hunyuan repair-success claims.",
        ],
    }
    return tuple(targets), report


def write_adaptation_report(report: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
