"""将多个实际修复 trial 统一转换为监督 target action。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .contracts import RepairAction, RepairContext, RepairExample


@dataclass(frozen=True)
class RewardConfig:
    minimum_physics_score: float = 0.8
    minimum_semantic_score: float = 0.85
    minimum_quality_score: float = 0.75
    physics_gain_weight: float = 1.0
    semantic_weight: float = 0.3
    quality_weight: float = 0.2
    cost_weight: float = 0.1
    cost_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_physics_score",
            "minimum_semantic_score",
            "minimum_quality_score",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.cost_scale <= 0:
            raise ValueError("cost_scale must be positive")


@dataclass(frozen=True)
class RepairTrial:
    action: RepairAction
    strategy: str
    after_score: float
    semantic_score: float
    quality_score: float
    cost: float
    artifacts: dict[str, str]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, RepairAction):
            object.__setattr__(self, "action", RepairAction(self.action))
        if self.action is RepairAction.REJECT:
            raise ValueError("reject is derived when no repair trial passes the gates")
        for name in ("after_score", "semantic_score", "quality_score"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class TrialSelection:
    selected: RepairTrial | None
    reward: float
    evaluated_rewards: dict[str, float | None]


def select_best_trial(
    *,
    before_score: float,
    trials: Iterable[RepairTrial],
    config: RewardConfig | None = None,
) -> TrialSelection:
    """先执行质量门控，再按固定 reward 选择成功 trial。"""

    config = config or RewardConfig()
    if not 0.0 <= before_score <= 1.0:
        raise ValueError("before_score must be within [0, 1]")
    candidates = []
    rewards: dict[str, float | None] = {}
    for index, trial in enumerate(trials):
        key = f"{index}:{trial.action.value}:{trial.strategy}"
        valid = (
            trial.after_score >= config.minimum_physics_score
            and trial.semantic_score >= config.minimum_semantic_score
            and trial.quality_score >= config.minimum_quality_score
        )
        if not valid:
            rewards[key] = None
            continue
        reward = (
            config.physics_gain_weight * (trial.after_score - before_score)
            + config.semantic_weight * trial.semantic_score
            + config.quality_weight * trial.quality_score
            - config.cost_weight * min(1.0, trial.cost / config.cost_scale)
        )
        rewards[key] = reward
        candidates.append((reward, -index, trial))
    if not candidates:
        return TrialSelection(None, 0.0, rewards)
    reward, _stable_index, selected = max(candidates, key=lambda item: (item[0], item[1]))
    return TrialSelection(selected, reward, rewards)


def build_labeled_example(
    *,
    sample_id: str,
    group_id: str,
    prompt: str,
    critic_report: Any,
    before_score: float,
    trials: Iterable[RepairTrial],
    context: RepairContext | None = None,
    base_artifacts: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    reward_config: RewardConfig | None = None,
) -> RepairExample:
    """构造最终监督样本；无 trial 过门时明确标为 reject。"""

    selection = select_best_trial(
        before_score=before_score,
        trials=trials,
        config=reward_config,
    )
    selected = selection.selected
    artifacts = dict(base_artifacts or {})
    if selected is not None:
        artifacts.update(selected.artifacts)
    label_metadata = dict(metadata or {})
    label_metadata["labeling"] = {
        "reward": selection.reward,
        "evaluated_rewards": selection.evaluated_rewards,
        "reward_config": vars(reward_config or RewardConfig()),
    }
    return RepairExample(
        sample_id=sample_id,
        group_id=group_id,
        prompt=prompt,
        critic_report=critic_report,
        target_action=(selected.action if selected else RepairAction.REJECT),
        strategy=(selected.strategy if selected else "reject_after_all_trials_failed_gates"),
        before_score=before_score,
        after_score=(selected.after_score if selected else before_score),
        successful=True,
        semantic_score=(selected.semantic_score if selected else None),
        quality_score=(selected.quality_score if selected else None),
        repair_cost=(selected.cost if selected else None),
        context=context or RepairContext(),
        artifacts=artifacts,
        metadata=label_metadata,
    )
