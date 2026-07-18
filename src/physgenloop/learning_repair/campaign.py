"""Actual multi-action RepairTrial collection for Blender and Hunyuan domains."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping

from physgenloop.contracts import CandidateEvaluation
from physgenloop.learning_repair.contracts import ACTION_ORDER, RepairAction, RepairContext

from .baselines import _target
from .contracts import (
    CandidateRecord,
    ExecutionRequest,
    LearningTargetV1,
    RepairDecision,
    RepairTrialV1,
    ScoreBundle,
)
from .executors import ExecutorRegistry
from .recording import JsonlTrialRecorder


MetricScorer = Callable[[Any, Any, str], float]


def _report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        report = report.to_dict()
    if not isinstance(report, Mapping):
        raise ValueError("Critic report must be a mapping or expose to_dict()")
    return dict(report)


@dataclass(frozen=True)
class RewardSpec:
    """Pre-registered gate and reward; its fingerprint is stored in every Trial."""

    minimum_physics_score: float = 0.80
    minimum_semantic_score: float = 0.85
    minimum_quality_score: float = 0.75
    physics_gain_weight: float = 1.0
    semantic_weight: float = 0.30
    quality_weight: float = 0.20
    cost_weight: float = 0.10
    cost_scale: float = 1.0
    version: str = "repair-reward/1.0"

    def __post_init__(self) -> None:
        for name in (
            "minimum_physics_score",
            "minimum_semantic_score",
            "minimum_quality_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.cost_scale <= 0:
            raise ValueError("cost_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "minimum_physics_score": self.minimum_physics_score,
            "minimum_semantic_score": self.minimum_semantic_score,
            "minimum_quality_score": self.minimum_quality_score,
            "physics_gain_weight": self.physics_gain_weight,
            "semantic_weight": self.semantic_weight,
            "quality_weight": self.quality_weight,
            "cost_weight": self.cost_weight,
            "cost_scale": self.cost_scale,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def valid(self, scores: ScoreBundle | None) -> bool:
        return bool(
            scores is not None
            and scores.semantic is not None
            and scores.quality is not None
            and scores.physics >= self.minimum_physics_score
            and scores.semantic >= self.minimum_semantic_score
            and scores.quality >= self.minimum_quality_score
        )

    def reward(
        self,
        *,
        before: ScoreBundle,
        after: ScoreBundle,
        cost: float,
    ) -> float:
        if not self.valid(after):
            raise ValueError("reward cannot be computed before quality gates pass")
        assert after.semantic is not None and after.quality is not None
        return (
            self.physics_gain_weight * (after.physics - before.physics)
            + self.semantic_weight * after.semantic
            + self.quality_weight * after.quality
            - self.cost_weight * min(1.0, cost / self.cost_scale)
        )


@dataclass(frozen=True)
class CampaignResult:
    trials: tuple[RepairTrialV1, ...]
    target: LearningTargetV1


class ActualTrialCampaign:
    """Execute every available repair action against the same broken candidate."""

    def __init__(
        self,
        *,
        critic: Any,
        executors: ExecutorRegistry,
        semantic_scorer: MetricScorer,
        quality_scorer: MetricScorer,
        reward_spec: RewardSpec | None = None,
        recorder: JsonlTrialRecorder | None = None,
        compatibility: Mapping[str, str] | None = None,
    ) -> None:
        self.critic = critic
        self.executors = executors
        self.semantic_scorer = semantic_scorer
        self.quality_scorer = quality_scorer
        self.reward_spec = reward_spec or RewardSpec()
        self.recorder = recorder
        self.compatibility = dict(compatibility or {})

    def _decision(
        self,
        action: RepairAction,
        *,
        report: Any,
        candidate: Any,
        prompt: str,
    ) -> RepairDecision:
        probabilities = {
            item: (1.0 if item is action else 0.0) for item in ACTION_ORDER
        }
        return RepairDecision(
            action=action,
            confidence=1.0,
            instruction=f"Execute registered {action.value} trial.",
            action_probabilities=probabilities,
            per_action_values={item: 0.0 for item in ACTION_ORDER},
            parameters={"campaign_forced_action": True, "original_prompt": prompt},
            local_target=_target(report, candidate) if action is RepairAction.LOCAL_EDITING else None,
            source="actual-trial-campaign-v1",
            compatibility_id=self.compatibility.get("compatibility_id", "unknown"),
        )

    def run(
        self,
        *,
        sample_id: str,
        group_id: str,
        domain: str,
        source_evaluation: CandidateEvaluation,
        prompt: str,
        physics_plan: Any,
        base_seed: int,
        split: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CampaignResult:
        report_before = source_evaluation.report
        report_before_dict = _report_dict(report_before)
        before = ScoreBundle(physics=float(report_before_dict["physics_score"]))
        context = self.executors.context(attempt_index=0, max_attempts=1)
        trials: list[RepairTrialV1] = []
        rewards = {action: None for action in ACTION_ORDER}
        availability = {
            action: self.executors.supports(action) for action in ACTION_ORDER
        }
        availability[RepairAction.REJECT] = True

        executable_actions = tuple(
            action
            for action in ACTION_ORDER
            if action is not RepairAction.REJECT and self.executors.supports(action)
        )
        for offset, action in enumerate(executable_actions):
            decision = self._decision(
                action,
                report=report_before,
                candidate=source_evaluation.candidate,
                prompt=prompt,
            )
            execution = self.executors.execute(
                ExecutionRequest(
                    decision=decision,
                    candidate=source_evaluation.candidate,
                    critic_report=report_before,
                    prompt=prompt,
                    physics_plan=physics_plan,
                    seed=base_seed + offset,
                    history=(source_evaluation,),
                    metadata={"sample_id": sample_id, "group_id": group_id, "domain": domain},
                )
            )
            after_report = None
            after_scores = None
            successful = False
            failure_reason = execution.failure_reason
            if execution.status == "succeeded" and execution.candidate is not None:
                next_prompt = execution.next_prompt or prompt
                after_report = self.critic.evaluate(
                    execution.candidate,
                    prompt=next_prompt,
                    physics_plan=physics_plan,
                )
                after_dict = _report_dict(after_report)
                after_scores = ScoreBundle(
                    physics=float(after_dict["physics_score"]),
                    semantic=float(
                        self.semantic_scorer(
                            source_evaluation.candidate, execution.candidate, prompt
                        )
                    ),
                    quality=float(
                        self.quality_scorer(
                            source_evaluation.candidate, execution.candidate, prompt
                        )
                    ),
                )
                successful = self.reward_spec.valid(after_scores)
                failure_reason = None if successful else "quality_gate_failed"
                if successful:
                    rewards[action] = self.reward_spec.reward(
                        before=before,
                        after=after_scores,
                        cost=execution.cost,
                    )
            trial_key = (
                f"{sample_id}\0{group_id}\0{domain}\0{source_evaluation.candidate.candidate_id}"
                f"\0{action.value}\0{base_seed + offset}\0{self.reward_spec.fingerprint}"
            )
            trial = RepairTrialV1(
                trial_id=f"trial-{hashlib.sha256(trial_key.encode('utf-8')).hexdigest()[:24]}",
                group_id=group_id,
                domain=domain,
                source_candidate=CandidateRecord.from_candidate(source_evaluation.candidate),
                prompt=prompt,
                critic_before=report_before_dict,
                decision=decision,
                execution=execution.to_dict(),
                critic_after=None if after_report is None else _report_dict(after_report),
                before_scores=before,
                after_scores=after_scores,
                successful=successful,
                failure_reason=(
                    None
                    if successful
                    else failure_reason or "executor_returned_no_candidate"
                ),
                compatibility=self.compatibility,
                metadata={
                    **dict(metadata or {}),
                    "sample_id": sample_id,
                    "reward_spec": self.reward_spec.to_dict(),
                    "reward_spec_sha256": self.reward_spec.fingerprint,
                },
            )
            trials.append(trial)
            if self.recorder is not None:
                self.recorder.append(trial)

        valid_rewards = {
            action: reward
            for action, reward in rewards.items()
            if action is not RepairAction.REJECT and reward is not None
        }
        if valid_rewards:
            target_action = max(
                ACTION_ORDER,
                key=lambda action: (
                    float("-inf") if action not in valid_rewards else valid_rewards[action],
                    -ACTION_ORDER.index(action),
                ),
            )
            rewards[RepairAction.REJECT] = 0.0
        else:
            target_action = RepairAction.REJECT
            rewards[RepairAction.REJECT] = 0.0
        target = LearningTargetV1(
            sample_id=sample_id,
            group_id=group_id,
            domain=domain,
            critic_report=report_before_dict,
            context=context,
            target_action=target_action,
            action_rewards=rewards,
            available_actions=availability,
            source_trial_ids=tuple(item.trial_id for item in trials),
            split=split,
            metadata={
                **dict(metadata or {}),
                "reward_spec_sha256": self.reward_spec.fingerprint,
                "proxy_label": False,
            },
        )
        return CampaignResult(tuple(trials), target)


def targets_from_trials(
    trials: Iterable[RepairTrialV1],
    *,
    reward_spec: RewardSpec | None = None,
) -> tuple[LearningTargetV1, ...]:
    """Rebuild targets from immutable Trials using the same pre-registered reward."""

    reward_spec = reward_spec or RewardSpec()
    grouped: dict[tuple[str, str, str], list[RepairTrialV1]] = {}
    for trial in trials:
        key = (trial.domain, trial.group_id, trial.source_candidate.candidate_id)
        grouped.setdefault(key, []).append(trial)
    targets = []
    for (_domain, _group, _candidate), records in sorted(grouped.items()):
        first = records[0]
        rewards = {action: None for action in ACTION_ORDER}
        available = {action: False for action in ACTION_ORDER}
        available[RepairAction.REJECT] = True
        for trial in records:
            action = trial.decision.action
            available[action] = True
            if trial.after_scores is not None and reward_spec.valid(trial.after_scores):
                rewards[action] = reward_spec.reward(
                    before=trial.before_scores,
                    after=trial.after_scores,
                    cost=float(trial.execution.get("cost", 0.0)),
                )
        valid = {action: value for action, value in rewards.items() if value is not None}
        target_action = (
            max(valid, key=lambda action: (valid[action], -ACTION_ORDER.index(action)))
            if valid
            else RepairAction.REJECT
        )
        rewards[RepairAction.REJECT] = 0.0
        targets.append(
            LearningTargetV1(
                sample_id=str(first.metadata.get("sample_id", first.trial_id)),
                group_id=first.group_id,
                domain=first.domain,
                critic_report=first.critic_before,
                context=RepairContext(),
                target_action=target_action,
                action_rewards=rewards,
                available_actions=available,
                source_trial_ids=tuple(item.trial_id for item in records),
                metadata={
                    "reward_spec_sha256": reward_spec.fingerprint,
                    "proxy_label": False,
                },
            )
        )
    return tuple(targets)
