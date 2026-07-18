"""Actual-Trial Memory baselines including explicit failed-action risk."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from physgenloop.learning_repair.contracts import ACTION_ORDER, RepairAction, RepairContext
from physgenloop.learning_repair.features import ReportFeatureEncoder

from .contracts import LearningTargetV1
from .value_policy import ActionValuePrediction


def _cosine(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return numerator / (first_norm * second_norm)


@dataclass(frozen=True)
class TargetMemoryMatch:
    target: LearningTargetV1
    similarity: float


class ActualTrialMemory:
    """Read-only retrieval over actual targets; failed trials lower action utility."""

    def __init__(
        self,
        targets: Iterable[LearningTargetV1],
        *,
        encoder: ReportFeatureEncoder | None = None,
        failed_action_utility: float = -0.25,
    ) -> None:
        self.targets = tuple(targets)
        self.encoder = encoder or ReportFeatureEncoder()
        self.failed_action_utility = float(failed_action_utility)
        self.selection_mode = (
            "classification_proxy"
            if self.targets
            and all(bool(item.metadata.get("proxy_label")) for item in self.targets)
            else "action_value"
        )
        self._vectors = tuple(
            self.encoder.encode(item.critic_report, item.context) for item in self.targets
        )

    def retrieve(
        self,
        report: Any,
        *,
        context: RepairContext | None = None,
        k: int = 5,
        minimum_similarity: float = 0.2,
    ) -> tuple[TargetMemoryMatch, ...]:
        query = self.encoder.encode(report, context)
        matches = [
            TargetMemoryMatch(target, max(0.0, _cosine(query, vector)))
            for target, vector in zip(self.targets, self._vectors)
        ]
        matches = [item for item in matches if item.similarity >= minimum_similarity]
        matches.sort(key=lambda item: (-item.similarity, item.target.sample_id))
        return tuple(matches[:k])

    def action_values(
        self,
        matches: Iterable[TargetMemoryMatch],
    ) -> dict[RepairAction, float]:
        weighted = {action: 0.0 for action in ACTION_ORDER}
        weights = {action: 0.0 for action in ACTION_ORDER}
        for match in matches:
            for action in ACTION_ORDER:
                if not match.target.available_actions[action]:
                    continue
                reward = match.target.action_rewards[action]
                if reward is None and match.target.metadata.get("proxy_label"):
                    # For proxy targets, null means the action was not executed.
                    # It is missing feedback, not a failed trial.
                    continue
                utility = self.failed_action_utility if reward is None else reward
                weighted[action] += match.similarity * utility
                weights[action] += match.similarity
        return {
            action: (weighted[action] / weights[action] if weights[action] else self.failed_action_utility)
            for action in ACTION_ORDER
        }


class MemoryValuePredictor:
    def __init__(
        self,
        memory: ActualTrialMemory,
        *,
        compatibility_id: str,
        k: int = 5,
    ) -> None:
        self.memory = memory
        self.compatibility_id = compatibility_id
        self.model_id = "actual-trial-memory-v1"
        self.selection_mode = memory.selection_mode
        self.k = k

    def predict(
        self,
        critic_report: Any,
        *,
        context: RepairContext | None = None,
    ) -> ActionValuePrediction:
        matches = self.memory.retrieve(critic_report, context=context, k=self.k)
        values = self.memory.action_values(matches)
        maximum = max(values.values())
        exp_values = {action: math.exp(values[action] - maximum) for action in ACTION_ORDER}
        total = sum(exp_values.values())
        return ActionValuePrediction(
            action_probabilities={action: exp_values[action] / total for action in ACTION_ORDER},
            per_action_values=values,
            model_id=self.model_id,
        )


class BlendedValuePredictor:
    """R4/R5 predictor: learned values plus auditable actual-Trial Memory."""

    def __init__(
        self,
        policy: Any,
        memory: ActualTrialMemory,
        *,
        memory_weight: float = 0.25,
        k: int = 5,
        model_id: str = "action-value+memory",
    ) -> None:
        if not 0.0 <= memory_weight <= 1.0:
            raise ValueError("memory_weight must be within [0, 1]")
        self.policy = policy
        self.memory = memory
        self.memory_weight = memory_weight
        self.k = k
        self.model_id = model_id
        self.compatibility_id = str(policy.compatibility_id)
        self.selection_mode = getattr(policy, "selection_mode", "action_value")

    def predict(
        self,
        critic_report: Any,
        *,
        context: RepairContext | None = None,
    ) -> ActionValuePrediction:
        learned = self.policy.predict(critic_report, context=context)
        matches = self.memory.retrieve(critic_report, context=context, k=self.k)
        memory_values = self.memory.action_values(matches)
        weight = self.memory_weight if matches else 0.0
        values = {
            action: (1.0 - weight) * learned.per_action_values[action]
            + weight * memory_values[action]
            for action in ACTION_ORDER
        }
        probabilities = {
            action: (1.0 - weight) * learned.action_probabilities[action]
            + weight / len(ACTION_ORDER)
            for action in ACTION_ORDER
        }
        return ActionValuePrediction(probabilities, values, self.model_id)
