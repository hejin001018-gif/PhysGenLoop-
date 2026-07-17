"""Capability-aware action selection separated from Policy prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import ACTION_ORDER, RepairAction, RepairContext


@dataclass(frozen=True)
class RepairSelection:
    action: RepairAction
    probabilities: dict[RepairAction, float]
    selection_mode: str
    abstained: bool = False
    fallback_reason: str | None = None

    @property
    def confidence(self) -> float:
        return self.probabilities[self.action]


class RepairSelector:
    """Apply capability masks and choose from Policy probability/value outputs."""

    def __init__(
        self,
        *,
        probability_weight: float = 0.15,
        minimum_confidence: float = 0.35,
    ) -> None:
        if probability_weight < 0:
            raise ValueError("probability_weight must be non-negative")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be within [0, 1]")
        self.probability_weight = float(probability_weight)
        self.minimum_confidence = float(minimum_confidence)

    def select(
        self,
        *,
        action_probabilities: Mapping[RepairAction | str, float],
        per_action_values: Mapping[RepairAction | str, float],
        context: RepairContext,
        selection_mode: str,
    ) -> RepairSelection:
        probabilities = {
            RepairAction(action): float(value)
            for action, value in action_probabilities.items()
        }
        values = {
            RepairAction(action): float(value)
            for action, value in per_action_values.items()
        }
        if set(probabilities) != set(ACTION_ORDER):
            raise ValueError("action_probabilities must contain every repair action")
        if set(values) != set(ACTION_ORDER):
            raise ValueError("per_action_values must contain every repair action")
        if any(value < 0.0 for value in probabilities.values()):
            raise ValueError("action probabilities must be non-negative")

        available = {
            action: context.action_available(action) for action in ACTION_ORDER
        }
        candidates = tuple(action for action in ACTION_ORDER if available[action])
        total = sum(probabilities[action] for action in candidates)
        if total <= 0.0:
            masked = {
                action: 1.0 if action is RepairAction.REJECT else 0.0
                for action in ACTION_ORDER
            }
            return RepairSelection(
                action=RepairAction.REJECT,
                probabilities=masked,
                selection_mode=selection_mode,
                abstained=True,
                fallback_reason="no positive probability on an executable action",
            )
        masked = {
            action: probabilities[action] / total if available[action] else 0.0
            for action in ACTION_ORDER
        }

        if selection_mode == "classification_proxy":
            action = max(
                candidates,
                key=lambda item: (masked[item], -ACTION_ORDER.index(item)),
            )
        elif selection_mode == "action_value":
            action = max(
                candidates,
                key=lambda item: (
                    values[item] + self.probability_weight * masked[item],
                    masked[item],
                    -ACTION_ORDER.index(item),
                ),
            )
        else:
            raise ValueError(f"unsupported selection mode: {selection_mode!r}")

        confidence = masked[action]
        return RepairSelection(
            action=action,
            probabilities=masked,
            selection_mode=selection_mode,
            abstained=confidence < self.minimum_confidence,
            fallback_reason=(
                None
                if confidence >= self.minimum_confidence
                else (
                    f"selected confidence {confidence:.4f} is below "
                    f"{self.minimum_confidence:.4f}"
                )
            ),
        )
