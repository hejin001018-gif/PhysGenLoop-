"""Category-only and evidence-aware baselines for Milestone 1."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from physgenloop.learning_repair.contracts import ACTION_ORDER, RepairAction, RepairContext
from physgenloop.learning_repair.features import ReportFeatureEncoder

from .contracts import LocalEditTarget, RepairDecision


class DecisionPolicy(Protocol):
    def decide(
        self,
        *,
        critic_report: Any,
        candidate: Any,
        prompt: str,
        context: RepairContext,
    ) -> RepairDecision: ...


_CATEGORY_ACTION = {
    "gravity_violation": RepairAction.PROMPT_REPAIR,
    "friction_violation": RepairAction.PROMPT_REPAIR,
    "contact_violation": RepairAction.PROMPT_REPAIR,
    "collision_violation": RepairAction.LOCAL_EDITING,
    "trajectory_violation": RepairAction.LOCAL_EDITING,
    "continuity_violation": RepairAction.LOCAL_EDITING,
    "appearance_violation": RepairAction.LOCAL_EDITING,
    "unknown_violation": RepairAction.GLOBAL_REGENERATION,
}


def _raw(report: Any) -> Mapping[str, Any]:
    if hasattr(report, "to_dict"):
        report = report.to_dict()
    return report if isinstance(report, Mapping) else {}


def _available(context: RepairContext) -> dict[RepairAction, bool]:
    return {action: context.action_available(action) for action in ACTION_ORDER}


def _masked_choice(
    preferred: RepairAction,
    context: RepairContext,
) -> tuple[RepairAction, bool, str | None]:
    available = _available(context)
    if available[preferred]:
        return preferred, False, None
    for fallback in (
        RepairAction.GLOBAL_REGENERATION,
        RepairAction.PROMPT_REPAIR,
        RepairAction.LOCAL_EDITING,
        RepairAction.REJECT,
    ):
        if available[fallback]:
            return fallback, True, f"preferred action {preferred.value} is unavailable"
    return RepairAction.REJECT, True, "no executable repair backend"


def _distribution(action: RepairAction, context: RepairContext) -> dict[RepairAction, float]:
    available = _available(context)
    weights = {item: (0.05 if available[item] else 0.0) for item in ACTION_ORDER}
    weights[action] = 0.85
    total = sum(weights.values())
    return {item: weights[item] / total for item in ACTION_ORDER}


def _target(report: Any, candidate: Any) -> LocalEditTarget:
    raw = _raw(report)
    violations = [item for item in raw.get("violations", ()) if isinstance(item, Mapping)]
    objects: list[str] = []
    critical: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    mask_uri = None
    for violation in violations:
        obj = str(violation.get("object", "")).strip()
        if obj and obj not in objects:
            objects.append(obj)
        critical.extend(int(item) for item in violation.get("critical_frames", ()) if int(item) >= 0)
        for name, destination in (("start_frame", starts), ("end_frame", ends)):
            if violation.get(name) is not None:
                destination.append(int(violation[name]))
        evidence = violation.get("evidence", {})
        if mask_uri is None and isinstance(evidence, Mapping):
            raw_mask = evidence.get("mask_uri") or evidence.get("mask_path")
            if raw_mask:
                mask_uri = str(raw_mask)
    return LocalEditTarget(
        parent_candidate_id=str(candidate.candidate_id),
        objects=tuple(objects),
        start_frame=min(starts) if starts and ends else None,
        end_frame=max(ends) if starts and ends else None,
        critical_frames=tuple(sorted(set(critical))),
        mask_uri=mask_uri,
    )


class CategoryOnlyPolicy:
    """R0 baseline: explicit category-to-action mapping with capability masking."""

    def __init__(
        self,
        *,
        encoder: ReportFeatureEncoder | None = None,
        compatibility_id: str = "unknown",
    ) -> None:
        self.encoder = encoder or ReportFeatureEncoder()
        self.compatibility_id = compatibility_id

    def decide(
        self,
        *,
        critic_report: Any,
        candidate: Any,
        prompt: str,
        context: RepairContext,
    ) -> RepairDecision:
        category = self.encoder.primary_category(critic_report)
        preferred = _CATEGORY_ACTION[category]
        action, abstained, reason = _masked_choice(preferred, context)
        probabilities = _distribution(action, context)
        values = {item: probabilities[item] for item in ACTION_ORDER}
        return RepairDecision(
            action=action,
            confidence=probabilities[action],
            instruction=f"Apply the registered repair for {category}.",
            action_probabilities=probabilities,
            per_action_values=values,
            parameters={"primary_category": category, "original_prompt": prompt},
            local_target=_target(critic_report, candidate) if action is RepairAction.LOCAL_EDITING else None,
            source="category-only-v1",
            abstained=abstained,
            fallback_reason=reason,
            compatibility_id=self.compatibility_id,
        )


class HeuristicDecisionPolicy(CategoryOnlyPolicy):
    """R1 baseline with explicit abstention for weak or failed Critic evidence."""

    def __init__(
        self,
        *,
        minimum_coverage: float = 0.25,
        encoder: ReportFeatureEncoder | None = None,
        compatibility_id: str = "unknown",
    ) -> None:
        super().__init__(encoder=encoder, compatibility_id=compatibility_id)
        self.minimum_coverage = float(minimum_coverage)

    def decide(
        self,
        *,
        critic_report: Any,
        candidate: Any,
        prompt: str,
        context: RepairContext,
    ) -> RepairDecision:
        raw = _raw(critic_report)
        decision = str(raw.get("decision", "unknown"))
        coverage = float(raw.get("coverage", 0.0))
        provider_failure = any(
            isinstance(item, Mapping) and str(item.get("status", "")) == "failed"
            for item in raw.get("evidence_bundles", ())
        )
        if decision == "unknown" or coverage < self.minimum_coverage or provider_failure:
            action, _masked, mask_reason = _masked_choice(
                RepairAction.GLOBAL_REGENERATION, context
            )
            reason = "critic evidence is unknown, low-coverage, or provider-failed"
            if mask_reason:
                reason = f"{reason}; {mask_reason}"
            probabilities = _distribution(action, context)
            return RepairDecision(
                action=action,
                confidence=probabilities[action],
                instruction="Regenerate and obtain stronger independent Critic evidence.",
                action_probabilities=probabilities,
                per_action_values={item: probabilities[item] for item in ACTION_ORDER},
                parameters={"coverage": coverage, "provider_failure": provider_failure},
                local_target=_target(critic_report, candidate) if action is RepairAction.LOCAL_EDITING else None,
                source="heuristic-v2",
                abstained=True,
                fallback_reason=reason,
                compatibility_id=self.compatibility_id,
            )
        baseline = super().decide(
            critic_report=critic_report,
            candidate=candidate,
            prompt=prompt,
            context=context,
        )
        payload = baseline.to_dict()
        payload["source"] = "heuristic-v2"
        return RepairDecision.from_dict(payload)


def adapt_legacy_decision(
    decision: Any,
    *,
    critic_report: Any,
    candidate: Any,
    compatibility_id: str,
) -> RepairDecision:
    """Convert an existing LearningRepairAgent decision without changing it."""

    action = RepairAction(decision.action)
    raw_probabilities = getattr(decision, "action_probabilities", {})
    probabilities = {
        item: float(raw_probabilities.get(item.value, raw_probabilities.get(item, 0.0)))
        for item in ACTION_ORDER
    }
    if sum(probabilities.values()) <= 0:
        probabilities = {item: (1.0 if item is action else 0.0) for item in ACTION_ORDER}
    expected = float(getattr(decision, "expected_gain", 0.0))
    values = {
        item: expected * probabilities[item]
        for item in ACTION_ORDER
    }
    return RepairDecision(
        action=action,
        confidence=float(decision.confidence),
        instruction=str(decision.instruction),
        action_probabilities=probabilities,
        per_action_values=values,
        parameters=dict(getattr(decision, "parameters", {})),
        local_target=_target(critic_report, candidate) if action is RepairAction.LOCAL_EDITING else None,
        source=f"legacy-adapter:{getattr(decision, 'source', 'unknown')}",
        compatibility_id=compatibility_id,
    )
