"""V2 action-value repair policy adapter.

This module is intentionally small: it only loads the proxy research
checkpoint and exposes ``repair_with_decision`` for the V2 runner. Historical
training, memory, campaign, and release code are not part of the runtime path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pavg_critic.schemas import CriticReport
from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair import (
    ActionValueDecisionPolicy,
    CompatibilityManifest,
    RepairAction,
    RepairContext,
    TorchActionValuePolicy,
)
from physgenloop.learning_repair.contracts import LocalEditTarget, RepairDecision

_ACTION_PREFIX = {
    RepairAction.PROMPT_REPAIR: "Physics correction",
    RepairAction.LOCAL_EDITING: "Local-edit fallback constraint",
    RepairAction.REJECT: "Replacement constraint",
}


def _report_dict(report: Any) -> Mapping[str, Any]:
    if hasattr(report, "to_dict"):
        report = report.to_dict()
    return report if isinstance(report, Mapping) else {}


def _target(report: Any, candidate: GeneratedCandidate) -> LocalEditTarget:
    raw = _report_dict(report)
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
        for frame in violation.get("critical_frames", ()) or ():
            try:
                idx = int(frame)
            except (TypeError, ValueError):
                continue
            if idx >= 0:
                critical.append(idx)
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


class ActionValueRepairer:
    def __init__(
        self,
        decision_policy: ActionValueDecisionPolicy,
        max_attempts: int = 2,
        local_editor_available: bool = False,
    ) -> None:
        self._policy = decision_policy
        self._max_attempts = max_attempts
        self._attempt_index = 0
        self._previous_actions: list[RepairAction] = []
        self._local_editor_available = local_editor_available

    def repair_with_decision(self, *, prompt: str, report: CriticReport):
        context = RepairContext(
            attempt_index=self._attempt_index,
            max_attempts=self._max_attempts,
            local_editor_available=self._local_editor_available,
            previous_actions=tuple(self._previous_actions),
        )
        placeholder = GeneratedCandidate(
            candidate_id="repair-placeholder",
            video_path="pending://",
            prompt=prompt,
            seed=self._attempt_index,
        )
        decision = self._policy.decide(
            critic_report=report,
            candidate=placeholder,
            prompt=prompt,
            context=context,
        )

        if decision.local_target is None:
            candidate_target = _target(report, placeholder)
            if candidate_target.mask_uri or candidate_target.critical_frames:
                decision = RepairDecision(
                    action=decision.action,
                    confidence=decision.confidence,
                    instruction=decision.instruction,
                    action_probabilities=decision.action_probabilities,
                    per_action_values=decision.per_action_values,
                    parameters=decision.parameters,
                    local_target=candidate_target,
                    source=decision.source,
                    abstained=decision.abstained,
                    fallback_reason=decision.fallback_reason,
                    compatibility_id=decision.compatibility_id,
                )

        self._previous_actions.append(decision.action)
        self._attempt_index += 1

        instruction = decision.instruction.strip()
        if not instruction:
            return prompt, decision
        prefix = _ACTION_PREFIX[decision.action]
        return f"{prompt}\n{prefix}: {instruction}", decision

    def repair(self, *, prompt: str, report: CriticReport) -> str:
        repaired, _ = self.repair_with_decision(prompt=prompt, report=report)
        return repaired


def load_action_value_repairer(
    ckpt_root: str,
    max_attempts: int = 2,
    local_editor_available: bool = False,
) -> ActionValueRepairer:
    ckpt_path = Path(ckpt_root)
    compatibility = CompatibilityManifest.load(
        str(ckpt_path / "config/critic_compatibility_v1.json")
    )

    if not compatibility.deployment_ready:
        print(
            "[repairer] WARNING: checkpoint compatibility manifest has deployment_ready=False; "
            "using proxy research policy.",
        )

    learned_policy = TorchActionValuePolicy.load(
        str(ckpt_path / "model/best_action_value_policy.pt"),
        device="cpu",
        compatibility_manifest=compatibility,
    )
    decision_policy = ActionValueDecisionPolicy(learned_policy, minimum_confidence=0.35)
    return ActionValueRepairer(
        decision_policy,
        max_attempts=max_attempts,
        local_editor_available=local_editor_available,
    )
