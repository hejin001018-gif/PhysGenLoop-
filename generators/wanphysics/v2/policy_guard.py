"""Three-action capability validator; it never rewrites Policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physgenloop.learning_repair.base_contracts import RepairAction
from generators.wanphysics.error_scope import extract_critical_frames

POLICY_GUARD_SCHEMA_VERSION = "policy-guard/2.0"
_VALID_ACTIONS = {action.value for action in RepairAction}


def normalize_action(action: Any) -> str:
    if isinstance(action, RepairAction):
        return action.value
    text = str(action).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.lower()
    for candidate in RepairAction:
        if text in {candidate.value, candidate.name.lower()}:
            return candidate.value
    return text


def classify_scope(report: Any, total_frames: int, local_threshold: float) -> str:
    violations = tuple(getattr(report, "violations", ()) or ())
    if not violations:
        return "unknown"
    critical = extract_critical_frames(report)
    if total_frames <= 0:
        return "broad"
    ratio = len(critical) / float(total_frames)
    return "local" if 0.0 < ratio < float(local_threshold) else "broad"


def _has_instruction(report: Any) -> bool:
    return any(
        str(getattr(item, "repair_instruction", "") or "").strip()
        for item in tuple(getattr(report, "violations", ()) or ())
    )


@dataclass(frozen=True)
class GuardResult:
    policy_action: str
    status: str
    scope: str
    blocked_reason: str | None
    capability_available: dict[str, bool]
    total_frames: int
    threshold: float
    mask_valid: bool
    schema_version: str = POLICY_GUARD_SCHEMA_VERSION

    @property
    def allowed(self) -> bool:
        return self.status == "allowed"

    @property
    def final_action(self) -> str:
        return self.policy_action if self.allowed else RepairAction.REJECT.value

    @property
    def override_reason(self) -> str | None:
        return self.blocked_reason

    @property
    def overridden(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_action": self.policy_action,
            "status": self.status,
            "scope": self.scope,
            "blocked_reason": self.blocked_reason,
            "executed_action": self.final_action,
            "capability_available": dict(self.capability_available),
            "total_frames": self.total_frames,
            "threshold": self.threshold,
            "mask_valid": self.mask_valid,
        }


def resolve_action(
    *,
    policy_action: Any,
    report: Any,
    total_frames: int,
    local_threshold: float,
    capability_available: dict[str, bool],
    mask_valid: bool,
) -> GuardResult:
    """Validate the selected action; blocked decisions terminate through audited reject."""

    action = normalize_action(policy_action)
    scope = classify_scope(report, total_frames, local_threshold)
    reason: str | None = None
    if action not in _VALID_ACTIONS:
        reason = "action_not_in_three_action_contract"
    elif not capability_available.get(action, action == RepairAction.REJECT.value):
        reason = f"capability_unavailable:{action}"
    elif action == RepairAction.LOCAL_EDITING.value:
        if scope != "local":
            reason = f"local_editing_requires_local_scope:{scope}"
        elif not mask_valid:
            reason = "local_editing_requires_valid_mask_manifest"
    elif action == RepairAction.PROMPT_REPAIR.value and not _has_instruction(report):
        reason = "prompt_repair_requires_reliable_instruction"

    return GuardResult(
        policy_action=action,
        status="allowed" if reason is None else "blocked",
        scope=scope,
        blocked_reason=reason,
        capability_available=dict(capability_available),
        total_frames=int(total_frames),
        threshold=float(local_threshold),
        mask_valid=bool(mask_valid),
    )
