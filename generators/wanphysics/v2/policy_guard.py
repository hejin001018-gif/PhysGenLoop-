"""Scope 判定 + capability mask + action override（V2）。

修复 P0-4：把 "Policy 决策" 与 "最终执行动作" 解耦，用可解释的 scope 启发式 +
capability mask 对 Policy 输出做兜底/覆盖，并完整记录 override provenance。

与现有 ``generators/wanphysics/error_scope.py`` 的关系：
  - 复用其 ``extract_critical_frames`` / ``classify_error_scope`` 的核心比例逻辑；
  - 额外加入修复方案 §15 的两条规则：无 violations → ``unknown``；mask manifest
    无效 → 强制 ``global``（避免在没有有效 mask 时误走局部修复）。

动作名称统一为小写字符串 ``prompt_repair/global_regeneration/local_editing/reject``，
禁止把 ``RepairAction.PROMPT_REPAIR`` 这类 repr 写进审计产物（方案 §14）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from physgenloop.learning_repair.base_contracts import RepairAction
from generators.wanphysics.error_scope import extract_critical_frames

POLICY_GUARD_SCHEMA_VERSION = "policy-guard/1.0"

_VALID_ACTIONS = {a.value for a in RepairAction}


def normalize_action(action: Any) -> str:
    """把任意 action 表示规范化为小写字符串动作名。

    接受 RepairAction 枚举、``RepairAction.PROMPT_REPAIR`` repr、大小写混合字符串。
    """

    if isinstance(action, RepairAction):
        return action.value
    text = str(action).strip()
    # 处理 "RepairAction.PROMPT_REPAIR" / "repairaction.prompt_repair"
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.lower()
    if text in _VALID_ACTIONS:
        return text
    # 处理枚举名 "PROMPT_REPAIR"
    for a in RepairAction:
        if text == a.name.lower():
            return a.value
    return text


def classify_scope(report: Any, total_frames: int, local_threshold: float, *, mask_valid: bool) -> str:
    """修复方案 §15 的 scope 判定。返回 unknown|global|local。"""

    violations = tuple(getattr(report, "violations", ()) or ())
    if not violations:
        return "unknown"
    critical = extract_critical_frames(report)
    if total_frames <= 0:
        return "global"
    ratio = len(critical) / float(total_frames)
    if ratio == 0.0 or ratio >= float(local_threshold):
        return "global"
    if not mask_valid:
        return "global"
    return "local"


@dataclass(frozen=True)
class GuardResult:
    """一次动作裁决的完整可审计结果。"""

    policy_action: str
    final_action: str
    scope: str
    override_reason: str | None
    capability_available: dict[str, bool]
    total_frames: int
    threshold: float
    mask_valid: bool
    schema_version: str = POLICY_GUARD_SCHEMA_VERSION

    @property
    def overridden(self) -> bool:
        return self.policy_action != self.final_action

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_action": self.policy_action,
            "final_action": self.final_action,
            "scope": self.scope,
            "override_reason": self.override_reason,
            "overridden": self.overridden,
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
    """结合 scope + capability mask 决定最终动作（方案 §15 动作覆盖规则）。

    规则：
      - scope==local 且 local_editing 可用 且 mask 有效 → local_editing；
      - scope==local 但 local_editing 不可用 → mask 掉，回退 prompt_repair/global；
      - scope==global 且 Policy 选了 local_editing → 覆盖为 global_regeneration；
      - 其余保持 Policy 决策，但若该动作 capability 不可用则回退。
    """

    policy = normalize_action(policy_action)
    scope = classify_scope(report, total_frames, local_threshold, mask_valid=mask_valid)
    final = policy
    reason: str | None = None

    local_ok = bool(capability_available.get("local_editing", False)) and mask_valid

    if scope == "local":
        if local_ok:
            if policy != "local_editing":
                final, reason = "local_editing", f"scope_local_override_from_{policy}"
            else:
                final = "local_editing"
        else:
            if policy == "local_editing":
                final = "global_regeneration"
                reason = "local_masked_no_backend_or_mask"
    elif scope in {"global", "unknown"}:
        if policy == "local_editing":
            final = "global_regeneration"
            reason = f"scope_{scope}_forbids_local"

    # capability 兜底：最终动作不可用则回退（reject 恒可用）。
    if final != "reject" and not capability_available.get(final, final == "prompt_repair"):
        fallback = "global_regeneration" if capability_available.get("global_regeneration", False) else "prompt_repair"
        if fallback != final:
            reason = (reason + ";" if reason else "") + f"capability_masked_{final}->{fallback}"
            final = fallback

    return GuardResult(
        policy_action=policy,
        final_action=final,
        scope=scope,
        override_reason=reason,
        capability_available=dict(capability_available),
        total_frames=int(total_frames),
        threshold=float(local_threshold),
        mask_valid=mask_valid,
    )
