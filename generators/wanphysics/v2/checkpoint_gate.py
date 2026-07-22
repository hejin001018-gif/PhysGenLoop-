"""Checkpoint compatibility 分层硬门禁（P0-08）。

修复方案 §28 / 差距审查 P0-08：现有 `load_action_value_repairer` 对 non-deployable /
mismatch 只 print WARNING 后继续加载。V2 增加一个显式的分层模式判定，在 **加载前**
决定是否允许，并产出可审计结论，不修改旧 loader（旧行为保留）。

四层模式：
  disabled              : 不加载 checkpoint
  proxy_research        : 允许加载 non-deployable，但所有产物必须标 proxy_only=true
  actual_trial_research : 要求 actual_trial schema/feature 兼容
  deployment            : 所有 hard gate 通过才允许

本模块只读探测 checkpoint bundle（config/critic_compatibility_v1.json + release_manifest），
不触碰 torch 权重，可在 CPU 环境完整测试。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_GATE_SCHEMA_VERSION = "checkpoint-gate/1.0"

MODE_DISABLED = "disabled"
MODE_PROXY_RESEARCH = "proxy_research"
MODE_ACTUAL_TRIAL_RESEARCH = "actual_trial_research"
MODE_DEPLOYMENT = "deployment"

_VALID_MODES = {MODE_DISABLED, MODE_PROXY_RESEARCH, MODE_ACTUAL_TRIAL_RESEARCH, MODE_DEPLOYMENT}


@dataclass(frozen=True)
class CheckpointGateResult:
    mode: str
    allow_load: bool
    proxy_only: bool
    deployment_ready: bool
    actual_trial_count: int
    source_revision: str
    reasons: tuple[str, ...] = ()
    facts: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CHECKPOINT_GATE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "allow_load": self.allow_load,
            "proxy_only": self.proxy_only,
            "deployment_ready": self.deployment_ready,
            "actual_trial_count": self.actual_trial_count,
            "source_revision": self.source_revision,
            "reasons": list(self.reasons),
            "facts": dict(self.facts),
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def inspect_checkpoint(ckpt_root: str | Path) -> dict[str, Any]:
    """只读读取 checkpoint bundle 的关键事实。"""
    root = Path(ckpt_root)
    compat = _read_json(root / "config" / "critic_compatibility_v1.json")
    release = _read_json(root / "release_manifest.json")
    training = _read_json(root / "reports" / "training_report.json")
    source_revision = str(compat.get("source_revision", release.get("source_revision", "unknown")))
    deployment_ready = source_revision not in {"", "unknown"} and bool(
        release.get("deployment_ready", compat.get("deployment_ready", False))
    )
    actual = int(
        training.get("actual_trial_label_count", release.get("actual_trial_count", 0)) or 0
    )
    return {
        "source_revision": source_revision,
        "deployment_ready": deployment_ready,
        "actual_trial_count": actual,
        "selection_mode": str(training.get("selection_mode", release.get("selection_mode", "unknown"))),
        "has_compat": bool(compat),
        "has_release": bool(release),
    }


def evaluate_checkpoint_gate(
    ckpt_root: str | Path,
    *,
    mode: str,
    allow_proxy_override: bool,
) -> CheckpointGateResult:
    """按模式硬判定是否允许加载。

    - deployment：deployment_ready 必须 True 且 actual_trial_count>0，否则拒绝；
    - actual_trial_research：actual_trial_count>0 才允许；
    - proxy_research：允许加载 non-deployable，但要求显式 allow_proxy_override=True；
    - disabled：永不加载。
    """
    if mode not in _VALID_MODES:
        mode = MODE_PROXY_RESEARCH
    facts = inspect_checkpoint(ckpt_root)
    reasons: list[str] = []
    allow = False
    proxy_only = True

    if mode == MODE_DISABLED:
        reasons.append("mode=disabled")
    elif mode == MODE_DEPLOYMENT:
        if not facts["deployment_ready"]:
            reasons.append("deployment_ready=false")
        if facts["actual_trial_count"] <= 0:
            reasons.append("actual_trial_count=0")
        if facts["source_revision"] in {"", "unknown"}:
            reasons.append("source_revision=unknown")
        allow = not reasons
        proxy_only = not allow
    elif mode == MODE_ACTUAL_TRIAL_RESEARCH:
        if facts["actual_trial_count"] <= 0:
            reasons.append("actual_trial_count=0")
        allow = not reasons
        proxy_only = True
    else:  # proxy_research
        if not allow_proxy_override:
            reasons.append("proxy_research requires explicit allow_proxy_override")
        else:
            allow = True
        proxy_only = True

    return CheckpointGateResult(
        mode=mode,
        allow_load=allow,
        proxy_only=proxy_only,
        deployment_ready=facts["deployment_ready"],
        actual_trial_count=facts["actual_trial_count"],
        source_revision=facts["source_revision"],
        reasons=tuple(reasons),
        facts=facts,
    )
