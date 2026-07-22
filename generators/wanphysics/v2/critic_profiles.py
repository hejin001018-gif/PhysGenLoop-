"""显式 Critic profile（V2）。

修复方案 §26：当前生产入口把只用 Qwen3-VL 做 SAM2 首帧目标种子的链路统一记成
``sam2+vlm``，高估了实际 Critic 能力。V2 用命名 profile 显式区分真实模块组合，并在
每轮 Critic 产物里写 requested/effective/fallback/degraded，供审计。

本模块是纯声明 + 纯函数，无外部依赖，可在任意环境导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

PROFILE_SCHEMA_VERSION = "critic-profile/1.0"

# 已知 profile 及其真实模块定位（对齐修复方案 §26 表）。
KNOWN_PROFILES = (
    "rules_color_blob",
    "sam2_seeded_rules",
    "m4_evidence_vlm",
    "full_pavg_model",
)

# 旧链路的 detector_backend 名 → V2 profile 的保守映射。
_BACKEND_TO_PROFILE = {
    "sam2+vlm": "sam2_seeded_rules",
    "sam2": "sam2_seeded_rules",
    "rules_fallback": "rules_color_blob",
    "rules": "rules_color_blob",
}


@dataclass(frozen=True)
class CriticProfile:
    """一次 Critic 评估的显式能力画像。"""

    requested_profile: str
    effective_profile: str
    modules: dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    degraded_reasons: tuple[str, ...] = ()
    schema_version: str = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.requested_profile not in KNOWN_PROFILES:
            raise ValueError(f"unknown requested_profile: {self.requested_profile!r}")
        if self.effective_profile not in KNOWN_PROFILES:
            raise ValueError(f"unknown effective_profile: {self.effective_profile!r}")

    @property
    def matched(self) -> bool:
        """requested == effective 且未降级，才算 profile 名副其实。"""

        return self.requested_profile == self.effective_profile and not self.fallback_used

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_profile": self.requested_profile,
            "effective_profile": self.effective_profile,
            "modules": dict(self.modules),
            "fallback_used": self.fallback_used,
            "degraded_reasons": list(self.degraded_reasons),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CriticProfile":
        return cls(
            requested_profile=str(raw["requested_profile"]),
            effective_profile=str(raw["effective_profile"]),
            modules=dict(raw.get("modules", {})),
            fallback_used=bool(raw.get("fallback_used", False)),
            degraded_reasons=tuple(str(r) for r in raw.get("degraded_reasons", ())),
        )


def profile_from_backend(
    requested_profile: str,
    detector_backend: str,
    *,
    sam2_postprocess: str = "unknown",
    diagnostics: Mapping[str, Any] | None = None,
) -> CriticProfile:
    """根据 eval_step 返回的 detector_backend 推断真实生效 profile。

    - detector_backend=="rules_fallback" → 记 fallback_used 并降级到 rules_color_blob；
    - SAM2 后处理 disabled（缺 ``_C`` 扩展）→ 记 degraded_reason，但不改 profile 名。
    """

    diagnostics = dict(diagnostics or {})
    effective = _BACKEND_TO_PROFILE.get(detector_backend, requested_profile)
    if effective not in KNOWN_PROFILES:
        effective = "rules_color_blob"

    degraded: list[str] = []
    fallback_used = detector_backend in {"rules_fallback", "rules"}
    if fallback_used:
        degraded.append(f"detector_fallback:{detector_backend}")

    postprocess = str(diagnostics.get("sam2_postprocess", sam2_postprocess))
    if postprocess in {"disabled", "skipped"}:
        degraded.append("sam2_postprocess_disabled")

    modules = {
        "detector": "sam2" if "sam2" in detector_backend else "rules",
        "vlm_object_seed": "sam2" in detector_backend,
        "planner": "template",
        "question_graph": "template",
        "checklist": True,
        "mechanics": True,
        "sam2_postprocess": postprocess,
    }
    return CriticProfile(
        requested_profile=requested_profile if requested_profile in KNOWN_PROFILES else "sam2_seeded_rules",
        effective_profile=effective,
        modules=modules,
        fallback_used=fallback_used,
        degraded_reasons=tuple(degraded),
    )
