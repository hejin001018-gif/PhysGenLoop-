"""WanRepairTrialV2 与 canonical adapter（V2）。

修复 P1-2 / 方案 §20：现有 canonical ``RepairTrialV1`` 的 domain 仅允许
``blender/hunyuan/fake``，而真实 Generator 是 Wan2.2。V2 **不**为了通过旧校验把 Wan
Trial 冒充成 Hunyuan，而是新增显式 :class:`WanRepairTrialV2`，完整保存 generator /
critic profile / policy mode / research_only 等来源信息；只有团队批准 domain/schema
映射后，才通过 :func:`to_canonical_trial` 转成 ``RepairTrialV1``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from physgenloop.learning_repair.contracts import (
    CandidateRecord,
    RepairDecision,
    RepairTrialV1,
    ScoreBundle,
)

WAN_TRIAL_SCHEMA_VERSION = "wan-repair-trial/2.0"

GENERATOR_FAMILY = "wan"
GENERATOR_MODEL = "Wan2.2-TI2V-5B"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WanRepairTrialV2:
    """Wan2.2 域的真实 before/action/after Trial，显式标注来源与 research_only。"""

    trial_id: str
    group_id: str
    source_candidate: CandidateRecord
    prompt: str
    critic_before: dict[str, Any]
    decision: RepairDecision
    execution: dict[str, Any]
    before_scores: ScoreBundle
    critic_after: dict[str, Any] | None = None
    after_scores: ScoreBundle | None = None
    successful: bool = False
    failure_reason: str | None = None
    # 来源 provenance（方案 §20）
    generator_family: str = GENERATOR_FAMILY
    generator_model: str = GENERATOR_MODEL
    generator_revision: str = "unknown"
    critic_profile: str = "sam2_seeded_rules"
    critic_revision: str = "unknown"
    policy_mode: str = "proxy_research"
    research_only: bool = True
    compatibility: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    schema_version: str = WAN_TRIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.trial_id.strip() or not self.group_id.strip():
            raise ValueError("trial_id and group_id must not be empty")
        # 与 RepairTrialV1 一致的最小一致性约束。
        if self.successful and self.after_scores is None:
            raise ValueError("successful trial requires after_scores")
        if not self.successful and not self.failure_reason:
            raise ValueError("unsuccessful trial requires failure_reason")

    @property
    def physics_gain(self) -> float | None:
        if self.after_scores is None:
            return None
        return self.after_scores.physics - self.before_scores.physics

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "group_id": self.group_id,
            "created_at": self.created_at,
            "generator": {
                "family": self.generator_family,
                "model": self.generator_model,
                "revision": self.generator_revision,
            },
            "critic": {"profile": self.critic_profile, "revision": self.critic_revision},
            "policy_mode": self.policy_mode,
            "research_only": self.research_only,
            "source_candidate": self.source_candidate.to_dict(),
            "prompt": self.prompt,
            "critic_before": self.critic_before,
            "decision": self.decision.to_dict(),
            "execution": self.execution,
            "before_scores": self.before_scores.to_dict(),
            "critic_after": self.critic_after,
            "after_scores": None if self.after_scores is None else self.after_scores.to_dict(),
            "physics_gain": self.physics_gain,
            "successful": self.successful,
            "failure_reason": self.failure_reason,
            "compatibility": self.compatibility,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WanRepairTrialV2":
        gen = dict(raw.get("generator", {}))
        critic = dict(raw.get("critic", {}))
        return cls(
            schema_version=str(raw.get("schema_version", WAN_TRIAL_SCHEMA_VERSION)),
            trial_id=str(raw["trial_id"]),
            group_id=str(raw["group_id"]),
            created_at=str(raw.get("created_at", _utc_now())),
            generator_family=str(gen.get("family", GENERATOR_FAMILY)),
            generator_model=str(gen.get("model", GENERATOR_MODEL)),
            generator_revision=str(gen.get("revision", "unknown")),
            critic_profile=str(critic.get("profile", "sam2_seeded_rules")),
            critic_revision=str(critic.get("revision", "unknown")),
            policy_mode=str(raw.get("policy_mode", "proxy_research")),
            research_only=bool(raw.get("research_only", True)),
            source_candidate=CandidateRecord.from_dict(raw["source_candidate"]),
            prompt=str(raw.get("prompt", "")),
            critic_before=dict(raw["critic_before"]),
            decision=RepairDecision.from_dict(raw["decision"]),
            execution=dict(raw["execution"]),
            before_scores=ScoreBundle.from_dict(raw["before_scores"]),
            critic_after=None if raw.get("critic_after") is None else dict(raw["critic_after"]),
            after_scores=(
                None if raw.get("after_scores") is None else ScoreBundle.from_dict(raw["after_scores"])
            ),
            successful=bool(raw.get("successful", False)),
            failure_reason=None if raw.get("failure_reason") is None else str(raw["failure_reason"]),
            compatibility={str(k): str(v) for k, v in dict(raw.get("compatibility", {})).items()},
            metadata=dict(raw.get("metadata", {})),
        )


def to_canonical_trial(
    trial: WanRepairTrialV2,
    *,
    domain: str,
    approved: bool = False,
) -> RepairTrialV1:
    """把 WanRepairTrialV2 转成 canonical ``RepairTrialV1``，需显式批准。

    默认 ``approved=False`` 会拒绝转换——防止未经团队同意就把 Wan Trial 映射成
    某个 canonical domain 并混入正式训练集（方案 §20 / §28）。
    """

    if not approved:
        raise ValueError(
            "canonical trial mapping requires explicit team approval (approved=True); "
            "WanRepairTrialV2 must not silently masquerade as a canonical domain"
        )
    if domain not in {"blender", "hunyuan", "fake"}:
        raise ValueError(f"unsupported canonical domain: {domain!r}")
    compat = dict(trial.compatibility)
    compat.update(
        {
            "source_schema": trial.schema_version,
            "generator_family": trial.generator_family,
            "generator_model": trial.generator_model,
            "critic_profile": trial.critic_profile,
            "mapped_from": "wan_repair_trial_v2",
        }
    )
    return RepairTrialV1(
        trial_id=trial.trial_id,
        group_id=trial.group_id,
        domain=domain,
        source_candidate=trial.source_candidate,
        prompt=trial.prompt,
        critic_before=trial.critic_before,
        decision=trial.decision,
        execution=trial.execution,
        before_scores=trial.before_scores,
        critic_after=trial.critic_after,
        after_scores=trial.after_scores,
        successful=trial.successful,
        failure_reason=trial.failure_reason,
        compatibility=compat,
        metadata={**trial.metadata, "research_only": trial.research_only},
    )
