"""Strict WanRepairTrialV3 causal audit contract."""

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

WAN_TRIAL_SCHEMA_VERSION = "wan-repair-trial/3.0"
GENERATOR_FAMILY = "wan"
GENERATOR_MODEL = "Wan2.2-TI2V-5B"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WanRepairTrialV3:
    trial_id: str
    group_id: str
    source_candidate: CandidateRecord
    original_prompt: str
    prompt: str
    critic_before: dict[str, Any]
    decision: RepairDecision
    guard: dict[str, Any]
    execution: dict[str, Any]
    before_scores: ScoreBundle
    critic_after: dict[str, Any] | None = None
    after_scores: ScoreBundle | None = None
    repair_improved: bool = False
    successful: bool = False
    failure_reason: str | None = None
    generator_family: str = GENERATOR_FAMILY
    generator_model: str = GENERATOR_MODEL
    generator_revision: str = "unknown"
    critic_profile: str = "sam2_seeded_rules"
    critic_revision: str = "unknown"
    policy_mode: str = "three_action_heuristic"
    decision_source: str = "three_action_policy"
    compatibility: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    schema_version: str = WAN_TRIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WAN_TRIAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported Wan trial schema: {self.schema_version}")
        if not self.trial_id.strip() or not self.group_id.strip():
            raise ValueError("trial_id and group_id must not be empty")
        if self.successful:
            if self.after_scores is None or self.critic_after is None:
                raise ValueError("successful trial requires complete after evidence")
            after_gate = dict(self.metadata.get("gates", {}).get("after", {}) or {})
            if str(after_gate.get("status", "")).upper() != "ACCEPTED":
                raise ValueError("successful trial requires Strict Re-Gate ACCEPTED")
            if not self.repair_improved:
                raise ValueError("successful trial must also be repair_improved")
        elif not self.failure_reason:
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
            "critic": {
                "profile": self.critic_profile,
                "revision": self.critic_revision,
            },
            "policy_mode": self.policy_mode,
            "decision_source": self.decision_source,
            "source_candidate": self.source_candidate.to_dict(),
            "original_prompt": self.original_prompt,
            "prompt": self.prompt,
            "critic_before": self.critic_before,
            "decision": self.decision.to_dict(),
            "guard": self.guard,
            "execution": self.execution,
            "before_scores": self.before_scores.to_dict(),
            "critic_after": self.critic_after,
            "after_scores": None if self.after_scores is None else self.after_scores.to_dict(),
            "physics_gain": self.physics_gain,
            "repair_improved": self.repair_improved,
            "successful": self.successful,
            "failure_reason": self.failure_reason,
            "compatibility": self.compatibility,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WanRepairTrialV3":
        generator = dict(raw.get("generator", {}))
        critic = dict(raw.get("critic", {}))
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            trial_id=str(raw["trial_id"]),
            group_id=str(raw["group_id"]),
            created_at=str(raw.get("created_at", _utc_now())),
            generator_family=str(generator.get("family", GENERATOR_FAMILY)),
            generator_model=str(generator.get("model", GENERATOR_MODEL)),
            generator_revision=str(generator.get("revision", "unknown")),
            critic_profile=str(critic.get("profile", "sam2_seeded_rules")),
            critic_revision=str(critic.get("revision", "unknown")),
            policy_mode=str(raw.get("policy_mode", "three_action_heuristic")),
            decision_source=str(raw.get("decision_source", "three_action_policy")),
            source_candidate=CandidateRecord.from_dict(raw["source_candidate"]),
            original_prompt=str(raw.get("original_prompt", raw.get("prompt", ""))),
            prompt=str(raw.get("prompt", "")),
            critic_before=dict(raw["critic_before"]),
            decision=RepairDecision.from_dict(raw["decision"]),
            guard=dict(raw.get("guard", {})),
            execution=dict(raw["execution"]),
            before_scores=ScoreBundle.from_dict(raw["before_scores"]),
            critic_after=None if raw.get("critic_after") is None else dict(raw["critic_after"]),
            after_scores=(
                None
                if raw.get("after_scores") is None
                else ScoreBundle.from_dict(raw["after_scores"])
            ),
            repair_improved=bool(raw.get("repair_improved", False)),
            successful=bool(raw.get("successful", False)),
            failure_reason=(
                None if raw.get("failure_reason") is None else str(raw["failure_reason"])
            ),
            compatibility={
                str(key): str(value)
                for key, value in dict(raw.get("compatibility", {})).items()
            },
            metadata=dict(raw.get("metadata", {})),
        )


# Import compatibility only; new writers always emit wan-repair-trial/3.0.
WanRepairTrialV2 = WanRepairTrialV3


def to_canonical_trial(
    trial: WanRepairTrialV3,
    *,
    domain: str,
    approved: bool = False,
) -> RepairTrialV1:
    if not approved:
        raise ValueError("canonical trial mapping requires explicit team approval")
    if domain not in {"blender", "hunyuan", "fake"}:
        raise ValueError(f"unsupported canonical domain: {domain!r}")
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
        compatibility={**trial.compatibility, "source_schema": trial.schema_version},
        metadata=dict(trial.metadata),
    )
