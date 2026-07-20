"""ActionValueDecisionPolicy 的 PromptRepairer 协议适配器。"""
from __future__ import annotations

from pathlib import Path

from physgenloop.learning_repair import (
    ActionValueDecisionPolicy,
    CompatibilityManifest,
    RepairAction,
    RepairContext,
    RepairMemory,
    TorchActionValuePolicy,
)
from physgenloop.contracts import GeneratedCandidate
from pavg_critic.schemas import CriticReport

_ACTION_PREFIX = {
    RepairAction.PROMPT_REPAIR: "Physics correction",
    RepairAction.GLOBAL_REGENERATION: "Regeneration constraint",
    RepairAction.LOCAL_EDITING: "Local-edit fallback constraint",
    RepairAction.REJECT: "Replacement constraint",
}


class ActionValueRepairer:
    def __init__(
        self,
        decision_policy: ActionValueDecisionPolicy,
        max_attempts: int = 2,
        proxy_memory: RepairMemory | None = None,
        memory_weight: float = 0.25,
        local_editor_available: bool = False,
    ) -> None:
        self._policy = decision_policy
        self._max_attempts = max_attempts
        self._attempt_index = 0
        self._previous_actions: list[RepairAction] = []
        self._proxy_memory = proxy_memory
        self._memory_weight = memory_weight
        self._local_editor_available = local_editor_available

    def repair_with_decision(self, *, prompt: str, report: CriticReport):
        from physgenloop.learning_repair.baselines import _target
        from physgenloop.learning_repair.contracts import RepairDecision

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

        if self._proxy_memory is not None:
            matches = self._proxy_memory.retrieve(report, context=context)
            if matches:
                mem_dist = self._proxy_memory.action_distribution(matches)
                w = self._memory_weight
                blended_values = {
                    action: (1.0 - w) * decision.per_action_values.get(action, 0.0)
                    + w * mem_dist.get(action, 0.0)
                    for action in RepairAction
                }
                best_action = max(blended_values, key=lambda a: blended_values[a])
                if best_action != decision.action:
                    decision = RepairDecision(
                        action=best_action,
                        confidence=decision.confidence,
                        instruction=decision.instruction,
                        action_probabilities=decision.action_probabilities,
                        per_action_values=blended_values,
                        parameters=decision.parameters,
                        local_target=decision.local_target,
                        source=f"{decision.source}+proxy-memory",
                        compatibility_id=decision.compatibility_id,
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
            "[repairer] WARNING: checkpoint compatibility manifest has deployment_ready=False "
            "(source_revision='unknown'). This is a proxy-trained checkpoint; "
            "actual_trial_count=0. Proceeding as proxy mode.",
        )

    _ROOT = Path("/root/PhysGenLoop-")
    _LOCAL_COMPAT = _ROOT / "configs/learning_repair/critic_compatibility_v1.json"
    if _LOCAL_COMPAT.exists():
        try:
            local_compat = CompatibilityManifest.load(str(_LOCAL_COMPAT))
            local_compat.verify_files(
                critic_config=_ROOT / "configs/default.yaml",
                critic_schema=_ROOT / "schemas/critic_output.schema.json",
                feature_schema=_ROOT / "configs/learning_repair/feature_schema.json",
            )
        except Exception as exc:
            print(
                f"[repairer] WARNING: Critic file hash mismatch ({exc}). "
                "Policy was trained on a different Critic revision.",
            )

    learned_policy = TorchActionValuePolicy.load(
        str(ckpt_path / "model/best_action_value_policy.pt"),
        device="cpu",
        compatibility_manifest=compatibility,
    )
    decision_policy = ActionValueDecisionPolicy(learned_policy, minimum_confidence=0.35)

    proxy_memory: RepairMemory | None = None
    memory_jsonl = ckpt_path / "memory/proxy_memory_train.jsonl"
    if memory_jsonl.exists():
        try:
            proxy_memory = RepairMemory.from_manifest(memory_jsonl)
            print(
                f"[repairer] loaded proxy memory: {len(proxy_memory)} examples from {memory_jsonl}",
            )
        except Exception as exc:
            print(f"[repairer] WARNING: failed to load proxy memory ({exc}), proceeding without memory")

    return ActionValueRepairer(
        decision_policy,
        max_attempts=max_attempts,
        proxy_memory=proxy_memory,
        memory_weight=0.25,
        local_editor_available=local_editor_available,
    )
