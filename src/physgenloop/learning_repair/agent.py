"""融合学习策略、Repair Memory 与可解释回退策略。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .contracts import ACTION_ORDER, LegacyRepairDecision, RepairAction, RepairContext
from .features import ReportFeatureEncoder
from .memory import RepairMemory
from .policy import HeuristicRepairPolicy, RepairPolicy


@dataclass(frozen=True)
class AgentConfig:
    memory_weight: float = 0.25
    memory_k: int = 5
    minimum_policy_confidence: float = 0.4
    minimum_memory_similarity: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.memory_weight <= 1.0:
            raise ValueError("memory_weight must be within [0, 1]")
        if self.memory_k < 1:
            raise ValueError("memory_k must be positive")
        if not 0.0 <= self.minimum_policy_confidence <= 1.0:
            raise ValueError("minimum_policy_confidence must be within [0, 1]")


_DEFAULT_INSTRUCTIONS = {
    "gravity_violation": "Increase physically consistent downward acceleration while preserving scene semantics.",
    "collision_violation": "Restore the contact boundary and prevent object-surface penetration.",
    "friction_violation": "Restore plausible friction and make tangential motion decay after contact.",
    "trajectory_violation": "Replace the discontinuous trajectory with a temporally smooth physical path.",
    "continuity_violation": "Preserve object identity and visibility continuously across adjacent frames.",
    "contact_violation": "Align rebound or stopping with the actual contact event.",
    "appearance_violation": "Preserve object shape, mask, lighting, and appearance through the edit.",
    "unknown_violation": "Regenerate with explicit physical constraints and obtain stronger observations.",
}


def _report_mapping(report: Any) -> Mapping[str, Any]:
    if hasattr(report, "to_dict"):
        report = report.to_dict()
    return report if isinstance(report, Mapping) else {}


def _instructions(report: Any) -> tuple[str, ...]:
    raw = _report_mapping(report)
    result = []
    for violation in raw.get("violations", ()):
        if isinstance(violation, Mapping):
            instruction = str(violation.get("repair_instruction", "")).strip()
            if instruction and instruction not in result:
                result.append(instruction)
    return tuple(result)


class LearningRepairAgent:
    """输出动作决策；动作执行由 Prompt/Hunyuan/局部编辑适配器负责。"""

    def __init__(
        self,
        policy: RepairPolicy,
        *,
        memory: RepairMemory | None = None,
        fallback_policy: RepairPolicy | None = None,
        encoder: ReportFeatureEncoder | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.policy = policy
        self.encoder = encoder or getattr(policy, "encoder", ReportFeatureEncoder())
        self.memory = memory
        self.fallback_policy = fallback_policy or HeuristicRepairPolicy(self.encoder)
        self.config = config or AgentConfig()

    @staticmethod
    def _predict(policy, critic_report, context):
        return policy.predict(critic_report, context=context)

    @staticmethod
    def _mask_unavailable(probabilities, context):
        masked = {
            action: value if context.action_available(action) else 0.0
            for action, value in probabilities.items()
        }
        total = sum(masked.values())
        if total:
            return {action: value / total for action, value in masked.items()}
        return {
            action: 1.0 if action is RepairAction.REJECT else 0.0
            for action in ACTION_ORDER
        }

    def decide(
        self,
        *,
        critic_report: Any,
        prompt: str = "",
        context: RepairContext | None = None,
    ) -> LegacyRepairDecision:
        context = context or RepairContext()
        prediction = self._predict(self.policy, critic_report, context)
        source = prediction.model_id
        probabilities = dict(prediction.action_probabilities)
        matches = ()
        if self.memory is not None and len(self.memory):
            matches = self.memory.retrieve(
                critic_report,
                context=context,
                k=self.config.memory_k,
                minimum_similarity=self.config.minimum_memory_similarity,
            )
            if matches:
                memory_probabilities = self.memory.action_distribution(matches)
                weight = self.config.memory_weight
                probabilities = {
                    action: (1.0 - weight) * probabilities[action]
                    + weight * memory_probabilities[action]
                    for action in ACTION_ORDER
                }
                source = f"{source}+memory"

        probabilities = self._mask_unavailable(probabilities, context)
        action = max(ACTION_ORDER, key=lambda item: probabilities[item])
        confidence = probabilities[action]
        if confidence < self.config.minimum_policy_confidence:
            fallback = self._predict(self.fallback_policy, critic_report, context)
            probabilities = self._mask_unavailable(
                dict(fallback.action_probabilities), context
            )
            action = max(ACTION_ORDER, key=lambda item: probabilities[item])
            confidence = probabilities[action]
            source = f"fallback:{fallback.model_id}"

        category = self.encoder.primary_category(critic_report)
        instructions = _instructions(critic_report)
        instruction = " ".join(instructions) or _DEFAULT_INSTRUCTIONS[category]
        return LegacyRepairDecision(
            action=action,
            confidence=confidence,
            instruction=instruction,
            expected_gain=prediction.expected_gain,
            parameters={
                "primary_category": category,
                "original_prompt": prompt,
                "execution_backend_required": {
                    RepairAction.PROMPT_REPAIR: "prompt_generator",
                    RepairAction.GLOBAL_REGENERATION: "video_generator",
                    RepairAction.LOCAL_EDITING: "local_video_editor",
                    RepairAction.REJECT: "candidate_selector",
                }[action],
            },
            source=source,
            memory_ids=tuple(item.example.sample_id for item in matches),
            action_probabilities={
                item.value: round(probabilities[item], 8) for item in ACTION_ORDER
            },
        )


class LearningRepairPromptAdapter:
    """把结构化动作显式降级为旧 LoopController 的 PromptRepairer 协议。

    该适配器用于当前仅支持“生成下一轮 prompt”的控制器。真实 Local Editing 后端
    接入后应消费 ``LearningRepairAgent.decide``，而不是经过此降级适配器。
    """

    def __init__(
        self,
        agent: LearningRepairAgent,
        *,
        context: RepairContext | None = None,
    ) -> None:
        self.agent = agent
        self.context = context or RepairContext(local_editor_available=False)
        self.last_decision: RepairDecision | None = None
        self._previous_actions = list(self.context.previous_actions)
        self._repair_calls = self.context.attempt_index

    def repair_with_decision(self, *, prompt: str, report):
        runtime_context = replace(
            self.context,
            attempt_index=min(self._repair_calls, self.context.max_attempts),
            previous_actions=tuple(self._previous_actions),
        )
        decision = self.agent.decide(
            critic_report=report, prompt=prompt, context=runtime_context
        )
        self.last_decision = decision
        self._repair_calls += 1
        self._previous_actions.append(decision.action)
        prefix = {
            RepairAction.PROMPT_REPAIR: "Physics correction",
            RepairAction.GLOBAL_REGENERATION: "Regeneration constraint",
            RepairAction.LOCAL_EDITING: "Local-edit fallback constraint",
            RepairAction.REJECT: "Replacement constraint",
        }[decision.action]
        instruction = decision.instruction.strip()
        if not instruction:
            return prompt, decision
        return f"{prompt}\n{prefix}: {instruction}", decision

    def repair(self, *, prompt: str, report) -> str:
        repaired, _decision = self.repair_with_decision(prompt=prompt, report=report)
        return repaired
