"""PhysGenLoop 的有界 Best-of-K 反馈循环。"""

from __future__ import annotations

from pavg_critic.planner import PhysicsPlanResolver, TemplatePhysicsPlanner
from pavg_critic.schemas import CriticRequest, PhysicsPlan

from generators.wanphysics.error_scope import classify_error_scope, has_local_editing_evidence

from .contracts import CandidateEvaluation, LoopConfig, LoopResult, LoopRound
from .interfaces import CandidateCritic, CandidateSelector, PlanResolver, PromptRepairer, VideoGenerator


class LoopController:
    def __init__(
        self,
        *,
        generator: VideoGenerator,
        critic: CandidateCritic,
        repairer: PromptRepairer,
        selector: CandidateSelector,
        plan_resolver: PlanResolver | None = None,
        config: LoopConfig | None = None,
        executor_registry=None,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.repairer = repairer
        self.selector = selector
        self.plan_resolver = plan_resolver or PhysicsPlanResolver(TemplatePhysicsPlanner())
        self.config = config or LoopConfig()
        self.executor_registry = executor_registry

    def _total_frames(self, evaluation: CandidateEvaluation) -> int:
        metadata = getattr(evaluation.candidate, "metadata", {}) or {}
        raw = metadata.get("num_frames", self.config.default_total_frames)
        try:
            total = int(raw)
        except (TypeError, ValueError):
            return 0
        return total if total > 0 else 0

    def run(self, *, prompt: str, physics_plan: PhysicsPlan | None = None) -> LoopResult:
        explicit_plan = physics_plan if physics_plan is not None else PhysicsPlan()
        resolved_plan = self.plan_resolver.resolve(
            CriticRequest(
                video_path="pending://generation",
                prompt=prompt,
                physics_plan=explicit_plan,
            )
        ).plan
        current_prompt = prompt
        original_prompt = prompt
        history: list[LoopRound] = []
        round_winners: list[CandidateEvaluation] = []

        for round_index in range(self.config.max_rounds):
            evaluations: list[CandidateEvaluation] = []
            for offset in range(self.config.candidates_per_round):
                seed = self.config.base_seed + round_index * self.config.candidates_per_round + offset
                if hasattr(self.critic, "prepare_for_generation"):
                    self.critic.prepare_for_generation()
                candidate = self.generator.generate(
                    prompt=current_prompt,
                    physics_plan=resolved_plan,
                    seed=seed,
                )
                report = self.critic.evaluate(
                    candidate,
                    prompt=current_prompt,
                    physics_plan=resolved_plan,
                )
                evaluations.append(CandidateEvaluation(candidate, report))

            frozen_evaluations = tuple(evaluations)
            selected = self.selector.select(frozen_evaluations)
            round_winners.append(selected)
            history.append(
                LoopRound(
                    round_index=round_index,
                    prompt=current_prompt,
                    evaluations=frozen_evaluations,
                    selected_candidate_id=selected.candidate.candidate_id,
                )
            )

            if selected.report.decision == "physical" and selected.report.physics_score >= self.config.acceptance_score:
                return LoopResult(
                    best=selected,
                    history=tuple(history),
                    stop_reason="accepted",
                    resolved_plan=resolved_plan,
                )

            if round_index == self.config.max_rounds - 1:
                break

            if hasattr(self.repairer, "repair_with_decision"):
                next_prompt, decision = self.repairer.repair_with_decision(
                    prompt=current_prompt,
                    report=selected.report,
                )
                policy_action = str(getattr(decision, "action", "")).lower()
                final_action = policy_action
                total_frames = self._total_frames(selected)
                scope = classify_error_scope(
                    selected.report,
                    total_frames,
                    self.config.error_scope_threshold,
                )
                local_evidence = has_local_editing_evidence(selected.report)
                registry_supports_local = bool(
                    self.executor_registry is not None and hasattr(self.executor_registry, "supports")
                ) and self.executor_registry.supports("local_editing")
                if scope == "local" and local_evidence and registry_supports_local:
                    final_action = "local_editing"
                elif scope == "global" and "local_editing" in policy_action:
                    final_action = "global_regeneration"
                selected.report.diagnostics["error_scope"] = {
                    "round_index": round_index,
                    "policy_action": policy_action,
                    "final_action": final_action,
                    "scope": scope,
                    "total_frames": total_frames,
                    "threshold": self.config.error_scope_threshold,
                    "has_local_evidence": local_evidence,
                }

                if "reject" in final_action:
                    return LoopResult(
                        best=self.selector.select(tuple(round_winners)),
                        history=tuple(history),
                        stop_reason="rejected",
                        resolved_plan=resolved_plan,
                    )

                if "local_editing" in final_action and self.executor_registry is not None:
                    try:
                        from physgenloop.learning_repair.contracts import RepairAction, RepairDecision
                        from physgenloop.learning_repair.executors import ExecutionRequest

                        local_decision = decision
                        if str(getattr(decision, "action", "")).lower() != "local_editing":
                            local_decision = RepairDecision(
                                action=RepairAction.LOCAL_EDITING,
                                confidence=decision.confidence,
                                instruction=decision.instruction,
                                action_probabilities=decision.action_probabilities,
                                per_action_values=decision.per_action_values,
                                parameters=dict(getattr(decision, "parameters", {})),
                                local_target=decision.local_target,
                                source=decision.source,
                                abstained=decision.abstained,
                                fallback_reason=decision.fallback_reason,
                                compatibility_id=decision.compatibility_id,
                            )
                        exec_request = ExecutionRequest(
                            candidate=selected.candidate,
                            prompt=current_prompt,
                            physics_plan=resolved_plan,
                            critic_report=selected.report,
                            decision=local_decision,
                            seed=self.config.base_seed + round_index + 1000,
                        )
                        exec_result = self.executor_registry.execute(exec_request)
                        if exec_result.status == "succeeded" and exec_result.candidate is not None:
                            edit_report = self.critic.evaluate(
                                exec_result.candidate,
                                prompt=current_prompt,
                                physics_plan=resolved_plan,
                            )
                            round_winners.append(CandidateEvaluation(exec_result.candidate, edit_report))
                    except Exception as exc:
                        import sys
                        print(f"[LoopController] LOCAL_EDITING failed ({exc}), falling back to prompt repair", file=sys.stderr)
                elif "global_regeneration" in final_action:
                    current_prompt = original_prompt
                else:
                    current_prompt = next_prompt
            else:
                current_prompt = self.repairer.repair(prompt=current_prompt, report=selected.report)

        return LoopResult(
            best=self.selector.select(tuple(round_winners)),
            history=tuple(history),
            stop_reason="max_rounds",
            resolved_plan=resolved_plan,
        )
