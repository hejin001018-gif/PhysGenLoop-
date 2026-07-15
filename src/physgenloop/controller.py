"""PhysGenLoop 的有界 Best-of-K 反馈循环。"""

from __future__ import annotations

from pavg_critic.planner import PhysicsPlanResolver, TemplatePhysicsPlanner
from pavg_critic.schemas import CriticRequest, PhysicsPlan

from .contracts import CandidateEvaluation, LoopConfig, LoopResult, LoopRound
from .interfaces import (
    CandidateCritic,
    CandidateSelector,
    PlanResolver,
    PromptRepairer,
    VideoGenerator,
)


class LoopController:
    """编排生成、Critic、修复和选择，不绑定具体模型后端。"""

    def __init__(
        self,
        *,
        generator: VideoGenerator,
        critic: CandidateCritic,
        repairer: PromptRepairer,
        selector: CandidateSelector,
        plan_resolver: PlanResolver | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.repairer = repairer
        self.selector = selector
        self.plan_resolver = plan_resolver or PhysicsPlanResolver(
            TemplatePhysicsPlanner()
        )
        self.config = config or LoopConfig()

    def run(
        self, *, prompt: str, physics_plan: PhysicsPlan | None = None
    ) -> LoopResult:
        explicit_plan = physics_plan if physics_plan is not None else PhysicsPlan()
        resolved_plan = self.plan_resolver.resolve(
            CriticRequest(
                video_path="pending://generation",
                prompt=prompt,
                physics_plan=explicit_plan,
            )
        ).plan
        current_prompt = prompt
        history: list[LoopRound] = []
        round_winners: list[CandidateEvaluation] = []

        for round_index in range(self.config.max_rounds):
            evaluations: list[CandidateEvaluation] = []
            for offset in range(self.config.candidates_per_round):
                seed = (
                    self.config.base_seed
                    + round_index * self.config.candidates_per_round
                    + offset
                )
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

            if (
                selected.report.decision == "physical"
                and selected.report.physics_score >= self.config.acceptance_score
            ):
                return LoopResult(
                    best=selected,
                    history=tuple(history),
                    stop_reason="accepted",
                    resolved_plan=resolved_plan,
                )

            current_prompt = self.repairer.repair(
                prompt=current_prompt,
                report=selected.report,
            )

        return LoopResult(
            best=self.selector.select(tuple(round_winners)),
            history=tuple(history),
            stop_reason="max_rounds",
            resolved_plan=resolved_plan,
        )
