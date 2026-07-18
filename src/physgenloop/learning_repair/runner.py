"""Independent Critic→Policy→Executor→Critic loop; LoopController is untouched."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

from physgenloop.contracts import CandidateEvaluation
from physgenloop.learning_repair.contracts import RepairAction

from .baselines import DecisionPolicy
from .campaign import MetricScorer, RewardSpec, _report_dict
from .compatibility import CompatibilityManifest
from .contracts import (
    CandidateRecord,
    ExecutionRequest,
    RepairRunResult,
    RepairTrialV1,
    ScoreBundle,
)
from .executors import ExecutorRegistry
from .recording import JsonlTrialRecorder


@dataclass(frozen=True)
class RunnerConfig:
    max_attempts: int = 3
    acceptance_score: float = 0.80
    base_seed: int = 42
    domain: str = "fake"
    require_quality_metrics: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not 0.0 <= self.acceptance_score <= 1.0:
            raise ValueError("acceptance_score must be within [0, 1]")
        if self.domain not in {"blender", "hunyuan", "fake"}:
            raise ValueError("runner domain must be blender, hunyuan, or fake")


class LearningRepairLoopRunner:
    """Parallel research entry point that never mutates the team mainline."""

    def __init__(
        self,
        *,
        generator: Any,
        critic: Any,
        selector: Any,
        policy: DecisionPolicy,
        executors: ExecutorRegistry,
        semantic_scorer: MetricScorer | None,
        quality_scorer: MetricScorer | None,
        config: RunnerConfig | None = None,
        reward_spec: RewardSpec | None = None,
        recorder: JsonlTrialRecorder | None = None,
        compatibility_manifest: CompatibilityManifest | None = None,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.selector = selector
        self.policy = policy
        self.executors = executors
        self.semantic_scorer = semantic_scorer
        self.quality_scorer = quality_scorer
        self.config = config or RunnerConfig()
        self.reward_spec = reward_spec or RewardSpec()
        self.recorder = recorder
        self.compatibility_manifest = compatibility_manifest
        if self.config.require_quality_metrics and (
            semantic_scorer is None or quality_scorer is None
        ):
            raise ValueError("semantic and quality scorers are required for auditable repair")

    def _report(self, candidate: Any, *, prompt: str, physics_plan: Any) -> Any:
        report = self.critic.evaluate(
            candidate,
            prompt=prompt,
            physics_plan=physics_plan,
        )
        if self.compatibility_manifest is not None:
            self.compatibility_manifest.assert_report(report)
        return report

    def _trial(
        self,
        *,
        run_id: str,
        attempt: int,
        group_id: str,
        source: CandidateEvaluation,
        decision: Any,
        execution: Any,
        after: CandidateEvaluation | None,
        prompt: str,
        successful: bool,
        failure_reason: str | None,
        after_scores: ScoreBundle | None,
    ) -> RepairTrialV1:
        key = f"{run_id}\0{attempt}\0{decision.action.value}\0{source.candidate.candidate_id}"
        trial = RepairTrialV1(
            trial_id=f"trial-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}",
            group_id=group_id,
            domain=self.config.domain,
            source_candidate=CandidateRecord.from_candidate(source.candidate),
            prompt=prompt,
            critic_before=_report_dict(source.report),
            decision=decision,
            execution=execution.to_dict(),
            critic_after=None if after is None else _report_dict(after.report),
            before_scores=ScoreBundle(
                physics=float(_report_dict(source.report)["physics_score"])
            ),
            after_scores=after_scores,
            successful=successful,
            failure_reason=(
                None if successful else failure_reason or "repair_not_accepted"
            ),
            compatibility=(
                {}
                if self.compatibility_manifest is None
                else {
                    "compatibility_id": self.compatibility_manifest.compatibility_id,
                    "critic_model_id": self.compatibility_manifest.critic_model_id,
                }
            ),
            metadata={
                "run_id": run_id,
                "attempt_index": attempt,
                "reward_spec_sha256": self.reward_spec.fingerprint,
            },
        )
        if self.recorder is not None:
            self.recorder.append(trial)
        return trial

    def run(
        self,
        *,
        prompt: str,
        physics_plan: Any,
        group_id: str,
        run_id: str,
    ) -> RepairRunResult:
        current_prompt = prompt
        candidate = self.generator.generate(
            prompt=current_prompt,
            physics_plan=physics_plan,
            seed=self.config.base_seed,
        )
        report = self._report(candidate, prompt=current_prompt, physics_plan=physics_plan)
        current = CandidateEvaluation(candidate, report)
        evaluations = [current]
        candidates = [CandidateRecord.from_candidate(candidate)]
        trials: list[RepairTrialV1] = []
        previous_actions: list[RepairAction] = []

        for attempt in range(self.config.max_attempts):
            raw_report = _report_dict(current.report)
            if (
                raw_report.get("decision") == "physical"
                and float(raw_report["physics_score"]) >= self.config.acceptance_score
            ):
                return RepairRunResult(
                    final_candidate=CandidateRecord.from_candidate(current.candidate),
                    final_report=raw_report,
                    stop_reason="accepted",
                    trials=tuple(trials),
                    candidate_history=tuple(candidates),
                )
            context = self.executors.context(
                attempt_index=attempt,
                max_attempts=self.config.max_attempts,
                previous_actions=tuple(previous_actions),
            )
            decision = self.policy.decide(
                critic_report=current.report,
                candidate=current.candidate,
                prompt=current_prompt,
                context=context,
            )
            if self.compatibility_manifest is not None and (
                decision.compatibility_id != self.compatibility_manifest.compatibility_id
            ):
                raise RuntimeError(
                    "policy decision compatibility_id does not match the active manifest"
                )
            execution = self.executors.execute(
                ExecutionRequest(
                    decision=decision,
                    candidate=current.candidate,
                    critic_report=current.report,
                    prompt=current_prompt,
                    physics_plan=physics_plan,
                    seed=self.config.base_seed + attempt + 1,
                    history=tuple(evaluations),
                    metadata={"run_id": run_id, "group_id": group_id},
                )
            )
            previous_actions.append(decision.action)
            if decision.action is RepairAction.REJECT:
                best = self.selector.select(tuple(evaluations))
                trials.append(
                    self._trial(
                        run_id=run_id,
                        attempt=attempt,
                        group_id=group_id,
                        source=current,
                        decision=decision,
                        execution=execution,
                        after=None,
                        prompt=current_prompt,
                        successful=False,
                        failure_reason="policy_reject",
                        after_scores=None,
                    )
                )
                return RepairRunResult(
                    final_candidate=CandidateRecord.from_candidate(best.candidate),
                    final_report=_report_dict(best.report),
                    stop_reason="rejected",
                    trials=tuple(trials),
                    candidate_history=tuple(candidates),
                )
            if execution.status != "succeeded" or execution.candidate is None:
                trials.append(
                    self._trial(
                        run_id=run_id,
                        attempt=attempt,
                        group_id=group_id,
                        source=current,
                        decision=decision,
                        execution=execution,
                        after=None,
                        prompt=current_prompt,
                        successful=False,
                        failure_reason=execution.failure_reason or "executor_failed",
                        after_scores=None,
                    )
                )
                best = self.selector.select(tuple(evaluations))
                return RepairRunResult(
                    final_candidate=CandidateRecord.from_candidate(best.candidate),
                    final_report=_report_dict(best.report),
                    stop_reason="executor_failed",
                    trials=tuple(trials),
                    candidate_history=tuple(candidates),
                )
            next_prompt = execution.next_prompt or current_prompt
            next_report = self._report(
                execution.candidate,
                prompt=next_prompt,
                physics_plan=physics_plan,
            )
            after = CandidateEvaluation(execution.candidate, next_report)
            evaluations.append(after)
            candidates.append(CandidateRecord.from_candidate(execution.candidate))
            after_dict = _report_dict(next_report)
            semantic = (
                None
                if self.semantic_scorer is None
                else float(
                    self.semantic_scorer(
                        current.candidate, execution.candidate, current_prompt
                    )
                )
            )
            quality = (
                None
                if self.quality_scorer is None
                else float(
                    self.quality_scorer(
                        current.candidate, execution.candidate, current_prompt
                    )
                )
            )
            scores = ScoreBundle(
                physics=float(after_dict["physics_score"]),
                semantic=semantic,
                quality=quality,
            )
            accepted = self.reward_spec.valid(scores)
            trials.append(
                self._trial(
                    run_id=run_id,
                    attempt=attempt,
                    group_id=group_id,
                    source=current,
                    decision=decision,
                    execution=execution,
                    after=after,
                    prompt=current_prompt,
                    successful=accepted,
                    failure_reason=None if accepted else "quality_gate_failed",
                    after_scores=scores,
                )
            )
            current = after
            current_prompt = next_prompt
            if accepted:
                return RepairRunResult(
                    final_candidate=CandidateRecord.from_candidate(after.candidate),
                    final_report=after_dict,
                    stop_reason="accepted",
                    trials=tuple(trials),
                    candidate_history=tuple(candidates),
                )

        best = self.selector.select(tuple(evaluations))
        return RepairRunResult(
            final_candidate=CandidateRecord.from_candidate(best.candidate),
            final_report=_report_dict(best.report),
            stop_reason="max_attempts",
            trials=tuple(trials),
            candidate_history=tuple(candidates),
        )
