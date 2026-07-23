"""Single strict three-action WanPhysics V2 state machine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

from physgenloop.learning_repair.base_contracts import RepairAction, RepairContext
from physgenloop.learning_repair.contracts import ExecutionRequest, ExecutionResult

from .guardrails import (
    GateResult,
    GateThresholds,
    MODE_ENFORCE,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNAVAILABLE,
    evaluate_gate,
)
from .policy_guard import GuardResult, resolve_action

RUNNER_SCHEMA_VERSION = "v2-runner/2.0"
SAMPLE_TERMINAL_STATES = frozenset(
    {
        "ACCEPTED",
        "REJECTED",
        "MAX_ROUNDS",
        "EVALUATION_FAILED",
        "EXECUTION_FAILED",
        "PREFLIGHT_FAILED",
    }
)


class _Generator(Protocol):
    def generate(self, *, prompt: str, seed: int) -> Any: ...


class _Critic(Protocol):
    def evaluate(self, candidate: Any, *, prompt: str) -> Any: ...


@dataclass(frozen=True)
class RunnerConfig:
    max_rounds: int = 2
    candidates_per_round: int = 1
    base_seed: int = 42
    acceptance_mode: str = MODE_ENFORCE
    error_scope_threshold: float = 0.4
    default_total_frames: int = 81
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    fail_on_degraded_critic: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if self.candidates_per_round != 1:
            raise ValueError("V2 strict runtime currently requires candidates_per_round=1")
        if self.acceptance_mode != MODE_ENFORCE:
            raise ValueError("V2 active runtime requires acceptance.mode=enforce")


def _candidate_dict(candidate: Any) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.candidate_id),
        "video_path": str(candidate.video_path),
        "prompt": str(candidate.prompt),
        "seed": int(candidate.seed),
        "metadata": dict(getattr(candidate, "metadata", {}) or {}),
    }


def _report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        return dict(report.to_dict())
    if isinstance(report, dict):
        return dict(report)
    raise TypeError("CriticReport must expose to_dict()")


def _score_dict(report: Any, side: dict[str, float | None]) -> dict[str, float | None]:
    return {
        "physics": float(getattr(report, "physics_score", 0.0)),
        "semantic": side.get("semantic_score"),
        "original_prompt_semantic": side.get("original_prompt_semantic_score"),
        "quality": side.get("quality_score"),
    }


@dataclass
class RoundRecord:
    round_index: int
    candidate_id: str
    state: str
    scope: str | None = None
    policy_action: str | None = None
    executed_action: str | None = None
    gate: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    before_candidate: dict[str, Any] | None = None
    after_candidate: dict[str, Any] | None = None
    before_prompt: str | None = None
    after_prompt: str | None = None
    critic_before: dict[str, Any] | None = None
    critic_after: dict[str, Any] | None = None
    before_scores: dict[str, Any] | None = None
    after_scores: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    guard: dict[str, Any] | None = None
    after_gate: dict[str, Any] | None = None
    execution_id: str | None = None
    terminal_reason: str | None = None

    @property
    def final_action(self) -> str | None:
        return self.executed_action

    @property
    def before_candidate_id(self) -> str | None:
        return None if self.before_candidate is None else str(self.before_candidate["candidate_id"])

    @property
    def after_candidate_id(self) -> str | None:
        return None if self.after_candidate is None else str(self.after_candidate["candidate_id"])

    @property
    def before_physics(self) -> float | None:
        return None if self.before_scores is None else self.before_scores.get("physics")

    @property
    def after_physics(self) -> float | None:
        return None if self.after_scores is None else self.after_scores.get("physics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "candidate_id": self.candidate_id,
            "state": self.state,
            "scope": self.scope,
            "policy_action": self.policy_action,
            "executed_action": self.executed_action,
            "gate": self.gate,
            "execution": self.execution,
            "before_candidate": self.before_candidate,
            "after_candidate": self.after_candidate,
            "before_prompt": self.before_prompt,
            "after_prompt": self.after_prompt,
            "critic_before": self.critic_before,
            "critic_after": self.critic_after,
            "before_scores": self.before_scores,
            "after_scores": self.after_scores,
            "decision": self.decision,
            "guard": self.guard,
            "after_gate": self.after_gate,
            "execution_id": self.execution_id,
            "terminal_reason": self.terminal_reason,
        }


@dataclass
class V2RunResult:
    sample_id: str
    stop_reason: str
    best_candidate_id: str | None
    best_physics_score: float | None
    rounds: list[RoundRecord] = field(default_factory=list)
    final_state: str = "MAX_ROUNDS"
    final_candidate_disposition: str | None = None
    schema_version: str = RUNNER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "stop_reason": self.stop_reason,
            "best_candidate_id": self.best_candidate_id,
            "best_physics_score": self.best_physics_score,
            "final_state": self.final_state,
            "final_candidate_disposition": self.final_candidate_disposition,
            "rounds": [record.to_dict() for record in self.rounds],
        }


@dataclass(frozen=True)
class _Eval:
    candidate: Any
    report: Any
    gate: GateResult
    scores: dict[str, float | None]


class ActionAwareRunnerV2:
    """One entry, one Gate, one Policy decision per rejected round."""

    def __init__(
        self,
        *,
        generator: _Generator,
        critic: _Critic,
        decider: Callable[..., Any],
        executor_registry: Any,
        selector: Any,
        config: RunnerConfig | None = None,
        capability_fn: Callable[[], dict[str, bool]] | None = None,
        mask_valid_fn: Callable[[Any, Any], bool] | None = None,
        local_target_fn: Callable[[Any, Any], Any | None] | None = None,
        side_score_fn: Callable[[Any], dict[str, float | None]] | None = None,
        hooks: "RunnerHooks | None" = None,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.decider = decider
        self.executor_registry = executor_registry
        self.selector = selector
        self.config = config or RunnerConfig()
        self.capability_fn = capability_fn or (
            lambda: {
                "prompt_repair": True,
                "local_editing": False,
                "reject": True,
            }
        )
        self.mask_valid_fn = mask_valid_fn or (lambda report, candidate: False)
        self.local_target_fn = local_target_fn or (lambda report, candidate: None)
        self.side_score_fn = side_score_fn or (
            lambda candidate: {
                "quality_score": None,
                "semantic_score": None,
                "original_prompt_semantic_score": None,
            }
        )
        self.hooks = hooks or RunnerHooks()

    def _physics(self, report: Any) -> float:
        return float(getattr(report, "physics_score", 0.0))

    def _total_frames(self, candidate: Any) -> int:
        try:
            value = int((getattr(candidate, "metadata", {}) or {}).get(
                "num_frames", self.config.default_total_frames
            ))
        except (TypeError, ValueError):
            value = self.config.default_total_frames
        return value if value > 0 else self.config.default_total_frames

    def _is_degraded(self, report: Any) -> bool:
        diagnostics = dict(getattr(report, "diagnostics", {}) or {})
        return bool(
            diagnostics.get("degraded")
            or diagnostics.get("fallback_used")
            or diagnostics.get("sam2_postprocess") in {"disabled", "hole_filling_disabled"}
        )

    def _evaluate(self, candidate: Any, prompt: str) -> _Eval | None:
        try:
            report = self.critic.evaluate(candidate, prompt=prompt)
        except Exception:  # noqa: BLE001
            return None
        if report is None:
            return None
        side = self.side_score_fn(candidate)
        gate = evaluate_gate(
            report=report,
            mode=self.config.acceptance_mode,
            thresholds=self.config.thresholds,
            semantic_score=side.get("semantic_score"),
            original_prompt_semantic_score=side.get(
                "original_prompt_semantic_score"
            ),
            quality_score=side.get("quality_score"),
            critic_degraded=self._is_degraded(report),
            fail_on_degraded=self.config.fail_on_degraded_critic,
        )
        return _Eval(candidate, report, gate, _score_dict(report, side))

    def _best(self, history: list[_Eval]) -> _Eval | None:
        eligible = [item for item in history if item.gate.status != STATUS_UNAVAILABLE]
        if not eligible:
            return None
        return max(eligible, key=lambda item: self._physics(item.report))

    def _decide(
        self,
        report: Any,
        evaluation: _Eval,
        context: RepairContext,
    ) -> Any:
        try:
            return self.decider(report, evaluation, context)
        except TypeError:
            return self.decider(report, evaluation)

    def run(self, *, sample_id: str, prompt: str) -> V2RunResult:
        cfg = self.config
        original_prompt = str(prompt)
        result = V2RunResult(sample_id, "max_rounds", None, None)
        history: list[_Eval] = []
        previous_actions: list[RepairAction] = []
        self.hooks.on_state(sample_id, "GENERATING", 0)
        try:
            current_candidate = self.generator.generate(prompt=original_prompt, seed=cfg.base_seed)
        except Exception as exc:  # noqa: BLE001
            result.stop_reason = f"initial_generation_failed:{type(exc).__name__}"
            result.final_state = "EXECUTION_FAILED"
            self.hooks.on_state(sample_id, "EXECUTION_FAILED", 0)
            return result
        current_prompt = original_prompt
        self.hooks.on_state(sample_id, "CRITIC_RUNNING", 0)
        current_eval = self._evaluate(current_candidate, current_prompt)
        if current_eval is None:
            result.stop_reason = "critic_roundtrip_failed"
            result.final_state = "EVALUATION_FAILED"
            self.hooks.on_state(sample_id, "EVALUATION_FAILED", 0)
            return result
        history.append(current_eval)

        for round_index in range(cfg.max_rounds):
            gate = current_eval.gate
            record = RoundRecord(
                round_index=round_index,
                candidate_id=str(current_candidate.candidate_id),
                state="GATE_EVALUATED",
                gate=gate.to_dict(),
                before_candidate=_candidate_dict(current_candidate),
                before_prompt=current_prompt,
                critic_before=_report_dict(current_eval.report),
                before_scores=dict(current_eval.scores),
            )
            if gate.status == STATUS_ACCEPTED:
                record.state = "ACCEPTED"
                record.terminal_reason = "accepted"
                result.rounds.append(record)
                result.stop_reason = "accepted"
                result.final_state = "ACCEPTED"
                self.hooks.on_state(sample_id, "ACCEPTED", round_index)
                break
            if gate.status == STATUS_UNAVAILABLE:
                record.state = "EVALUATION_FAILED"
                record.terminal_reason = "gate_unavailable"
                result.rounds.append(record)
                result.stop_reason = "evaluation_unavailable"
                result.final_state = "EVALUATION_FAILED"
                self.hooks.on_state(sample_id, "EVALUATION_FAILED", round_index)
                break
            if gate.status != STATUS_REJECTED:
                raise RuntimeError(f"unexpected Gate status: {gate.status}")

            capabilities = dict(self.capability_fn())
            context = RepairContext(
                attempt_index=round_index,
                max_attempts=cfg.max_rounds,
                prompt_repair_available=bool(capabilities.get("prompt_repair", False)),
                local_editor_available=bool(capabilities.get("local_editing", False)),
                semantic_score=current_eval.scores.get("semantic"),
                original_prompt_semantic_score=current_eval.scores.get(
                    "original_prompt_semantic"
                ),
                quality_score=current_eval.scores.get("quality"),
                previous_actions=tuple(previous_actions),
            )
            decision = self._decide(current_eval.report, current_eval, context)
            if decision.action is RepairAction.LOCAL_EDITING:
                target = self.local_target_fn(current_eval.report, current_candidate)
                if target is not None:
                    decision = replace(decision, local_target=target)
            mask_valid = self.mask_valid_fn(current_eval.report, current_candidate)
            guard = resolve_action(
                policy_action=decision.action,
                report=current_eval.report,
                total_frames=self._total_frames(current_candidate),
                local_threshold=cfg.error_scope_threshold,
                capability_available=capabilities,
                mask_valid=mask_valid,
            )
            execution_id = f"{sample_id}-exec{round_index + 1}"
            record.state = "DECISION_READY"
            record.scope = guard.scope
            record.policy_action = decision.action.value
            record.executed_action = guard.final_action
            record.decision = decision.to_dict()
            record.guard = guard.to_dict()
            record.execution_id = execution_id
            self.hooks.on_decision(
                sample_id,
                current_candidate.candidate_id,
                decision,
                guard,
                round_index,
                execution_id,
            )

            execution_decision = decision
            if not guard.allowed:
                execution_decision = replace(
                    decision,
                    action=RepairAction.REJECT,
                    local_target=None,
                    fallback_reason=guard.blocked_reason,
                )
            request = ExecutionRequest(
                decision=execution_decision,
                candidate=current_candidate,
                critic_report=current_eval.report,
                prompt=current_prompt,
                seed=cfg.base_seed + round_index + 1000,
                history=tuple(history),
                metadata={
                    "original_prompt": original_prompt,
                    "execution_id": execution_id,
                    "guard": guard.to_dict(),
                },
            )
            self.hooks.on_state(sample_id, "EXECUTING", round_index)
            execution = self.executor_registry.execute(request)
            record.execution = execution.to_dict()
            self.hooks.on_execution(sample_id, execution)
            previous_actions.append(execution.action)

            if execution.status == "rejected" or execution.terminal:
                record.state = "REJECTED"
                record.terminal_reason = guard.blocked_reason or "policy_reject"
                result.rounds.append(record)
                result.stop_reason = "rejected"
                result.final_state = "REJECTED"
                self.hooks.on_state(sample_id, "REJECTED", round_index)
                break
            if execution.status == "failed" or execution.candidate is None:
                record.state = "EXECUTION_FAILED"
                record.terminal_reason = execution.failure_reason or "executor_failed"
                result.rounds.append(record)
                result.stop_reason = "execution_failed"
                result.final_state = "EXECUTION_FAILED"
                self.hooks.on_state(sample_id, "EXECUTION_FAILED", round_index)
                break

            after_candidate = execution.candidate
            after_prompt = execution.next_prompt or current_prompt
            self.hooks.on_state(sample_id, "RE_EVALUATING", round_index)
            after_eval = self._evaluate(after_candidate, after_prompt)
            record.after_candidate = _candidate_dict(after_candidate)
            record.after_prompt = after_prompt
            if after_eval is None:
                record.state = "EVALUATION_FAILED"
                record.terminal_reason = "after_critic_roundtrip_failed"
                result.rounds.append(record)
                result.stop_reason = "evaluation_failed"
                result.final_state = "EVALUATION_FAILED"
                self.hooks.on_state(sample_id, "EVALUATION_FAILED", round_index)
                break
            record.critic_after = _report_dict(after_eval.report)
            record.after_scores = dict(after_eval.scores)
            record.after_gate = after_eval.gate.to_dict()
            record.state = "RE_EVALUATED"
            result.rounds.append(record)
            history.append(after_eval)
            if after_eval.gate.status == STATUS_ACCEPTED:
                result.stop_reason = "accepted"
                result.final_state = "ACCEPTED"
                self.hooks.on_state(sample_id, "ACCEPTED", round_index)
                current_eval = after_eval
                break
            if after_eval.gate.status == STATUS_UNAVAILABLE:
                record.terminal_reason = "after_gate_unavailable"
                result.stop_reason = "evaluation_unavailable"
                result.final_state = "EVALUATION_FAILED"
                self.hooks.on_state(sample_id, "EVALUATION_FAILED", round_index)
                current_eval = after_eval
                break
            current_candidate = after_candidate
            current_prompt = after_prompt
            current_eval = after_eval
        else:
            result.final_state = "MAX_ROUNDS"
            result.stop_reason = "max_rounds"
            result.final_candidate_disposition = "best_effort"
            self.hooks.on_state(sample_id, "MAX_ROUNDS", cfg.max_rounds)

        best = self._best(history)
        if best is not None:
            result.best_candidate_id = str(best.candidate.candidate_id)
            result.best_physics_score = self._physics(best.report)
        if result.final_state not in SAMPLE_TERMINAL_STATES:
            raise RuntimeError(f"runner ended in non-terminal state: {result.final_state}")
        return result


class RunnerHooks:
    def on_state(self, sample_id: str, state: str, round_index: int) -> None:
        return None

    def on_decision(
        self,
        sample_id: str,
        candidate_id: str,
        decision: Any,
        guard: GuardResult,
        round_index: int,
        execution_id: str,
    ) -> None:
        return None

    def on_execution(self, sample_id: str, exec_result: Any) -> None:
        return None
