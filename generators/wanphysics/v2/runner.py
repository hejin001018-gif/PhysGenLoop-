"""V2 Action-aware Runner：Best-of-K + Decision → Executor → Re-Critic 状态机。

修复 P0-4：现有 ``ActionAwareRunnerV2`` 只对 local_editing 走 executor，其余动作直接改
current_prompt，且 local 编辑结果只进 round_winners、不成为下一轮 current。V2 Runner
把四动作全部经由 :class:`Decision → Executor → 立即复评` 处理，编辑/重生成结果成为
下一轮 current candidate，并对每一步写完整 RepairTrace。

设计要点（对齐修复方案 §11 状态机）：
  1. Policy 只决策一次（由注入的 decision_fn 提供），Executor 不再调用 Policy；
  2. 每个动作立即用同一 Critic 复评；
  3. 状态只前进，写入 sample_status + repair_trace；
  4. reject 从历史候选收口，terminal。

Runner 通过依赖注入接收 generator/critic/decider/executor_registry/selector，
因此可用 mock 组件在 CPU 上完整验证状态机（不依赖 GPU/模型）。

**不修改** ``ActionAwareRunnerV2``；这是并存的独立编排层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import ExecutionRequest, ExecutionResult

from .guardrails import GateResult, GateThresholds, evaluate_gate
from .policy_guard import GuardResult, normalize_action, resolve_action

RUNNER_SCHEMA_VERSION = "v2-runner/1.0"


class _Generator(Protocol):
    def generate(self, *, prompt: str, physics_plan: Any, seed: int) -> Any: ...


class _Critic(Protocol):
    def evaluate(self, candidate: Any, *, prompt: str, physics_plan: Any) -> Any: ...


@dataclass(frozen=True)
class RunnerConfig:
    max_rounds: int = 2
    candidates_per_round: int = 1
    base_seed: int = 42
    acceptance_mode: str = "shadow"
    error_scope_threshold: float = 0.4
    default_total_frames: int = 81
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    fail_on_degraded_critic: bool = False
    # P0-04：force_action 优先于 accept 短路（force_trial 模式）。
    force_action: str | None = None
    # P0-02：plan completeness gate（enforce 时非空 prompt 空 plan 拒绝）。
    require_plan: bool = False


@dataclass
class RoundRecord:
    round_index: int
    candidate_id: str
    state: str
    scope: str | None = None
    policy_action: str | None = None
    final_action: str | None = None
    gate: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    # P0-01：完整因果配对字段。
    before_candidate_id: str | None = None
    after_candidate_id: str | None = None
    before_physics: float | None = None
    after_physics: float | None = None
    after_gate: dict[str, Any] | None = None
    execution_id: str | None = None
    terminal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "candidate_id": self.candidate_id,
            "state": self.state,
            "scope": self.scope,
            "policy_action": self.policy_action,
            "final_action": self.final_action,
            "gate": self.gate,
            "execution": self.execution,
            "before_candidate_id": self.before_candidate_id,
            "after_candidate_id": self.after_candidate_id,
            "before_physics": self.before_physics,
            "after_physics": self.after_physics,
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
    trace: list[dict[str, Any]] = field(default_factory=list)
    final_state: str = "COMPLETED"
    schema_version: str = RUNNER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "stop_reason": self.stop_reason,
            "best_candidate_id": self.best_candidate_id,
            "best_physics_score": self.best_physics_score,
            "final_state": self.final_state,
            "rounds": [r.to_dict() for r in self.rounds],
        }


class ActionAwareRunnerV2:
    """独立于 ActionAwareRunnerV2 的四动作闭环编排器。

    参数（全部注入，便于 mock）：
      generator      : 有 generate() 的视频生成器
      critic         : 有 evaluate() 的 Critic（返回带 violations 的 report）
      decider        : callable(report, context) -> RepairDecision（Policy，只调一次/轮）
      executor_registry : 有 execute()/supports() 的注册表
      selector       : 有 select() 的候选选择器（reject 收口用）
      capability_fn  : callable() -> dict[str,bool]，各动作后端是否可用
      mask_valid_fn  : callable(report, candidate) -> bool，是否存在有效 mask
      hooks          : 可选审计回调（写 artifacts）
    """

    def __init__(
        self,
        *,
        generator: _Generator,
        critic: _Critic,
        decider: Callable[[Any, Any], Any],
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
        self.capability_fn = capability_fn or (lambda: {"prompt_repair": True, "global_regeneration": True, "local_editing": False, "reject": True})
        self.mask_valid_fn = mask_valid_fn or (lambda report, cand: False)
        self.local_target_fn = local_target_fn or (lambda report, cand: None)
        # G8：quality/semantic scorer 结果由外部 critic 计算后通过此 fn 注入 gate。
        self.side_score_fn = side_score_fn or (lambda cand: {"quality_score": None, "semantic_score": None})
        self.hooks = hooks or RunnerHooks()

    def _physics(self, report: Any) -> float:
        return float(getattr(report, "physics_score", 0.0))

    def _total_frames(self, candidate: Any) -> int:
        meta = getattr(candidate, "metadata", {}) or {}
        try:
            n = int(meta.get("num_frames", self.config.default_total_frames))
        except (TypeError, ValueError):
            return self.config.default_total_frames
        return n if n > 0 else self.config.default_total_frames

    def _is_degraded(self, report: Any) -> bool:
        diag = getattr(report, "diagnostics", {}) or {}
        return bool(diag.get("sam2_postprocess") == "disabled" or diag.get("fallback_used"))

    def _plan_ready(self, prompt: str, physics_plan: Any) -> bool:
        """P0-02：非空 prompt 必须有非空 plan（有 events/constraints）才算 ready。"""
        if not str(prompt).strip():
            return True  # 空 prompt 无所谓
        if physics_plan is None:
            return False
        # PhysicsPlan 有 relations/constraints 即视为有内容。
        for attr in ("relations", "constraints", "events", "entities"):
            val = getattr(physics_plan, attr, None)
            if val:
                return True
        return False

    def run(self, *, sample_id: str, prompt: str, physics_plan: Any) -> V2RunResult:
        cfg = self.config
        original_prompt = prompt
        result = V2RunResult(
            sample_id=sample_id,
            stop_reason="max_rounds",
            best_candidate_id=None,
            best_physics_score=None,
        )
        history: list[Any] = []
        best_eval: Any = None
        exec_counter = 0

        self.hooks.on_state(sample_id, "CREATED", 0)

        # P0-01：初始候选只生成一次；之后 current 沿因果链推进。
        self.hooks.on_state(sample_id, "GENERATING", 0)
        current_candidate = self.generator.generate(
            prompt=prompt, physics_plan=physics_plan, seed=cfg.base_seed
        )
        current_prompt = prompt
        self.hooks.on_state(sample_id, "GENERATED", 0)

        for round_index in range(cfg.max_rounds):
            # --- 评估 current candidate ---
            self.hooks.on_state(sample_id, "CRITIC_RUNNING", round_index)
            report = self.critic.evaluate(current_candidate, prompt=current_prompt, physics_plan=physics_plan)
            if report is None:
                self.hooks.on_state(sample_id, "CRITIC_FAILED", round_index)
                result.rounds.append(
                    RoundRecord(round_index, str(getattr(current_candidate, "candidate_id", "?")), "CRITIC_FAILED",
                                before_candidate_id=str(getattr(current_candidate, "candidate_id", "?")),
                                terminal_reason="critic_roundtrip_failed")
                )
                result.stop_reason = "critic_failed"
                result.final_state = "CRITIC_FAILED"
                break
            self.hooks.on_state(sample_id, "CRITIC_COMPLETED", round_index)

            evaluation = _Eval(current_candidate, report)
            history.append(evaluation)
            phys = self._physics(report)
            if best_eval is None or phys > self._physics(best_eval.report):
                best_eval = evaluation

            # --- 接受门（含 plan completeness）---
            side = self.side_score_fn(current_candidate)
            plan_ready = self._plan_ready(prompt, physics_plan)
            gate = evaluate_gate(
                report=report,
                mode=cfg.acceptance_mode,
                thresholds=cfg.thresholds,
                semantic_score=side.get("semantic_score"),
                quality_score=side.get("quality_score"),
                critic_degraded=self._is_degraded(report),
                fail_on_degraded=cfg.fail_on_degraded_critic,
            )
            rec = RoundRecord(
                round_index, str(current_candidate.candidate_id), "CRITIC_COMPLETED",
                gate=gate.to_dict(),
                before_candidate_id=str(current_candidate.candidate_id),
                before_physics=phys,
            )

            # P0-02：enforce 且要求 plan 但 plan 不完整 → 不接受。
            plan_blocks = cfg.require_plan and cfg.acceptance_mode == "enforce" and not plan_ready
            accepted = gate.accepted and not plan_blocks

            # P0-04：force_action 优先于 accept 短路。
            if accepted and cfg.force_action is None:
                self.hooks.on_state(sample_id, "ACCEPTED", round_index)
                rec.state = "ACCEPTED"
                rec.terminal_reason = "accepted"
                result.rounds.append(rec)
                result.stop_reason = "accepted"
                result.final_state = "ACCEPTED"
                best_eval = evaluation
                break

            # 最后一轮不再决策/执行。
            if round_index == cfg.max_rounds - 1:
                rec.terminal_reason = "max_rounds"
                result.rounds.append(rec)
                break

            # --- 决策（Policy 只一次；force 时用 forced decider）---
            decision = self.decider(report, evaluation)
            capabilities = self.capability_fn()
            mask_valid = self.mask_valid_fn(report, current_candidate)
            guard = resolve_action(
                policy_action=getattr(decision, "action", "prompt_repair"),
                report=report,
                total_frames=self._total_frames(current_candidate),
                local_threshold=cfg.error_scope_threshold,
                capability_available=capabilities,
                mask_valid=mask_valid,
            )
            rec.scope = guard.scope
            rec.policy_action = guard.policy_action
            rec.final_action = guard.final_action
            self.hooks.on_decision(sample_id, current_candidate.candidate_id, decision, guard)
            self.hooks.on_state(sample_id, "DECISION_READY", round_index)

            final_action = RepairAction(guard.final_action)

            # --- 执行（Executor 一次）---
            exec_counter += 1
            execution_id = f"{sample_id}-exec{exec_counter}"
            rec.execution_id = execution_id
            self.hooks.on_state(sample_id, "EXECUTING", round_index)
            exec_decision = decision
            if final_action is RepairAction.LOCAL_EDITING:
                local_target = self.local_target_fn(report, current_candidate) or getattr(decision, "local_target", None)
                if local_target is None:
                    exec_result = ExecutionResult(
                        action=final_action,
                        status="failed",
                        backend_id="v2-runner-local-target-precheck",
                        failure_reason="local_target_missing_or_invalid",
                    )
                    rec.execution = exec_result.to_dict()
                    rec.state = "EXECUTOR_FAILED"
                    rec.terminal_reason = "local_target_missing_or_invalid"
                    result.rounds.append(rec)
                    self.hooks.on_execution(sample_id, exec_result)
                    self.hooks.on_state(sample_id, "EXECUTOR_FAILED", round_index)
                    result.stop_reason = "executor_failed"
                    result.final_state = "EXECUTOR_FAILED"
                    break
                exec_decision = _with_action(decision, final_action, local_target=local_target)
            else:
                exec_decision = _with_action(decision, final_action)
            exec_request = ExecutionRequest(
                decision=exec_decision,
                candidate=current_candidate,
                critic_report=report,
                prompt=current_prompt,
                physics_plan=physics_plan,
                seed=cfg.base_seed + round_index + 1000,
                history=tuple(history),
                metadata={"original_prompt": original_prompt, "execution_id": execution_id},
            )
            exec_result = self.executor_registry.execute(exec_request)
            rec.execution = exec_result.to_dict() if hasattr(exec_result, "to_dict") else {"status": "unknown"}
            self.hooks.on_execution(sample_id, exec_result)

            # --- Reject：不产生 after video，terminal ---
            if exec_result.status == "rejected" or getattr(exec_result, "terminal", False):
                rec.state = "REJECTED"
                rec.terminal_reason = "rejected_by_design"
                result.rounds.append(rec)
                self.hooks.on_state(sample_id, "REJECTED", round_index)
                chosen = self.selector.select(tuple(history)) if history else best_eval
                best_eval = chosen if chosen is not None else best_eval
                result.stop_reason = "rejected"
                result.final_state = "REJECTED"
                break

            # --- 执行失败 ---
            if exec_result.status == "failed" or exec_result.candidate is None:
                rec.state = "EXECUTOR_FAILED"
                rec.terminal_reason = "executor_failed"
                result.rounds.append(rec)
                self.hooks.on_state(sample_id, "EXECUTOR_FAILED", round_index)
                # 无有效 after：终止本样本（不静默继续用旧 candidate）。
                result.stop_reason = "executor_failed"
                result.final_state = "EXECUTOR_FAILED"
                break

            # --- P0-01 核心：after candidate 成为下一轮 current + 立即 Re-Critic/Re-Gate ---
            self.hooks.on_state(sample_id, "RE_EVALUATING", round_index)
            after_candidate = exec_result.candidate
            after_prompt = exec_result.next_prompt or current_prompt
            rec.after_candidate_id = str(getattr(after_candidate, "candidate_id", "?"))

            after_report = self.critic.evaluate(after_candidate, prompt=after_prompt, physics_plan=physics_plan)
            if after_report is not None:
                after_phys = self._physics(after_report)
                after_side = self.side_score_fn(after_candidate)
                after_gate = evaluate_gate(
                    report=after_report,
                    mode=cfg.acceptance_mode,
                    thresholds=cfg.thresholds,
                    semantic_score=after_side.get("semantic_score"),
                    quality_score=after_side.get("quality_score"),
                    critic_degraded=self._is_degraded(after_report),
                    fail_on_degraded=cfg.fail_on_degraded_critic,
                )
                rec.after_physics = after_phys
                rec.after_gate = after_gate.to_dict()
                rec.state = "RE_EVALUATED"
                after_eval = _Eval(after_candidate, after_report)
                history.append(after_eval)
                if after_phys > self._physics(best_eval.report):
                    best_eval = after_eval
                result.rounds.append(rec)
                # after 通过严格门 → 接受终止（force 模式除外，force 只跑一轮采集）。
                if after_gate.accepted and cfg.force_action is None:
                    self.hooks.on_state(sample_id, "ACCEPTED", round_index)
                    result.stop_reason = "accepted"
                    result.final_state = "ACCEPTED"
                    break
                # 否则 after 成为下一轮 current。
                current_candidate = after_candidate
                current_prompt = after_prompt
            else:
                rec.state = "CRITIC_FAILED"
                rec.terminal_reason = "after_critic_failed"
                result.rounds.append(rec)
                result.stop_reason = "critic_failed"
                result.final_state = "CRITIC_FAILED"
                break

            # force_action 模式：只跑一轮采集，采完即止。
            if cfg.force_action is not None:
                result.stop_reason = "force_action_collected"
                break

        if best_eval is not None:
            result.best_candidate_id = str(best_eval.candidate.candidate_id)
            result.best_physics_score = self._physics(best_eval.report)
        self.hooks.on_state(sample_id, "COMPLETED", len(result.rounds))
        if result.final_state not in {"CRITIC_FAILED"}:
            result.final_state = "COMPLETED" if result.stop_reason != "accepted" else "ACCEPTED"
        return result


@dataclass(frozen=True)
class _Eval:
    candidate: Any
    report: Any


def _with_action(decision: Any, action: RepairAction, *, local_target: Any | None = None) -> Any:
    """返回把 action 替换为 final_action 的 decision（若已一致则原样返回）。

    使用 dataclasses.replace，保持其它字段与 provenance 不变。
    """

    current_target = getattr(decision, "local_target", None)
    target = local_target if local_target is not None else current_target
    if getattr(decision, "action", None) == action and target is current_target:
        return decision
    try:
        from dataclasses import replace

        if action is RepairAction.LOCAL_EDITING:
            return replace(decision, action=action, local_target=target)
        return replace(decision, action=action)
    except Exception:  # noqa: BLE001
        return decision


class RunnerHooks:
    """审计回调基类；默认 no-op，可由 runner 入口注入写 artifacts 的实现。"""

    def on_state(self, sample_id: str, state: str, round_index: int) -> None:  # noqa: D401
        return None

    def on_decision(self, sample_id: str, candidate_id: str, decision: Any, guard: GuardResult) -> None:
        return None

    def on_execution(self, sample_id: str, exec_result: Any) -> None:
        return None
