"""Action-specific executors added beside the existing Generator interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Protocol

from physgenloop.learning_repair.contracts import ACTION_ORDER, RepairAction, RepairContext

from .contracts import ExecutionRequest, ExecutionResult


class RepairExecutor(Protocol):
    action: RepairAction
    backend_id: str

    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


def _cost(cost_provider: Callable[[ExecutionRequest], float] | None, request: ExecutionRequest) -> float:
    return 0.0 if cost_provider is None else float(cost_provider(request))


class PromptRepairExecutor:
    """Rewrite a prompt, then generate a new full candidate from that prompt."""

    action = RepairAction.PROMPT_REPAIR

    def __init__(
        self,
        *,
        prompt_rewriter: Any,
        generator: Any,
        backend_id: str = "prompt-rewriter+video-generator",
        cost_provider: Callable[[ExecutionRequest], float] | None = None,
    ) -> None:
        self.prompt_rewriter = prompt_rewriter
        self.generator = generator
        self.backend_id = backend_id
        self.cost_provider = cost_provider

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        started = time.perf_counter()
        if hasattr(self.prompt_rewriter, "repair"):
            rewritten = self.prompt_rewriter.repair(
                prompt=request.prompt,
                report=request.critic_report,
            )
        elif callable(self.prompt_rewriter):
            rewritten = self.prompt_rewriter(
                prompt=request.prompt,
                report=request.critic_report,
                decision=request.decision,
            )
        else:
            raise TypeError("prompt_rewriter must be callable or expose repair()")
        rewritten = str(rewritten).strip()
        if not rewritten:
            raise ValueError("prompt rewriter returned an empty prompt")
        candidate = self.generator.generate(
            prompt=rewritten,
            physics_plan=request.physics_plan,
            seed=request.seed,
        )
        return ExecutionResult(
            action=self.action,
            status="succeeded",
            backend_id=self.backend_id,
            candidate=candidate,
            next_prompt=rewritten,
            cost=_cost(self.cost_provider, request),
            latency_seconds=time.perf_counter() - started,
            artifacts={"repaired_video": str(candidate.video_path)},
        )


class GlobalRegenerationExecutor:
    """Regenerate a complete candidate without changing the original prompt."""

    action = RepairAction.GLOBAL_REGENERATION

    def __init__(
        self,
        *,
        generator: Any,
        backend_id: str = "video-generator",
        cost_provider: Callable[[ExecutionRequest], float] | None = None,
    ) -> None:
        self.generator = generator
        self.backend_id = backend_id
        self.cost_provider = cost_provider

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        started = time.perf_counter()
        candidate = self.generator.generate(
            prompt=request.prompt,
            physics_plan=request.physics_plan,
            seed=request.seed,
        )
        return ExecutionResult(
            action=self.action,
            status="succeeded",
            backend_id=self.backend_id,
            candidate=candidate,
            next_prompt=request.prompt,
            cost=_cost(self.cost_provider, request),
            latency_seconds=time.perf_counter() - started,
            artifacts={"repaired_video": str(candidate.video_path)},
        )


class LocalEditingExecutor:
    """Wrap an editor while keeping its object/time/mask inputs outside Critic."""

    action = RepairAction.LOCAL_EDITING

    def __init__(
        self,
        *,
        editor: Any,
        backend_id: str = "local-video-editor",
        cost_provider: Callable[[ExecutionRequest], float] | None = None,
    ) -> None:
        self.editor = editor
        self.backend_id = backend_id
        self.cost_provider = cost_provider

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if request.decision.local_target is None:
            raise ValueError("local editing requires a LocalEditTarget")
        started = time.perf_counter()
        if hasattr(self.editor, "edit"):
            candidate = self.editor.edit(
                candidate=request.candidate,
                target=request.decision.local_target,
                instruction=request.decision.instruction,
                critic_report=request.critic_report,
                physics_plan=request.physics_plan,
                seed=request.seed,
            )
        elif callable(self.editor):
            candidate = self.editor(request)
        else:
            raise TypeError("editor must be callable or expose edit()")
        return ExecutionResult(
            action=self.action,
            status="succeeded",
            backend_id=self.backend_id,
            candidate=candidate,
            next_prompt=request.prompt,
            cost=_cost(self.cost_provider, request),
            latency_seconds=time.perf_counter() - started,
            artifacts={
                "repaired_video": str(candidate.video_path),
                **(
                    {}
                    if request.decision.local_target.mask_uri is None
                    else {"mask": request.decision.local_target.mask_uri}
                ),
            },
            metadata={"local_target": request.decision.local_target.to_dict()},
        )


class RejectExecutor:
    """Terminate repair and delegate final choice to the existing selector."""

    action = RepairAction.REJECT

    def __init__(
        self,
        *,
        selector: Any,
        backend_id: str = "candidate-selector",
    ) -> None:
        self.selector = selector
        self.backend_id = backend_id

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        started = time.perf_counter()
        if not request.history:
            raise ValueError("reject execution requires candidate evaluation history")
        selected = self.selector.select(tuple(request.history))
        candidate = getattr(selected, "candidate", selected)
        return ExecutionResult(
            action=self.action,
            status="rejected",
            backend_id=self.backend_id,
            candidate=candidate,
            next_prompt=request.prompt,
            latency_seconds=time.perf_counter() - started,
            terminal=True,
            failure_reason=None,
            metadata={"selected_candidate_id": str(candidate.candidate_id)},
        )


@dataclass(frozen=True)
class ExecutorCapability:
    action: RepairAction
    backend_id: str

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action.value, "backend_id": self.backend_id}


class ExecutorRegistry:
    """One executor per action plus the capability mask consumed by the Policy."""

    def __init__(self, executors: Iterable[RepairExecutor]) -> None:
        by_action: dict[RepairAction, RepairExecutor] = {}
        for executor in executors:
            action = RepairAction(executor.action)
            if action in by_action:
                raise ValueError(f"duplicate executor for {action.value}")
            by_action[action] = executor
        if RepairAction.REJECT not in by_action:
            raise ValueError("ExecutorRegistry requires a reject executor")
        self._executors = by_action

    def supports(self, action: RepairAction) -> bool:
        return RepairAction(action) in self._executors

    @property
    def actions(self) -> tuple[RepairAction, ...]:
        return tuple(action for action in ACTION_ORDER if action in self._executors)

    def context(
        self,
        *,
        attempt_index: int,
        max_attempts: int,
        previous_actions: tuple[RepairAction, ...] = (),
        semantic_score: float | None = None,
        quality_score: float | None = None,
    ) -> RepairContext:
        return RepairContext(
            attempt_index=attempt_index,
            max_attempts=max_attempts,
            prompt_repair_available=self.supports(RepairAction.PROMPT_REPAIR),
            global_regeneration_available=self.supports(RepairAction.GLOBAL_REGENERATION),
            local_editor_available=self.supports(RepairAction.LOCAL_EDITING),
            semantic_score=semantic_score,
            quality_score=quality_score,
            previous_actions=previous_actions,
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        action = request.decision.action
        executor = self._executors.get(action)
        if executor is None:
            raise RuntimeError(
                f"policy selected unavailable action {action.value}; capability masking failed"
            )
        try:
            return executor.execute(request)
        except Exception as exc:
            return ExecutionResult(
                action=action,
                status="failed",
                backend_id=str(getattr(executor, "backend_id", "unknown")),
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

    def capability_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "repair-executor-capabilities/1.0",
            "action_order": [item.value for item in ACTION_ORDER],
            "capabilities": [
                ExecutorCapability(action, str(self._executors[action].backend_id)).to_dict()
                for action in self.actions
            ],
            "masked_actions": [
                action.value for action in ACTION_ORDER if action not in self._executors
            ],
        }
