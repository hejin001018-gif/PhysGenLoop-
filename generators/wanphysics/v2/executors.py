"""V2 local-edit and reject executors.

修复 P0-4 / P0-5：

- 现有 ``src/physgenloop/learning_repair/executors.py`` 的 ``PromptRepairExecutor`` 会在
  ``execute`` 内再调用 ``prompt_rewriter.repair()``，若该 rewriter 是
  ``ActionValueRepairer`` 就会**二次运行 Policy**并递增 attempt 状态；V2 的
  :class:`DecisionPromptRepairExecutor` 只消费已决策好的 ``RepairDecision`` 和确定性的
  prompt 渲染，**绝不再调用 Policy**（``policy_call_count`` 恒为 0）。
- Reject / Global / Local 均产生真实执行记录，不再由 Controller 直接 return。

这些 executor 复用（import，不修改）canonical 契约
``ExecutionRequest`` / ``ExecutionResult`` / ``RepairAction`` / ``LocalEditTarget``，
并实现与旧 ``ExecutorRegistry`` 相同的 ``action`` / ``backend_id`` / ``execute`` 协议，
因此可直接注入现有 ``ExecutorRegistry``（旧 executors 保持不动，二者并存）。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import ExecutionRequest, ExecutionResult

from .prompt_renderer import render_repair_prompt


def _cost(provider: Callable[[ExecutionRequest], float] | None, request: ExecutionRequest) -> float:
    return 0.0 if provider is None else float(provider(request))


class DecisionPromptRepairExecutor:
    """用已决策的 instruction + 确定性 prompt 渲染重新生成，绝不二次调用 Policy。"""

    action = RepairAction.PROMPT_REPAIR

    def __init__(
        self,
        *,
        generator: Any,
        backend_id: str = "v2-decision-prompt+generator",
        max_prompt_chars: int = 600,
        cost_provider: Callable[[ExecutionRequest], float] | None = None,
    ) -> None:
        self.generator = generator
        self.backend_id = backend_id
        self.max_prompt_chars = max_prompt_chars
        self.cost_provider = cost_provider
        # 审计：本 executor 从不调用 policy。
        self.policy_call_count = 0

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        started = time.perf_counter()
        violations = tuple(getattr(request.critic_report, "violations", ()) or ())
        rendered = render_repair_prompt(
            original_prompt=request.prompt,
            violations=violations,
            max_chars=self.max_prompt_chars,
        )
        candidate = self.generator.generate(
            prompt=rendered.prompt,
            seed=request.seed,
        )
        return ExecutionResult(
            action=self.action,
            status="succeeded",
            backend_id=self.backend_id,
            candidate=candidate,
            next_prompt=rendered.prompt,
            cost=_cost(self.cost_provider, request),
            latency_seconds=time.perf_counter() - started,
            artifacts={"repaired_video": str(candidate.video_path)},
            metadata={
                "instruction_sha256": rendered.instruction_sha256,
                "instruction_source": rendered.source,
                "policy_call_count": self.policy_call_count,
                "decision_action": request.decision.action.value,
            },
        )


class OriginalPromptGlobalRegenerationExecutor:
    """回到 immutable 原始 prompt 重新生成整段视频。"""

    action = "global_regeneration"

    def __init__(
        self,
        *,
        generator: Any,
        backend_id: str = "v2-global-regeneration",
        cost_provider: Callable[[ExecutionRequest], float] | None = None,
    ) -> None:
        self.generator = generator
        self.backend_id = backend_id
        self.cost_provider = cost_provider

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        started = time.perf_counter()
        # 原始 prompt 从 metadata 显式取，避免多轮后 current_prompt 已被改写。
        original_prompt = str(request.metadata.get("original_prompt", request.prompt))
        candidate = self.generator.generate(
            prompt=original_prompt,
            seed=request.seed,
        )
        return ExecutionResult(
            action=self.action,
            status="succeeded",
            backend_id=self.backend_id,
            candidate=candidate,
            next_prompt=original_prompt,
            cost=_cost(self.cost_provider, request),
            latency_seconds=time.perf_counter() - started,
            artifacts={"repaired_video": str(candidate.video_path)},
            metadata={
                "input_prompt": request.prompt,
                "original_prompt": original_prompt,
            },
        )


class MaskSequenceLocalEditingExecutor:
    """逐帧 mask 局部编辑；要求有效 LocalEditTarget，否则显式失败。"""

    action = RepairAction.LOCAL_EDITING

    def __init__(
        self,
        *,
        editor: Any,
        backend_id: str = "v2-mask-sequence-local-editor",
        cost_provider: Callable[[ExecutionRequest], float] | None = None,
    ) -> None:
        self.editor = editor
        self.backend_id = backend_id
        self.cost_provider = cost_provider

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        target = request.decision.local_target
        if target is None:
            raise ValueError("local editing requires a LocalEditTarget")
        if target.mask_uri is None or not target.critical_frames:
            raise ValueError("local editing requires mask_uri and critical_frames")
        started = time.perf_counter()
        if hasattr(self.editor, "edit"):
            candidate = self.editor.edit(
                candidate=request.candidate,
                target=target,
                instruction=request.decision.instruction,
                critic_report=request.critic_report,
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
                "source_video": str(request.candidate.video_path),
                "mask_manifest": str(target.mask_uri),
            },
            metadata={
                "executor": "MaskSequenceLocalEditingExecutor",
                "editor": "StrictProPainterLocalEditor",
                "editor_backend": "ProPainter",
                "repair_mode": "strict-mask-video-inpainting",
                "local_target": target.to_dict(),
                "critical_frames": list(target.critical_frames),
                "propainter": dict(candidate.metadata.get("propainter", {})),
                "output_validation": dict(
                    candidate.metadata.get("output_validation", {})
                ),
            },
        )


class AuditedRejectExecutor:
    """终止修复并从历史候选中选最终返回；产生 terminal 执行记录。"""

    action = RepairAction.REJECT

    def __init__(self, *, selector: Any, backend_id: str = "v2-audited-reject") -> None:
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
            metadata={
                "selected_candidate_id": str(candidate.candidate_id),
                "history_size": len(request.history),
            },
        )
