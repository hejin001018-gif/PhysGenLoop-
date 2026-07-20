"""构建注入了真实后端的 ExecutorRegistry。

把四个 Executor 与项目中已有的真实实现连接起来：
  PROMPT_REPAIR      → ActionValueRepairer + WanSubprocessGenerator
  GLOBAL_REGENERATION → WanSubprocessGenerator
  LOCAL_EDITING      → ProPainterLocalEditor
  REJECT             → RejectExecutor（需要注入 selector）

使用方：
    from generators.wanphysics.executor_factory import build_executor_registry
    from physgenloop.selector import EvidenceAwareSelector
    registry = build_executor_registry(run_dir="/root/PhysGenLoop-/outputs/run_xxx")
    # LoopController 直接消费 ExecutorRegistry，无需 LearningRepairLoopRunner
"""
from __future__ import annotations

from pathlib import Path

from physgenloop.learning_repair import (
    ExecutorRegistry,
    GlobalRegenerationExecutor,
    LocalEditingExecutor,
    PromptRepairExecutor,
    RejectExecutor,
)
from physgenloop.selector import EvidenceAwareSelector

from .adapter import WanSubprocessGenerator
from .local_editor import ProPainterLocalEditor
from .repairer import load_action_value_repairer

ROOT = Path("/root/PhysGenLoop-")
CKPT_ROOT = ROOT / "checkpoints/repair_agent/repair-agent-v3.1-proxy-20260717"
PY_MAIN = str(ROOT / "envs/main/bin/python")


def build_executor_registry(
    run_dir: str,
    ckpt_root: str = str(CKPT_ROOT),
    python: str = PY_MAIN,
    propainter_repo: str = "/root/ProPainter",
    max_attempts: int = 2,
) -> ExecutorRegistry:
    """返回注入了真实后端的 ExecutorRegistry。

    run_dir 作为本次运行的视频输出目录，传给 WanSubprocessGenerator
    和 ProPainterLocalEditor，确保输出文件落在带时间戳的独立子目录下。
    """
    generator = WanSubprocessGenerator(
        python=python,
        model_path=str(ROOT / "models/wan2.2_ti2v_5b"),
        output_root=run_dir,
    )
    repairer = load_action_value_repairer(
        ckpt_root,
        max_attempts=max_attempts,
        local_editor_available=True,
    )
    editor = ProPainterLocalEditor(
        propainter_repo=propainter_repo,
        python=python,
        output_root=run_dir,
    )
    selector = EvidenceAwareSelector()

    return ExecutorRegistry(
        executors=[
            PromptRepairExecutor(
                prompt_rewriter=repairer,
                generator=generator,
                backend_id="action-value-repairer+wan2.2",
            ),
            GlobalRegenerationExecutor(
                generator=generator,
                backend_id="wan2.2-global-regen",
            ),
            LocalEditingExecutor(
                editor=editor,
                backend_id="propainter-local-edit",
            ),
            RejectExecutor(selector=selector),
        ]
    )
