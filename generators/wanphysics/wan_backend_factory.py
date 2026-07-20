"""Wan 后端的 CloudBackendBundle factory。

实现 load_backend_factory 规范的 module:function 接口：
  generators.wanphysics.wan_backend_factory:build_wan_backend

提供 ActualTrialCampaign 所需的所有依赖：
  - critic       : Sam2VlmSubprocessCritic
  - executors    : ExecutorRegistry (build_executor_registry)
  - source_loader: 从 CampaignItem.prompt_path 读取 prompt 文本
  - physics_plan_provider: 返回默认 PhysicsPlan（后续可扩展注入 Planner）
  - semantic_scorer: no-op（require_quality_metrics=False 时不参与门槛）
  - quality_scorer : no-op

调用方式：
  从 run_trial_campaign.py 或 ActualTrialCampaign 传入
  backend_factory="generators.wanphysics.wan_backend_factory:build_wan_backend"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pavg_critic.schemas import PhysicsPlan
from physgenloop.learning_repair.campaign import CloudBackendBundle

from .executor_factory import build_executor_registry
from .sam2_vlm_critic import Sam2VlmSubprocessCritic

_ROOT = Path("/root/PhysGenLoop-")
_PY_MAIN = str(_ROOT / "envs/main/bin/python")
_PY_VLLM = str(_ROOT / "envs/vllm-cu128/bin/python")


def _noop_scorer(candidate: Any, report: Any, prompt: str) -> float:
    """No-op metric scorer，require_quality_metrics=False 时占位用。"""
    return 1.0


def build_wan_backend(
    *,
    run_dir: str,
    ckpt_root: str = str(_ROOT / "checkpoints/repair_agent/repair-agent-v3.1-proxy-20260717"),
    python: str = _PY_MAIN,
    vllm_python: str = _PY_VLLM,
    vllm_model: str = str(_ROOT / "models/Qwen3-VL-8B-Instruct"),
    vllm_served_name: str = "qwen3-vl-8b-instruct",
    vllm_log: str | None = None,
    vllm_gpu_util: float = 0.85,
    vllm_max_model_len: int = 16384,
    propainter_repo: str = "/root/ProPainter",
) -> CloudBackendBundle:
    """构造并返回注入了真实 Wan/SAM2 后端的 CloudBackendBundle。

    run_dir 用于隔离本次 campaign 的输出文件。
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    critic = Sam2VlmSubprocessCritic(
        python=python,
        vllm_python=vllm_python,
        vllm_model=vllm_model,
        vllm_served_name=vllm_served_name,
        vllm_log=vllm_log or str(run_path / "vllm_campaign.log"),
        vllm_gpu_util=vllm_gpu_util,
        vllm_max_model_len=vllm_max_model_len,
    )
    executors = build_executor_registry(
        run_dir=run_dir,
        ckpt_root=ckpt_root,
        python=python,
        propainter_repo=propainter_repo,
    )

    def source_loader(item: Any) -> Any:
        """从 CampaignItem 的 prompt 字段或 prompt_path 加载 prompt 文本。"""
        if hasattr(item, "prompt") and item.prompt:
            return item.prompt
        if hasattr(item, "prompt_path") and item.prompt_path:
            return Path(item.prompt_path).read_text(encoding="utf-8").strip()
        raise ValueError(f"CampaignItem has no prompt or prompt_path: {item}")

    def physics_plan_provider(item: Any) -> PhysicsPlan:
        """默认返回空 PhysicsPlan，后续可在此注入 TemplatePhysicsPlanner。"""
        return PhysicsPlan()

    return CloudBackendBundle(
        critic=critic,
        executors=executors,
        source_loader=source_loader,
        physics_plan_provider=physics_plan_provider,
        semantic_scorer=_noop_scorer,
        quality_scorer=_noop_scorer,
    )
