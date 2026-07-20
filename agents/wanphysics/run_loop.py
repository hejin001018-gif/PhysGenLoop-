"""PhysGenLoop 完整闭环入口（子进程 GPU 交接版）。

使用正式的 LoopController -> EvidenceAwareSelector -> ActionValueRepairer 链路，
通过 WanSubprocessGenerator 和 Sam2VlmSubprocessCritic 实现顺序化 GPU 交接：
  - 生成阶段：WanSubprocessGenerator 启动一次性子进程，Wan2.2 退出即释放显存
  - 评估阶段：Sam2VlmSubprocessCritic 拉起 vLLM，评估完毕后：
      P0-7：若还有下一轮生成，先 stop_vllm() 再交还 GPU，生成完毕后 start_vllm()
  - 循环结束后：Sam2VlmSubprocessCritic.shutdown() 自动停止 vLLM

B1：注入 ExecutorRegistry，LoopController 可原生执行四动作：
      REJECT -> stop_reason="rejected"
      LOCAL_EDITING -> 真实调用 ProPainterLocalEditor
      GLOBAL_REGENERATION -> 重置为原始 prompt
      PROMPT_REPAIR -> 原有 prompt 修正行为

P1-3：semantic_scorer / quality_scorer 接口预留，当前注入 None（
      LoopController 的 require_quality_metrics 默认 False，行为不变）。

每次运行的所有输出写入带时间戳的独立子目录：outputs/run_YYYYMMDD_HHMMSS/

配置优先级（高->低）：CLI 参数 > configs/loop.yaml > 代码默认值
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, "/root/PhysGenLoop-")
sys.path.insert(0, "/root/PhysGenLoop-/src")

from dotenv import load_dotenv

load_dotenv("/root/PhysGenLoop-/.env")

from physgenloop.controller import LoopController
from physgenloop.contracts import LoopConfig
from physgenloop.selector import EvidenceAwareSelector

from generators.wanphysics.adapter import WanSubprocessGenerator
from generators.wanphysics.executor_factory import build_executor_registry
from generators.wanphysics.repairer import load_action_value_repairer
from generators.wanphysics.sam2_vlm_critic import Sam2VlmSubprocessCritic

_DEFAULT_CONFIG = Path("/root/PhysGenLoop-/configs/loop.yaml")


def _load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _VllmHandoffCritic:
    """P0-7：在每轮 evaluate() 前后管理 vLLM 生命周期。

    第一轮 evaluate() 时 start_vllm()（首次调用自动触发）；
    evaluate() 完成后如果还有下一轮生成，调用 stop_vllm() 先释放 GPU，
    生成完毕、进入下一轮 evaluate() 前再 start_vllm()。

    实际的 start/stop 逻辑委托给 Sam2VlmSubprocessCritic，
    本类只负责在 LoopController 调用边界插入 stop 时机。
    """

    def __init__(self, inner: Sam2VlmSubprocessCritic, max_rounds: int) -> None:
        self._inner = inner
        self._max_rounds = max_rounds
        self._round_count = 0

    def prepare_for_generation(self) -> None:
        self._inner.prepare_for_generation()

    def evaluate(self, candidate, *, prompt, physics_plan):
        # evaluate() 内部会在需要时 start_vllm（Sam2VlmSubprocessCritic.start_vllm 是幂等的）
        report = self._inner.evaluate(candidate, prompt=prompt, physics_plan=physics_plan)
        self._round_count += 1
        # 评估完毕，若还需要下一轮生成，提前释放 GPU
        if self._round_count < self._max_rounds:
            self._inner.stop_vllm()
        return report

    def shutdown(self) -> None:
        self._inner.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="PhysGenLoop 单 prompt 闭环")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG), help="loop.yaml 路径")
    parser.add_argument("--prompt", default=None, help="覆盖 yaml 中的 prompt")
    parser.add_argument("--max-rounds", type=int, default=None, help="覆盖 yaml 中的 max_rounds")
    parser.add_argument("--candidates-per-round", type=int, default=None)
    parser.add_argument("--ckpt-root", default=None, help="覆盖 repair agent checkpoint 路径")
    parser.add_argument("--no-executor-registry", action="store_true",
                        help="禁用 ExecutorRegistry，退化为纯 PromptRepairer 模式（调试用）")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    paths = cfg["paths"]
    loop = cfg["loop"]
    gen = cfg["generator"]
    vllm = cfg["vllm"]

    prompt = args.prompt or loop["prompt"]
    max_rounds = args.max_rounds if args.max_rounds is not None else loop["max_rounds"]
    candidates_per_round = (
        args.candidates_per_round
        if args.candidates_per_round is not None
        else loop["candidates_per_round"]
    )
    ckpt_root = args.ckpt_root or paths["checkpoints"]["repair_agent"]

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(paths["outputs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_loop] output dir: {run_dir}", flush=True)

    generator = WanSubprocessGenerator(
        python=paths["envs"]["main"],
        model_path=paths["models"]["wan"],
        output_root=str(run_dir),
        num_frames=gen["num_frames"],
        height=gen["height"],
        width=gen["width"],
        fps=gen["fps"],
        negative_prompt=gen.get("negative_prompt"),
    )
    _raw_critic = Sam2VlmSubprocessCritic(
        python=paths["envs"]["main"],
        vllm_python=paths["envs"]["vllm"],
        vllm_model=paths["models"]["vllm"],
        vllm_served_name=vllm["served_name"],
        vllm_log=str(run_dir / "vllm.log"),
        vllm_gpu_util=vllm["gpu_util"],
        vllm_max_model_len=vllm["max_model_len"],
    )
    # P0-7：用 _VllmHandoffCritic 包装，在轮间自动交还 GPU
    critic = _VllmHandoffCritic(_raw_critic, max_rounds=max_rounds * candidates_per_round)

    repairer = load_action_value_repairer(
        ckpt_root,
        max_attempts=max_rounds,
        local_editor_available=not args.no_executor_registry,
    )
    selector = EvidenceAwareSelector()
    config = LoopConfig(
        max_rounds=max_rounds,
        candidates_per_round=candidates_per_round,
        acceptance_score=loop["acceptance_score"],
        base_seed=loop["base_seed"],
        error_scope_threshold=loop.get("error_scope_threshold", 0.4),
        default_total_frames=gen["num_frames"],
    )

    # B1：构建 ExecutorRegistry 并注入 LoopController
    # --no-executor-registry 时退化为 PromptRepairer 模式（调试/向后兼容）
    executor_registry = None
    if not args.no_executor_registry:
        try:
            executor_registry = build_executor_registry(
                run_dir=str(run_dir),
                ckpt_root=ckpt_root,
                python=paths["envs"]["main"],
                propainter_repo=paths.get("propainter_repo", "/root/ProPainter"),
            )
            print("[run_loop] ExecutorRegistry 已就绪（四动作模式）", flush=True)
        except Exception as exc:
            print(f"[run_loop] WARNING: ExecutorRegistry 构建失败 ({exc})，退化为 PromptRepairer 模式", file=sys.stderr)

    controller = LoopController(
        generator=generator,
        critic=critic,
        repairer=repairer,
        selector=selector,
        config=config,
        executor_registry=executor_registry,
    )

    try:
        result = controller.run(prompt=prompt)
    finally:
        critic.shutdown()

    # N1/P0-7：从 history 提取每轮 Trial 信息写 trials.jsonl
    # 记录 before/after physics_score、decision、detector_backend，供后续训练数据使用
    trials_path = run_dir / "trials.jsonl"
    with trials_path.open("w", encoding="utf-8") as tf:
        for i, rnd in enumerate(result.history):
            before_eval = rnd.evaluations[0] if rnd.evaluations else None
            after_eval = result.history[i + 1].evaluations[0] if i + 1 < len(result.history) else None
            trial = {
                "round_index": rnd.round_index,
                "prompt": rnd.prompt,
                "before_candidate_id": before_eval.candidate.candidate_id if before_eval else None,
                "before_physics_score": before_eval.report.physics_score if before_eval else None,
                "before_decision": before_eval.report.decision if before_eval else None,
                "before_detector_backend": (before_eval.report.diagnostics.get("detector_backend") if before_eval else None),
                "after_candidate_id": after_eval.candidate.candidate_id if after_eval else None,
                "after_physics_score": after_eval.report.physics_score if after_eval else None,
                "stop_reason": result.stop_reason if i == len(result.history) - 1 else "continued",
            }
            tf.write(json.dumps(trial, ensure_ascii=False) + "\n")

    # N3/P1-5：detector_backend 从 best report.diagnostics 取出写入 summary
    detector_backend = result.best.report.diagnostics.get("detector_backend", "unknown")

    summary = {
        "run_id": run_id,
        "stop_reason": result.stop_reason,
        "best_candidate_id": result.best.candidate.candidate_id,
        "best_video_path": result.best.candidate.video_path,
        "best_physics_score": result.best.report.physics_score,
        "best_decision": result.best.report.decision,
        "rounds": len(result.history),
        "detector_backend": detector_backend,
        # P1-3 预留：semantic_score / quality_score 待 scorer 接入后填充
        "semantic_score": None,
        "quality_score": None,
    }
    result_path = run_dir / "loop_result.json"
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
