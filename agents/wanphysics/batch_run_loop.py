"""批量 prompt 闭环入口，对接 htz 的批量生成流水线。

用法：
  python batch_run_loop.py --prompts-dir ./prompts [--config configs/loop.yaml]

prompts-dir 下每个 .txt 文件对应一条 prompt，文件名（去掉 .txt）作为 task ID。
每条 prompt 独立跑一次完整的 LoopController 闭环，输出写入：
  outputs/run_YYYYMMDD_HHMMSS/{task_id}/
    ├── {candidate_id}-v01.mp4
    ├── prompt.txt
    ├── metadata.json
    ├── critic.json        ← 生成时 waiting，Critic 完成后回填 completed
    └── loop_result.json   ← 闭环摘要
    │   ├── prompt.txt
    │   └── metadata.json
    ├── critic.json        ← Critic 最终报告（兼容 htz 预留的接口）
    └── loop_result.json   ← 闭环摘要

所有 prompt 共享同一个 vLLM 实例（run 结束后统一关闭），避免重复加载。

配置优先级（高→低）：CLI 参数 > configs/loop.yaml > 代码默认值
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
from generators.wanphysics.repairer import load_action_value_repairer
from generators.wanphysics.sam2_vlm_critic import Sam2VlmSubprocessCritic

_DEFAULT_CONFIG = Path("/root/PhysGenLoop-/configs/loop.yaml")


def _load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one(
    prompt: str,
    prompt_id: str,
    prompt_run_dir: Path,
    critic: Sam2VlmSubprocessCritic,
    cfg: dict,
    ckpt_root: str,
    max_rounds: int,
) -> dict:
    paths = cfg["paths"]
    gen = cfg["generator"]
    loop = cfg["loop"]

    generator = WanSubprocessGenerator(
        python=paths["envs"]["main"],
        model_path=paths["models"]["wan"],
        output_root=str(prompt_run_dir),
        num_frames=gen["num_frames"],
        height=gen["height"],
        width=gen["width"],
        fps=gen["fps"],
        negative_prompt=gen.get("negative_prompt"),
    )
    repairer = load_action_value_repairer(ckpt_root, max_attempts=max_rounds)
    selector = EvidenceAwareSelector()
    config = LoopConfig(
        max_rounds=max_rounds,
        candidates_per_round=loop["candidates_per_round"],
        acceptance_score=loop["acceptance_score"],
        base_seed=loop["base_seed"],
    )
    controller = LoopController(
        generator=generator,
        critic=critic,
        repairer=repairer,
        selector=selector,
        config=config,
    )
    result = controller.run(prompt=prompt)

    # N1：写 trials.jsonl，记录每轮 before/after physics_score 和 detector_backend
    trials_path = prompt_run_dir / "trials.jsonl"
    with trials_path.open("w", encoding="utf-8") as tf:
        for i, rnd in enumerate(result.history):
            before_eval = rnd.evaluations[0] if rnd.evaluations else None
            after_eval = result.history[i + 1].evaluations[0] if i + 1 < len(result.history) else None
            trial = {
                "prompt_id": prompt_id,
                "round_index": rnd.round_index,
                "before_physics_score": before_eval.report.physics_score if before_eval else None,
                "before_decision": before_eval.report.decision if before_eval else None,
                "before_detector_backend": (before_eval.report.diagnostics.get("detector_backend") if before_eval else None),
                "after_physics_score": after_eval.report.physics_score if after_eval else None,
                "stop_reason": result.stop_reason if i == len(result.history) - 1 else "continued",
            }
            tf.write(json.dumps(trial, ensure_ascii=False) + "\n")

    detector_backend = result.best.report.diagnostics.get("detector_backend", "unknown")

    summary = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "stop_reason": result.stop_reason,
        "best_candidate_id": result.best.candidate.candidate_id,
        "best_video_path": result.best.candidate.video_path,
        "best_physics_score": result.best.report.physics_score,
        "best_decision": result.best.report.decision,
        "rounds": len(result.history),
        "detector_backend": detector_backend,
    }
    (prompt_run_dir / "loop_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="批量 prompt 闭环")
    parser.add_argument("--prompts-dir", required=True, help="包含 .txt prompt 文件的目录")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG), help="loop.yaml 路径")
    parser.add_argument("--max-rounds", type=int, default=None, help="覆盖 yaml 中的 max_rounds")
    parser.add_argument("--ckpt-root", default=None, help="覆盖 repair agent checkpoint 路径")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    paths = cfg["paths"]
    loop = cfg["loop"]
    vllm = cfg["vllm"]

    max_rounds = args.max_rounds if args.max_rounds is not None else loop["max_rounds"]
    ckpt_root = args.ckpt_root or paths["checkpoints"]["repair_agent"]

    prompts_dir = Path(args.prompts_dir)
    prompt_files = sorted(prompts_dir.glob("*.txt"))
    if not prompt_files:
        print(f"[batch] 未找到 .txt 文件：{prompts_dir}", file=sys.stderr)
        return 1

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(paths["outputs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[batch] run_id={run_id}  prompts={len(prompt_files)}", flush=True)

    # vllm.log 放在 run 根目录，各 task 输出放在 outputs/{task_id}/
    outputs_root = Path(paths["outputs"])

    critic = Sam2VlmSubprocessCritic(
        python=paths["envs"]["main"],
        vllm_python=paths["envs"]["vllm"],
        vllm_model=paths["models"]["vllm"],
        vllm_served_name=vllm["served_name"],
        vllm_log=str(run_dir / "vllm.log"),
        vllm_gpu_util=vllm["gpu_util"],
        vllm_max_model_len=vllm["max_model_len"],
    )

    results = []
    try:
        for pf in prompt_files:
            prompt_id = pf.stem
            prompt = pf.read_text(encoding="utf-8").strip()
            if not prompt:
                print(f"[batch] 跳过空文件：{pf.name}", file=sys.stderr)
                continue
            # 输出目录 outputs/{task_id}/，与 htz 原始约定一致
            prompt_run_dir = outputs_root / prompt_id
            prompt_run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[batch] [{prompt_id}] 开始", flush=True)
            try:
                summary = run_one(
                    prompt=prompt,
                    prompt_id=prompt_id,
                    prompt_run_dir=prompt_run_dir,
                    critic=critic,
                    cfg=cfg,
                    ckpt_root=ckpt_root,
                    max_rounds=max_rounds,
                )
                results.append(summary)
                print(
                    f"[batch] [{prompt_id}] 完成  "
                    f"stop={summary['stop_reason']}  "
                    f"score={summary['best_physics_score']}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[batch] [{prompt_id}] 失败：{exc}", file=sys.stderr)
                results.append({"prompt_id": prompt_id, "error": str(exc)})
    finally:
        critic.shutdown()

    batch_result = {"run_id": run_id, "results": results}
    (run_dir / "batch_result.json").write_text(
        json.dumps(batch_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(batch_result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
