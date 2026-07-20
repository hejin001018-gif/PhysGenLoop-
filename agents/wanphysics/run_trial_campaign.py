"""真实 Trial 采集入口（单 prompt 版）。

与 run_loop.py 并列，使用 LearningRepairLoopRunner 而非 LoopController。
完整实现 hj 构想中的 Trial 采集链路：

  prompt -> WanSubprocessGenerator
         -> Sam2VlmSubprocessCritic（固定，不修改）
         -> ActionValueDecisionPolicy（proxy 模式）
         -> ExecutorRegistry（四动作真实执行）
         -> Critic 复评
         -> RepairTrialV1 落盘（JsonlTrialRecorder）
         -> trial_result.json

输出目录：outputs/trial_YYYYMMDD_HHMMSS/
  - trials.jsonl        RepairTrialV1 记录（before/action/after/gain）
  - trial_result.json   run 级摘要
  - vllm_trial.log      vLLM 服务日志

用法：
  python agents/wanphysics/run_trial_campaign.py     --prompt "a red ball rolling on a flat table"     [--config configs/loop.yaml]     [--ckpt-root ...]     [--max-attempts 3]     [--group-id my_group]
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

from pavg_critic.schemas import PhysicsPlan
from physgenloop.learning_repair import (
    ActionValueDecisionPolicy,
    CompatibilityManifest,
    JsonlTrialRecorder,
    LearningRepairLoopRunner,
    RunnerConfig,
    TorchActionValuePolicy,
)
from physgenloop.selector import EvidenceAwareSelector

from generators.wanphysics.adapter import WanSubprocessGenerator
from generators.wanphysics.executor_factory import build_executor_registry
from generators.wanphysics.sam2_vlm_critic import Sam2VlmSubprocessCritic

_DEFAULT_CONFIG = Path("/root/PhysGenLoop-/configs/loop.yaml")


def _load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_policy(ckpt_root: str):
    ckpt_path = Path(ckpt_root)
    compatibility = CompatibilityManifest.load(
        str(ckpt_path / "config/critic_compatibility_v1.json")
    )
    if not compatibility.deployment_ready:
        print("[trial] WARNING: deployment_ready=False (proxy mode, actual_trial_count=0)")
    learned = TorchActionValuePolicy.load(
        str(ckpt_path / "model/best_action_value_policy.pt"),
        device="cpu",
        compatibility_manifest=compatibility,
    )
    return ActionValueDecisionPolicy(learned, minimum_confidence=0.35), compatibility


def main() -> int:
    parser = argparse.ArgumentParser(description="PhysGenLoop 单 prompt Trial 采集")
    parser.add_argument("--prompt", required=True, help="输入 prompt 文本")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--ckpt-root", default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--group-id", default="default")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    paths = cfg["paths"]
    loop = cfg["loop"]
    gen = cfg["generator"]
    vllm_cfg = cfg["vllm"]

    ckpt_root = args.ckpt_root or paths["checkpoints"]["repair_agent"]
    max_attempts = args.max_attempts if args.max_attempts is not None else loop["max_rounds"]

    run_id = datetime.now().strftime("trial_%Y%m%d_%H%M%S")
    run_dir = Path(paths["outputs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[trial] output dir: {run_dir}", flush=True)

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
    critic = Sam2VlmSubprocessCritic(
        python=paths["envs"]["main"],
        vllm_python=paths["envs"]["vllm"],
        vllm_model=paths["models"]["vllm"],
        vllm_served_name=vllm_cfg["served_name"],
        vllm_log=str(run_dir / "vllm_trial.log"),
        vllm_gpu_util=vllm_cfg["gpu_util"],
        vllm_max_model_len=vllm_cfg["max_model_len"],
    )
    policy, compatibility = _build_policy(ckpt_root)
    executors = build_executor_registry(
        run_dir=str(run_dir),
        ckpt_root=ckpt_root,
        python=paths["envs"]["main"],
        propainter_repo=paths.get("propainter_repo", "/root/ProPainter"),
    )
    selector = EvidenceAwareSelector()
    recorder = JsonlTrialRecorder(run_dir / "trials.jsonl")

    runner_config = RunnerConfig(
        max_attempts=max_attempts,
        acceptance_score=loop["acceptance_score"],
        base_seed=loop["base_seed"],
        domain="hunyuan",
        require_quality_metrics=False,
    )

    runner = LearningRepairLoopRunner(
        generator=generator,
        critic=critic,
        selector=selector,
        policy=policy,
        executors=executors,
        semantic_scorer=None,
        quality_scorer=None,
        config=runner_config,
        recorder=recorder,
        compatibility_manifest=compatibility,
    )

    try:
        result = runner.run(
            prompt=args.prompt,
            physics_plan=PhysicsPlan(),
            group_id=args.group_id,
            run_id=run_id,
        )
    finally:
        critic.shutdown()

    trial_count = len(result.trials)
    trial_result = {
        "run_id": run_id,
        "group_id": args.group_id,
        "stop_reason": result.stop_reason,
        "final_physics_score": float(result.final_report.get("physics_score", 0.0)),
        "final_decision": result.final_report.get("decision"),
        "trial_count": trial_count,
        "trials_jsonl": str(run_dir / "trials.jsonl"),
    }
    (run_dir / "trial_result.json").write_text(
        json.dumps(trial_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(trial_result, ensure_ascii=False, indent=2))
    print(f"[trial] {trial_count} trial(s) written to {run_dir / 'trials.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
