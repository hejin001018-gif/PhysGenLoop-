"""Repair Agent — 依 Critic 输出决策修复策略，产出下一轮 generator_request。

策略层级（对齐 框架骨干§六）：
    L1 Prompt 增强：physics_score > 0.6
    L2 局部 inpainting：0.4 < score ≤ 0.6  【首期不启用，HunyuanVideo 未原生支持】
    L3 全片重生：score ≤ 0.4
    L4 提前终止：连续 max_rounds 轮无改善
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Any

from .prompt_rewriter import LLMClient, RewriteRequest, make_client, rewrite


@dataclass
class RepairConfig:
    prompt_hint_threshold: float = 0.6
    local_inpaint_threshold: float = 0.4
    max_rounds: int = 2
    enable_local_inpaint: bool = False   # 首期关闭


@dataclass
class RepairState:
    original_prompt: str
    round_idx: int = 0
    history_scores: list[float] = None    # type: ignore[assignment]
    prior_hints: list[str] = None         # type: ignore[assignment]

    def __post_init__(self):
        self.history_scores = self.history_scores or []
        self.prior_hints = self.prior_hints or []


@dataclass
class RepairDecision:
    action: str                # "prompt_only" | "local_inpaint" | "full_regen" | "stop"
    generator_request: dict[str, Any] | None
    reason: str


def _no_improvement(history: list[float], k: int = 2) -> bool:
    """连续 k 轮 score 无提升。"""
    if len(history) <= k:
        return False
    recent = history[-(k + 1):]
    return all(recent[i + 1] <= recent[i] + 1e-3 for i in range(len(recent) - 1))


def decide(
    critic_output: dict[str, Any],
    state: RepairState,
    config: RepairConfig | None = None,
    llm: LLMClient | None = None,
    output_path: str = "outputs/video.mp4",
    seed: int | None = None,
) -> RepairDecision:
    config = config or RepairConfig()
    score = float(critic_output.get("physics_score", 0.0))
    state.history_scores.append(score)

    # L4 停机条件
    if state.round_idx >= config.max_rounds:
        return RepairDecision("stop", None, f"max_rounds={config.max_rounds} reached")
    if _no_improvement(state.history_scores, k=2):
        return RepairDecision("stop", None, "no improvement over last 2 rounds")

    # 已通过物理校验则不需修复
    if critic_output.get("is_physical") is True:
        return RepairDecision("stop", None, "critic passed")

    # 生成 prompt 重写
    violations = critic_output.get("violations", [])
    rewrite_res = rewrite(
        RewriteRequest(
            original_prompt=state.original_prompt,
            violations=violations,
            prior_hints=state.prior_hints,
        ),
        client=llm,
    )
    state.prior_hints.append(rewrite_res.physics_hint)

    # 策略选择
    if score > config.prompt_hint_threshold:
        action = "prompt_only"
        new_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    elif score > config.local_inpaint_threshold and config.enable_local_inpaint:
        action = "local_inpaint"
        new_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    else:
        action = "full_regen"
        new_seed = random.randint(0, 2**31 - 1)  # 强制换 seed

    critical_frames: list[int] = []
    for v in violations:
        critical_frames.extend(v.get("critical_frames", []))

    request: dict[str, Any] = {
        "prompt": rewrite_res.prompt,
        "seed": new_seed,
        "resolution": "480p",
        "num_frames": 121,
        "num_inference_steps": 50,
        "image_path": None,
        "output_path": output_path,
        "physics_hint": rewrite_res.physics_hint,
        "repair_context": {
            "critical_frames": sorted(set(critical_frames)),
            "object_masks": [],
            "strategy": action,
        },
        "generation_meta": {
            "attempt": state.round_idx + 1,
            "parent_video_path": None,
            "loop_round": state.round_idx + 1,
        },
    }

    state.round_idx += 1
    return RepairDecision(action, request, f"score={score:.3f} → {action}")


# ---------- convenience one-shot ----------

def repair_once(
    critic_output: dict[str, Any],
    original_prompt: str,
    backend: str = "stub",
    output_path: str = "outputs/video.mp4",
) -> RepairDecision:
    """给一次 Critic 输出，返回下一轮请求（不维持状态）。"""
    state = RepairState(original_prompt=original_prompt)
    return decide(critic_output, state, llm=make_client(backend), output_path=output_path)


if __name__ == "__main__":
    import json, sys
    critic = json.loads(sys.stdin.read())
    d = repair_once(critic, original_prompt=critic.get("_original_prompt", "A red ball falls."))
    print(json.dumps({"action": d.action, "reason": d.reason, "request": d.generator_request},
                     indent=2, ensure_ascii=False))
