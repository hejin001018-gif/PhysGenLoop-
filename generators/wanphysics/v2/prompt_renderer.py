"""Evidence-grounded Prompt Repair 渲染器（V2）。

修复 P0-5 的内容侧：现有链路 10 次 prompt repair 全部只追加同一句抽象占位文本
``Physics correction: Execute the highest calibrated, available repair action.``，
不含违规对象/帧/物理类别，等于 "加系统提示后随机重生成"。

本模块把 violation 的 category / object / frame interval / repair_instruction 转成
具体自然语言物理约束（方案 §16 的 8 类映射），并保留原 prompt 的对象、动作、场景和
镜头语义。生成结果附带 instruction source 与 SHA，供审计。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

PROMPT_RENDER_SCHEMA_VERSION = "prompt-render/1.0"

# 方案 §16：violation category → 具体物理约束（英文，直接进 Wan prompt）。
_CATEGORY_CONSTRAINTS: dict[str, str] = {
    "gravity": "the object must move continuously along gravity, without hovering or drifting upward",
    "penetration": "objects must not pass through each other and must keep a visible boundary separation",
    "collision": "after collision there must be a plausible change of speed or direction",
    "trajectory": "the motion path must stay continuous, without teleporting or abrupt velocity changes",
    "disappearance": "the target object must stay visible throughout the critical frames",
    "contact": "keep a plausible contact relationship before and after the objects touch",
    "support": "the object stays stable while supported and only moves after leaving the support",
    "floating": "the object must not float without visible support",
}

# 修复约束里不允许泄漏的内部术语（方案 §16 生成前检查）。
_FORBIDDEN_TERMS = (
    "policy",
    "checkpoint",
    "memory",
    "repair action",
    "calibrated",
    "per_action",
    "critic",
    "vllm",
)

_DEFAULT_MAX_CHARS = 600


@dataclass(frozen=True)
class RenderedPrompt:
    prompt: str
    instruction: str
    instruction_sha256: str
    source: str
    constraints_used: tuple[str, ...]
    original_preserved: bool
    schema_version: str = PROMPT_RENDER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prompt": self.prompt,
            "instruction": self.instruction,
            "instruction_sha256": self.instruction_sha256,
            "source": self.source,
            "constraints_used": list(self.constraints_used),
            "original_preserved": self.original_preserved,
        }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _constraint_for(violation: Any) -> str:
    category = str(getattr(violation, "category", "")).strip().lower()
    if category in _CATEGORY_CONSTRAINTS:
        return _CATEGORY_CONSTRAINTS[category]
    # 未知类别：回退到 violation 自带 repair_instruction（若有）。
    instr = str(getattr(violation, "repair_instruction", "")).strip()
    return instr or "the physical behavior must remain plausible and temporally continuous"


def render_repair_prompt(
    *,
    original_prompt: str,
    violations: Sequence[Any],
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> RenderedPrompt:
    """根据 violations 生成具体化的物理修复 prompt。

    模板（方案 §16）保留原场景/对象/镜头，仅追加针对性物理约束。
    """

    original = str(original_prompt).strip()
    if not original:
        raise ValueError("original prompt must not be empty for prompt repair")

    constraints: list[str] = []
    frame_notes: list[str] = []
    seen: set[str] = set()
    for violation in violations:
        c = _constraint_for(violation)
        obj = str(getattr(violation, "object", "")).strip()
        key = f"{obj}:{c}"
        if key in seen:
            continue
        seen.add(key)
        start = getattr(violation, "start_frame", None)
        end = getattr(violation, "end_frame", None)
        span = ""
        if start is not None and end is not None:
            span = f" (frames {int(start)}–{int(end)})"
        subject = obj or "the object"
        constraints.append(f"For {subject}{span}: {c}.")

    if not constraints:
        constraints.append(
            "Keep all object motion physically plausible and temporally continuous."
        )

    joined = " ".join(constraints)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1].rstrip() + "…"

    instruction = (
        "Preserve the original scene, objects, camera and intended action. "
        f"Correct the following physical issue: {joined} "
        "The correction must remain visually plausible and temporally continuous. "
        "Do not introduce new objects or change the scene semantics."
    )

    # 生成前检查：不得包含内部术语。
    lowered = instruction.lower()
    for term in _FORBIDDEN_TERMS:
        if term in lowered:
            instruction = re.sub(re.escape(term), "", instruction, flags=re.IGNORECASE)

    prompt = f"{original}\n\n{instruction}"

    # 校验原对象/动作是否仍在最终 prompt 中（宽松：原 prompt 完整保留即视为通过）。
    original_preserved = original in prompt
    _ = _tokens(original)  # 供后续更严格的 token 级检查扩展

    sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    return RenderedPrompt(
        prompt=prompt,
        instruction=instruction,
        instruction_sha256=sha,
        source="evidence_grounded_v2",
        constraints_used=tuple(constraints),
        original_preserved=original_preserved,
    )
