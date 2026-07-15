"""Prompt Rewriter — 把 Critic 的 violations 翻译为新 prompt / physics_hint。

设计思路：LLM 侧解耦——统一 LLMClient 接口，具体 backend（Claude / OpenAI /
本地 Qwen）由 configs/default.yaml 决定。首期只做「基于 violations 的 prompt 增强」。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


REWRITE_SYSTEM_PROMPT = """你是视频生成的物理修正专家。用户会给你：
1. 原始 prompt
2. 上一次生成的物理违规列表（category / object / reason / repair_instruction）

请输出一份改写后的 prompt，要求：
- 保留原始语义与主体，不改变场景类型
- 显式加入违反的物理约束的正向表述（如"the ball must contact the floor before bouncing"）
- 用一到两句话补充，不要展开成长段
- 只输出 JSON：{"prompt": "...", "physics_hint": "..."}
"""


@dataclass
class RewriteRequest:
    original_prompt: str
    violations: list[dict[str, Any]]
    prior_hints: list[str] | None = None


@dataclass
class RewriteResult:
    prompt: str
    physics_hint: str
    raw_response: str | None = None


class LLMClient(Protocol):
    name: str
    def chat(self, system: str, user: str) -> str: ...


# ---------- backends ----------

class ClaudeClient:
    """Anthropic Claude API 客户端。需要 ANTHROPIC_API_KEY。"""
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def chat(self, system: str, user: str) -> str:
        import anthropic
        client = anthropic.Anthropic()  # 从 env 读 ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # 简化：取第一段 text
        return "".join(getattr(b, "text", "") for b in resp.content)


class OpenAIClient:
    """OpenAI Chat Completions。需要 OPENAI_API_KEY。"""
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def chat(self, system: str, user: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class StubClient:
    """离线开发用。返回规则拼接的伪结果，允许无 API Key 跑通流程。"""
    name = "stub"

    def chat(self, system: str, user: str) -> str:
        # 从 user 中抓原 prompt 与 violations，返回一个规则版
        try:
            data = json.loads(user)
        except Exception:
            data = {"original_prompt": user, "violations": []}
        hints = [
            v.get("repair_instruction") or f"Fix {v.get('category', 'anomaly')} on {v.get('object', 'object')}"
            for v in data.get("violations", [])
        ]
        hint = " ".join(hints).strip() or "Ensure physical plausibility."
        new_prompt = f"{data.get('original_prompt', '')}. {hint}".strip()
        return json.dumps({"prompt": new_prompt, "physics_hint": hint})


def make_client(backend: str = "stub") -> LLMClient:
    if backend == "claude":
        return ClaudeClient()
    if backend == "openai":
        return OpenAIClient()
    if backend == "stub" or os.environ.get("PAVG_LLM_STUB") == "1":
        return StubClient()
    raise ValueError(f"unknown backend: {backend}")


# ---------- main API ----------

def rewrite(req: RewriteRequest, client: LLMClient | None = None) -> RewriteResult:
    client = client or make_client("stub")
    user_payload = json.dumps({
        "original_prompt": req.original_prompt,
        "violations": req.violations,
        "prior_hints": req.prior_hints or [],
    }, ensure_ascii=False)
    raw = client.chat(REWRITE_SYSTEM_PROMPT, user_payload)

    # 兜底解析：先尝试 JSON，失败则整段作为 prompt
    try:
        parsed = json.loads(raw)
        return RewriteResult(
            prompt=parsed.get("prompt", req.original_prompt),
            physics_hint=parsed.get("physics_hint", ""),
            raw_response=raw,
        )
    except Exception:
        return RewriteResult(
            prompt=req.original_prompt,
            physics_hint=raw.strip(),
            raw_response=raw,
        )
