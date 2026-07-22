"""真实 Semantic Scorer：用 Qwen3-VL 看视频关键帧做语义保持判定（V2，P0-06）。

修复差距审查 P0-06：旧版只把 Prompt 文本发给 VLM，忽略 video_path，等于评估 Prompt
而非"生成视频是否保持 Prompt 语义"。本版：
  - 从 after candidate 视频抽取关键帧（cv2 均匀采样）；
  - 以 image_url(base64) 多模态输入发给 Qwen3-VL；
  - 返回结构化结果（entity/event/temporal/new_object + reason + evidence + degraded），
    不再只返回 float；
  - 视频不可读 / VLM 不可用时返回 available=False，绝不用 physics_score 冒充。

独立性：使用与 physics Critic 不同的独立 prompt/schema。是否共享同一 Qwen3-VL 实例
由 backend 字段显式标注，供正式评估阶段区分 shared vs independent。
"""
from __future__ import annotations

import base64
import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any

SEMANTIC_SCORER_SCHEMA_VERSION = "semantic-scorer/2.0"

_SEMANTIC_PROMPT = """\
You are an objective evaluator of whether a GENERATED VIDEO preserves the semantics of a text prompt.

Original prompt: {prompt}

You are shown {n} keyframes sampled from the generated video.
Judge whether the video depicts the same scene, objects and intended action as the prompt.

Answer ONLY with a JSON object:
{{"score": <0.0-1.0>, "entity_preservation": <0.0-1.0>, "event_preservation": <0.0-1.0>,
 "temporal_alignment": <0.0-1.0>, "new_object_penalty": <0.0-1.0>, "reason": "<one sentence>"}}

score: overall semantic preservation. Respond with valid JSON only.
"""


@dataclass(frozen=True)
class SemanticResult:
    score: float | None
    available: bool
    entity_preservation: float | None = None
    event_preservation: float | None = None
    temporal_alignment: float | None = None
    new_object_penalty: float | None = None
    reason: str = ""
    backend: str = ""
    model_revision: str = ""
    evidence_frames: tuple[int, ...] = ()
    degraded: bool = False
    degraded_reasons: tuple[str, ...] = ()
    schema_version: str = SEMANTIC_SCORER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "score": self.score,
            "available": self.available,
            "entity_preservation": self.entity_preservation,
            "event_preservation": self.event_preservation,
            "temporal_alignment": self.temporal_alignment,
            "new_object_penalty": self.new_object_penalty,
            "reason": self.reason,
            "backend": self.backend,
            "model_revision": self.model_revision,
            "evidence_frames": list(self.evidence_frames),
            "degraded": self.degraded,
            "degraded_reasons": list(self.degraded_reasons),
        }


def _extract_keyframes(video_path: str, n: int = 4) -> tuple[list[str], list[int]]:
    """均匀抽 n 帧，返回 (base64_jpegs, frame_indices)。cv2 缺失/不可读返回空。"""
    try:
        import cv2  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return [], []
    idxs = [int(total * k / (n + 1)) for k in range(1, n + 1)]
    frames_b64: list[str] = []
    got: list[int] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok2, buf = cv2.imencode(".jpg", frame)
        if not ok2:
            continue
        frames_b64.append(base64.b64encode(buf.tobytes()).decode())
        got.append(idx)
    cap.release()
    return frames_b64, got


class VlmSemanticScorer:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "local",
        model: str = "qwen3-vl-8b-instruct",
        timeout: int = 60,
        num_keyframes: int = 4,
        shared_with_critic: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._n = num_keyframes
        self._backend = "qwen3-vl-shared" if shared_with_critic else "qwen3-vl-independent"

    @property
    def available(self) -> bool:
        try:
            urllib.request.urlopen(f"{self._base_url.replace('/v1', '')}/health", timeout=2)
            return True
        except Exception:  # noqa: BLE001
            return False

    def score_structured(self, *, prompt: str, video_path: str) -> SemanticResult:
        if not self.available:
            return SemanticResult(score=None, available=False, backend=self._backend,
                                  degraded=True, degraded_reasons=("vllm_unavailable",))
        frames, idxs = _extract_keyframes(video_path, self._n)
        if not frames:
            return SemanticResult(score=None, available=False, backend=self._backend,
                                  degraded=True, degraded_reasons=("no_keyframes",))
        content: list[dict[str, Any]] = [
            {"type": "text", "text": _SEMANTIC_PROMPT.format(prompt=prompt, n=len(frames))}
        ]
        for b64 in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        try:
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self._base_url}/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return SemanticResult(score=None, available=False, backend=self._backend,
                                      degraded=True, degraded_reasons=("invalid_json",), reason=text[:120])
            data = json.loads(m.group())
            score = float(data.get("score", -1))
            if not 0.0 <= score <= 1.0:
                return SemanticResult(score=None, available=False, backend=self._backend,
                                      degraded=True, degraded_reasons=("score_out_of_range",))
            def _opt(k):
                v = data.get(k)
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None
            return SemanticResult(
                score=score, available=True, backend=self._backend, model_revision=self._model,
                entity_preservation=_opt("entity_preservation"),
                event_preservation=_opt("event_preservation"),
                temporal_alignment=_opt("temporal_alignment"),
                new_object_penalty=_opt("new_object_penalty"),
                reason=str(data.get("reason", ""))[:200],
                evidence_frames=tuple(idxs),
            )
        except Exception as exc:  # noqa: BLE001
            return SemanticResult(score=None, available=False, backend=self._backend,
                                  degraded=True, degraded_reasons=(f"error:{type(exc).__name__}",))

    def score(self, *, prompt: str, video_path: str) -> float | None:
        """兼容旧接口：返回 float 或 None。"""
        return self.score_structured(prompt=prompt, video_path=video_path).score
