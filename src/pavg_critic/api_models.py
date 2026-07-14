"""OpenAI Responses 与 DeepSeek Chat 的轻量结构化文本适配器。

本模块只负责 HTTPS/JSON 边界，不在源码、配置或日志中保存密钥。调用方通过环境变量
或构造参数提供凭据；测试可注入 transport，因此不会依赖网络或特定厂商 SDK。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class ModelAPIError(RuntimeError):
    """模型服务返回网络错误、非 JSON 内容或不完整响应。"""


class JsonTransport(Protocol):
    """可替换的 JSON POST 传输层。"""

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_sec: float,
    ) -> Mapping[str, Any]: ...


class UrllibJsonTransport:
    """仅使用标准库的 HTTPS JSON 传输，避免核心包强制依赖厂商 SDK。"""

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_sec: float,
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # 响应正文有助于定位配额/模型名错误，但绝不回显 Authorization header。
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ModelAPIError(f"Model API returned HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise ModelAPIError(f"Model API request failed: {exc}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelAPIError("Model API returned invalid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ModelAPIError("Model API response root must be an object")
        return parsed


@dataclass
class OpenAIResponsesModel:
    """通过 OpenAI Responses API 生成严格 JSON Schema 输出。"""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: float = 120.0
    transport: JsonTransport | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI api_key must not be empty")
        if not self.model:
            raise ValueError("OpenAI model must not be empty")
        self.transport = self.transport or UrllibJsonTransport()

    @classmethod
    def from_env(cls, **overrides: Any) -> "OpenAIResponsesModel":
        """从环境读取凭据和模型；不硬编码可能随时间变化的默认模型。"""

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY before enabling the OpenAI adapter")
        model = os.getenv("OPENAI_MODEL", "")
        if not model:
            raise ValueError("Set OPENAI_MODEL before enabling the OpenAI adapter")
        return cls(api_key=api_key, model=model, **overrides)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        assert self.transport is not None
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "pavg_structured_output",
                    "strict": True,
                    "schema": dict(schema),
                }
            },
        }
        response = self.transport.post_json(
            url=f"{self.base_url.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_sec=self.timeout_sec,
        )
        return _parse_json_text(_openai_output_text(response))

    def generate_json_with_images(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: tuple[str, ...] | list[str],
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """把关键帧 data URL 与文本问题一起发送到 Responses API。"""

        if not image_data_urls:
            raise ValueError("At least one image is required for multimodal generation")
        assert self.transport is not None
        user_content = [{"type": "input_text", "text": user_prompt}]
        user_content.extend(
            {"type": "input_image", "image_url": image_url}
            for image_url in image_data_urls
        )
        response = self.transport.post_json(
            url=f"{self.base_url.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "pavg_multimodal_review",
                        "strict": True,
                        "schema": dict(schema),
                    }
                },
            },
            timeout_sec=self.timeout_sec,
        )
        return _parse_json_text(_openai_output_text(response))


@dataclass
class DeepSeekChatModel:
    """通过 DeepSeek 的 OpenAI-compatible Chat Completions 接口生成 JSON。"""

    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    timeout_sec: float = 120.0
    transport: JsonTransport | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("DeepSeek api_key must not be empty")
        if not self.model:
            raise ValueError("DeepSeek model must not be empty")
        self.transport = self.transport or UrllibJsonTransport()

    @classmethod
    def from_env(cls, **overrides: Any) -> "DeepSeekChatModel":
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("Set DEEPSEEK_API_KEY before enabling the DeepSeek adapter")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return cls(api_key=api_key, model=model, **overrides)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        assert self.transport is not None
        # DeepSeek 的 json_object 模式不接收完整 schema，因此把约束文本一并放入
        # system message，返回后仍由 PAVG dataclass/DAG 校验执行第二道验证。
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        response = self.transport.post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"{system_prompt}\nReturn JSON matching: {schema_text}",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout_sec=self.timeout_sec,
        )
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError("DeepSeek response contains no assistant content") from exc
        return _parse_json_text(text)


def _openai_output_text(response: Mapping[str, Any]) -> str:
    """从原始 Responses API 输出数组提取第一个 output_text。"""

    output = response.get("output", ())
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content", ())
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "output_text":
                    return str(part.get("text", ""))
    raise ModelAPIError("OpenAI response contains no output_text")


def _parse_json_text(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ModelAPIError("Model returned empty structured output")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ModelAPIError("Model output is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ModelAPIError("Structured model output root must be an object")
    return parsed
