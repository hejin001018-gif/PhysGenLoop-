"""OpenAI Responses 与 DeepSeek Chat 的轻量结构化文本适配器。

本模块只负责 HTTPS/JSON 边界，不在源码、配置或日志中保存密钥。调用方通过环境变量
或构造参数提供凭据；测试可注入 transport，因此不会依赖网络或特定厂商 SDK。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover — python-dotenv 是可选的
    pass


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

    api_key: str = field(repr=False)
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: float = 120.0
    transport: JsonTransport | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI api_key must not be empty")
        if not self.model:
            raise ValueError("OpenAI model must not be empty")
        _validate_https_base_url(self.base_url, "OpenAI")
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

    api_key: str = field(repr=False)
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    timeout_sec: float = 120.0
    transport: JsonTransport | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("DeepSeek api_key must not be empty")
        if not self.model:
            raise ValueError("DeepSeek model must not be empty")
        _validate_https_base_url(self.base_url, "DeepSeek")
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


@dataclass
class OpenAIChatModel:
    """OpenAI 兼容 Chat Completions 模型，同时支持纯文本和多模态生成。

    使用 ``/chat/completions`` 端点而非 Responses API，因此兼容绝大多数
    OpenAI 兼容中转站（包括 DeepSeek、本地 vLLM 等）。

    同时满足 :class:`StructuredTextModel` 和 :class:`MultimodalStructuredModel`
    两个 Protocol，可直接注入到 :class:`PhysicsCritic` 中。
    """

    api_key: str = field(repr=False)
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: float = 120.0
    transport: JsonTransport | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAIChatModel api_key must not be empty")
        if not self.model:
            raise ValueError("OpenAIChatModel model must not be empty")
        _validate_base_url(self.base_url, "OpenAIChat")
        self.transport = self.transport or UrllibJsonTransport()

    @classmethod
    def from_env(
        cls,
        *,
        model_env: str = "TEXT_MODEL",
        **overrides: Any,
    ) -> "OpenAIChatModel":
        """从环境变量或 ``.env`` 文件读取配置。

        读取顺序（同名变量优先使用通用名称，再 fallback 到 provider 专属名称）：

        - api_key: ``API_KEY`` → ``OPENAI_API_KEY``
        - base_url: ``BASE_URL`` → ``OPENAI_BASE_URL`` → ``https://api.openai.com/v1``
        - model: ``model_env`` 指定的变量名 → ``OPENAI_MODEL``

        Args:
            model_env: 读取模型名时优先使用的环境变量名，例如 ``"VLM_MODEL"`` 或 ``"TEXT_MODEL"``。
            **overrides: 直接传给构造器的额外参数（如 ``timeout_sec``）。
        """
        api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not api_key:
            raise ValueError(
                "Set API_KEY (or OPENAI_API_KEY) in .env or environment "
                "before using OpenAIChatModel"
            )
        base_url = os.getenv(
            "BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        model = os.getenv(model_env, os.getenv("OPENAI_MODEL", ""))
        if not model:
            raise ValueError(
                f"Set {model_env} (or OPENAI_MODEL) in .env or environment "
                f"before using OpenAIChatModel"
            )
        return cls(api_key=api_key, model=model, base_url=base_url, **overrides)

    # ------------------------------------------------------------------
    # StructuredTextModel protocol
    # ------------------------------------------------------------------

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """纯文本 Chat Completions 调用，要求模型返回 JSON 对象。"""
        assert self.transport is not None
        # 把 JSON Schema 一并嵌入 system prompt，辅助 json_object 模式输出。
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        response = self.transport.post_json(
            url=_build_chat_completions_url(self.base_url),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"{system_prompt}\n\n"
                            f"Return a JSON object matching this schema: {schema_text}"
                        ),
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
            raise ModelAPIError(
                "Chat Completions response contains no assistant content"
            ) from exc
        return _parse_json_text(text)

    # ------------------------------------------------------------------
    # MultimodalStructuredModel protocol
    # ------------------------------------------------------------------

    def generate_json_with_images(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: Sequence[str],
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """多模态 Chat Completions 调用，将图片以 data URL 嵌入用户消息。"""
        if not image_data_urls:
            raise ValueError(
                "At least one image is required for multimodal generation"
            )
        assert self.transport is not None
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        # 构建 Vision API 格式的 content 数组
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt}
        ]
        user_content.extend(
            {"type": "image_url", "image_url": {"url": data_url}}
            for data_url in image_data_urls
        )
        response = self.transport.post_json(
            url=_build_chat_completions_url(self.base_url),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"{system_prompt}\n\n"
                            f"Return a JSON object matching this schema: {schema_text}"
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout_sec=self.timeout_sec,
        )
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError(
                "Chat Completions response contains no assistant content"
            ) from exc
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


def _validate_https_base_url(value: str, provider: str) -> None:
    """Bearer 凭据只允许发送到具有主机名的 HTTPS endpoint。"""

    parsed = urlparse(value)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"{provider} base_url must be an absolute HTTPS URL")


def _validate_base_url(value: str, provider: str) -> None:
    """校验 base_url，允许 HTTPS 及本地 localhost HTTP。

    公有云服务强制 HTTPS 以防止 Bearer token 泄漏，但本地 Ollama/vLLM
    实例通常运行在 http://localhost，需要放行。
    """
    parsed = urlparse(value)
    if not parsed.netloc:
        raise ValueError(f"{provider} base_url must be an absolute URL")
    scheme = parsed.scheme.lower()
    if scheme == "https":
        return
    if scheme == "http":
        host = parsed.hostname or ""
        if host in ("localhost", "127.0.0.1", "::1") or host.startswith("192.168."):
            return  # 本地/内网地址允许 HTTP
    raise ValueError(
        f"{provider} base_url must be HTTPS (or HTTP for localhost/private IP)"
    )


def _build_chat_completions_url(base_url: str) -> str:
    """构建 /chat/completions 完整 URL，避免重复拼接。

    兼容三种常见格式：

    - 官方 OpenAI: ``https://api.openai.com/v1`` → 追加 ``/chat/completions``
    - DeepSeek: ``https://api.deepseek.com`` → 追加 ``/chat/completions``
    - 已含完整路径的中转站: ``https://proxy.example.com/v1/chat/completions`` → 不变
    """
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"
