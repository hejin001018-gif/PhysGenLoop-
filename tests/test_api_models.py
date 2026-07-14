"""文本模型 API 适配器只测试请求/响应边界，不发送真实网络请求。"""

from __future__ import annotations

import pytest

from pavg_critic.api_models import DeepSeekChatModel, OpenAIResponsesModel


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, *, url, headers, payload, timeout_sec):
        self.calls.append(
            {"url": url, "headers": headers, "payload": payload, "timeout": timeout_sec}
        )
        return self.response


def test_openai_responses_adapter_requests_strict_json_schema():
    transport = FakeTransport(
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"answer":"yes"}'}],
                }
            ]
        }
    )
    model = OpenAIResponsesModel(
        api_key="test-key",
        model="test-model",
        transport=transport,
    )

    result = model.generate_json(
        system_prompt="system",
        user_prompt="user",
        schema={"type": "object"},
    )

    assert result == {"answer": "yes"}
    payload = transport.calls[0]["payload"]
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert "Bearer test-key" == transport.calls[0]["headers"]["Authorization"]


def test_deepseek_adapter_uses_openai_compatible_chat_endpoint():
    transport = FakeTransport(
        {"choices": [{"message": {"content": '{"nodes":[]}'}}]}
    )
    model = DeepSeekChatModel(api_key="test-key", transport=transport)

    result = model.generate_json(
        system_prompt="system",
        user_prompt="user",
        schema={"type": "object"},
    )

    assert result == {"nodes": []}
    call = transport.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["payload"]["response_format"] == {"type": "json_object"}


def test_openai_from_env_requires_key_and_explicit_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIResponsesModel.from_env()
