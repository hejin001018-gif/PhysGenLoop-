"""proxy memory 格式识别与状态。"""
import json
from generators.wanphysics.v2.memory_adapter import (
    detect_format, inspect_memory, FORMAT_PROXY_TARGET, FORMAT_REPAIR_EXAMPLE, FORMAT_INCOMPATIBLE, FORMAT_EMPTY,
)


def test_detect_formats():
    assert detect_format({"target": {"action": "x"}, "outcome": {}}) == FORMAT_REPAIR_EXAMPLE
    assert detect_format({"target_action": "prompt_repair", "action_rewards": {}}) == FORMAT_PROXY_TARGET
    assert detect_format({"foo": 1}) == FORMAT_INCOMPATIBLE


def test_inspect_missing(tmp_path):
    st = inspect_memory(tmp_path / "nope.jsonl")
    assert not st.enabled and st.memory_format == FORMAT_EMPTY


def test_proxy_target_disabled_by_default(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"target_action": "reject", "action_rewards": {"reject": 1.0}}) + "\n", encoding="utf-8")
    st = inspect_memory(p, enable=False)
    assert st.memory_format == FORMAT_PROXY_TARGET
    assert st.enabled is False  # 必须显式 enable


def test_proxy_target_enabled(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"target_action": "reject", "action_rewards": {"reject": 1.0}}) + "\n", encoding="utf-8")
    st = inspect_memory(p, enable=True)
    assert st.enabled is True and st.memory_records == 1
