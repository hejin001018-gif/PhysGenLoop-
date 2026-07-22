"""Proxy Memory 格式识别与兼容适配（V2）。

修复 P0-3(memory)：checkpoint 里的 ``proxy_memory_train.jsonl`` 是
``LearningTargetV1`` 格式（``target_action`` + ``action_rewards``），而生产 Repairer 仍
用旧 ``RepairMemory``/``RepairExample`` loader（要求 ``target`` 是 object），远端实测
``invalid repair sample 0: repair example target must be an object``，memory mixing 被
静默关闭还每样本刷屏。

本模块只做**只读识别**，不修改任何既有 loader：读取第一条记录判定格式，整轮只初始化
一次并写一次 ``memory_status``。默认 ``disabled``——proxy memory 只能提供 action 分布，
不能伪造未执行动作的失败、真实 cost、semantic/quality 或 Wan2.2 成功率（方案 §17）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MEMORY_ADAPTER_SCHEMA_VERSION = "memory-adapter/1.0"

FORMAT_REPAIR_EXAMPLE = "canonical_repair_example"
FORMAT_PROXY_TARGET = "proxy_target_memory"
FORMAT_ACTUAL_TRIAL = "actual_trial"
FORMAT_INCOMPATIBLE = "incompatible"
FORMAT_EMPTY = "empty"


def detect_format(record: Mapping[str, Any]) -> str:
    """按修复方案 §17 的判定顺序识别单条记录格式。"""

    has_target_obj = isinstance(record.get("target"), Mapping)
    has_outcome = isinstance(record.get("outcome"), Mapping)
    if has_target_obj and has_outcome:
        return FORMAT_REPAIR_EXAMPLE
    if record.get("target_action") is not None and record.get("action_rewards") is not None:
        return FORMAT_PROXY_TARGET
    if (
        record.get("decision") is not None
        and record.get("execution") is not None
        and record.get("critic_before") is not None
    ):
        return FORMAT_ACTUAL_TRIAL
    return FORMAT_INCOMPATIBLE


@dataclass(frozen=True)
class MemoryStatus:
    """整轮只写一次的 memory 加载状态。"""

    enabled: bool
    memory_format: str
    memory_records: int
    memory_path: str | None
    memory_error: str | None = None
    schema_version: str = MEMORY_ADAPTER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_enabled": self.enabled,
            "memory_format": self.memory_format,
            "memory_records": self.memory_records,
            "memory_path": self.memory_path,
            "memory_error": self.memory_error,
        }


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(records) >= limit:
                break
    return records


def inspect_memory(memory_path: str | Path, *, enable: bool = False) -> MemoryStatus:
    """只读探测 memory 文件，返回一次性状态。永不抛异常。

    ``enable=False``（默认）时即使格式兼容也标记 ``enabled=False``——启用 proxy memory
    混合必须由独立开关显式打开（方案 §28）。
    """

    path = Path(memory_path)
    if not path.exists():
        return MemoryStatus(
            enabled=False,
            memory_format=FORMAT_EMPTY,
            memory_records=0,
            memory_path=str(path),
            memory_error="memory file not found",
        )
    try:
        records = _read_jsonl(path)
    except Exception as exc:  # noqa: BLE001
        return MemoryStatus(
            enabled=False,
            memory_format=FORMAT_INCOMPATIBLE,
            memory_records=0,
            memory_path=str(path),
            memory_error=f"{type(exc).__name__}: {exc}",
        )
    if not records:
        return MemoryStatus(
            enabled=False,
            memory_format=FORMAT_EMPTY,
            memory_records=0,
            memory_path=str(path),
            memory_error="memory file is empty",
        )

    fmt = detect_format(records[0])
    # 仅 proxy target 或 canonical repair example 可作为 proxy 分布来源，
    # 且必须显式 enable。actual_trial/incompatible 不在本适配器启用范围。
    can_enable = enable and fmt in {FORMAT_PROXY_TARGET, FORMAT_REPAIR_EXAMPLE}
    error = None
    if fmt in {FORMAT_INCOMPATIBLE, FORMAT_ACTUAL_TRIAL} and enable:
        error = f"memory format {fmt!r} not supported by v2 proxy adapter"
    return MemoryStatus(
        enabled=can_enable,
        memory_format=fmt,
        memory_records=len(records),
        memory_path=str(path),
        memory_error=error,
    )


def proxy_action_distribution(memory_path: str | Path) -> dict[str, float] | None:
    """从 proxy target memory 聚合一个粗粒度 action 频率分布（只读，可选使用）。

    仅供 proxy research 混合参考，不代表真实修复价值。格式不符时返回 None。
    """

    path = Path(memory_path)
    if not path.exists():
        return None
    records = _read_jsonl(path)
    if not records or detect_format(records[0]) != FORMAT_PROXY_TARGET:
        return None
    counts: dict[str, float] = {}
    for rec in records:
        action = rec.get("target_action")
        if action is None:
            continue
        key = str(action)
        counts[key] = counts.get(key, 0.0) + 1.0
    total = sum(counts.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in counts.items()}
