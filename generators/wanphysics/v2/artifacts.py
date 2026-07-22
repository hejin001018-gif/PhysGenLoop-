"""不可变审计产物与样本状态（V2）。

修复方案 §11 / §19 / P1(可审计性)：V2 run 的每个 run/sample/candidate/action 都写
不可变审计产物，并维护 append-only 的 ``sample_status.json`` 供 resume 使用（不依赖
最终 summary.json 判断样本是否完成）。

所有写入只落在 V2 专属目录 ``outputs/v2_run_<ts>/`` 或 ``outputs/v2_trials_<ts>/``，
绝不在旧 ``videophy2_run_*`` 目录续写（方案 §4）。

本模块不假设 GPU/模型；纯文件 IO + JSON，可在 CPU 环境完整测试。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARTIFACTS_SCHEMA_VERSION = "v2-artifacts/1.0"

# 修复方案 §11 状态机的合法状态集合。
SAMPLE_STATES = (
    "CREATED",
    "PREFLIGHT_FAILED",
    "GENERATING",
    "GENERATED",
    "CRITIC_RUNNING",
    "CRITIC_FAILED",
    "CRITIC_COMPLETED",
    "ACCEPTED",
    "DECISION_READY",
    "EXECUTING",
    "EXECUTOR_FAILED",
    "RE_EVALUATING",
    "MAX_ROUNDS",
    "REJECTED",
    "COMPLETED",
)

# 终止状态：resume 时视为已完成，不重跑。
TERMINAL_STATES = frozenset({"ACCEPTED", "MAX_ROUNDS", "REJECTED", "COMPLETED", "PREFLIGHT_FAILED"})


def _atomic_write(path: Path, text: str) -> None:
    """原子写：先写临时文件再 rename，避免中断产生半截 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: str | Path, payload: Any) -> None:
    _atomic_write(Path(path), json.dumps(payload, ensure_ascii=False, indent=2))


def append_jsonl(path: str | Path, record: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class SampleStatus:
    sample_id: str
    state: str
    round_index: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    schema_version: str = ARTIFACTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.state not in SAMPLE_STATES:
            raise ValueError(f"invalid sample state: {self.state!r}")

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "state": self.state,
            "round_index": self.round_index,
            "detail": dict(self.detail),
        }


class RunArtifacts:
    """管理单次 V2 run 的目录树与审计产物写入。"""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # --- run level ---
    def write_run_manifest(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "run_manifest.json", payload)

    def write_summary(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "summary.json", payload)

    # --- sample level ---
    def sample_dir(self, sample_id: str) -> Path:
        d = self.run_dir / sample_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def set_status(self, status: SampleStatus) -> None:
        """写当前状态快照 + append 到状态历史（只前进，不覆盖历史）。"""

        d = self.sample_dir(status.sample_id)
        write_json(d / "sample_status.json", status.to_dict())
        append_jsonl(d / "sample_status_history.jsonl", status.to_dict())

    def read_status(self, sample_id: str) -> SampleStatus | None:
        p = self.run_dir / sample_id / "sample_status.json"
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return SampleStatus(
                sample_id=str(raw["sample_id"]),
                state=str(raw["state"]),
                round_index=int(raw.get("round_index", 0)),
                detail=dict(raw.get("detail", {})),
            )
        except Exception:  # noqa: BLE001
            return None

    def is_complete(self, sample_id: str) -> bool:
        status = self.read_status(sample_id)
        return bool(status and status.terminal)

    # --- candidate / action level ---
    def candidate_dir(self, sample_id: str, candidate_id: str) -> Path:
        d = self.sample_dir(sample_id) / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_critic_report(self, sample_id: str, candidate_id: str, payload: dict[str, Any]) -> None:
        write_json(self.candidate_dir(sample_id, candidate_id) / "critic_report.json", payload)

    def write_raw_payload(self, sample_id: str, candidate_id: str, raw: dict[str, Any], error: str) -> None:
        cdir = self.candidate_dir(sample_id, candidate_id)
        write_json(cdir / "critic_payload_raw.json", raw)
        write_json(cdir / "critic_roundtrip_error.json", {"error": error})

    def write_mask_manifest(self, sample_id: str, candidate_id: str, payload: dict[str, Any]) -> Path:
        path = self.candidate_dir(sample_id, candidate_id) / "mask_manifest.json"
        write_json(path, payload)
        return path

    def write_decision(self, sample_id: str, candidate_id: str, payload: dict[str, Any]) -> None:
        write_json(self.candidate_dir(sample_id, candidate_id) / "repair_decision.json", payload)

    def append_repair_trace(self, sample_id: str, record: dict[str, Any]) -> None:
        append_jsonl(self.sample_dir(sample_id) / "repair_trace.jsonl", record)

    def write_loop_result(self, sample_id: str, payload: dict[str, Any]) -> None:
        write_json(self.sample_dir(sample_id) / "loop_result.json", payload)

    def append_trial(self, sample_id: str, record: dict[str, Any]) -> None:
        """WanRepairTrialV2 一行（§20）。"""
        append_jsonl(self.sample_dir(sample_id) / "trials.jsonl", record)

    def append_resource_metrics(self, sample_id: str, record: dict[str, Any]) -> None:
        """每候选 GPU/耗时指标（§19）。"""
        append_jsonl(self.sample_dir(sample_id) / "resource_metrics.jsonl", record)

    def write_memory_status(self, payload: dict[str, Any]) -> None:
        """整轮只写一次的 memory 状态（§17/§28）。"""
        write_json(self.run_dir / "memory_status.json", payload)

    def write_owner(self, payload: dict[str, Any]) -> None:
        """vLLM 进程所有权 manifest（§19）。"""
        write_json(self.run_dir / "vllm.owner.json", payload)


def pending_samples(run_dir: str | Path, sample_ids: list[str]) -> list[str]:
    """resume：扫描 sample_status.json，返回尚未终止的样本（方案 §19）。"""

    artifacts = RunArtifacts(run_dir)
    return [sid for sid in sample_ids if not artifacts.is_complete(sid)]
