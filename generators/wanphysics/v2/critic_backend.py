"""V2 Critic 后端：无损评分子进程包装 + 完整报告持久化（V2）。

修复 P0-1 / P1-3：包装现有 ``agents/wanphysics/eval_step.py`` 评分子进程，但用
:mod:`critic_codec` 在父进程**无损**恢复 CriticReport，并持久化完整 critic_report.json
（含 violations / critical_frames / mask_uri / diagnostics / profile）；解析失败时保存
raw payload 并标记 roundtrip_failed，绝不静默清空 violations 继续跑。

与旧 ``Sam2VlmSubprocessCritic`` 并存、不修改它。真正起 vLLM/跑子进程依赖 GPU 授权，
默认不在导入或构造时触发；本模块的 payload→report 解码与产物组织逻辑可在 CPU 测试。
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .critic_codec import ReportDecodeResult, decode_report
from .critic_profiles import CriticProfile, profile_from_backend

CRITIC_BACKEND_SCHEMA_VERSION = "v2-critic-backend/1.0"


@dataclass(frozen=True)
class DecodedEvaluation:
    """一次 V2 评估的解码结果 + profile + 原始 payload。"""

    decode: ReportDecodeResult
    profile: CriticProfile
    detector_backend: str
    payload: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.decode.ok

    @property
    def report(self) -> Any:
        return self.decode.report

    def critic_report_document(self, *, candidate_id: str, video_path: str, prompt: str, physics_plan: dict[str, Any]) -> dict[str, Any]:
        """组织成修复方案 §12 的完整 critic_report.json 文档。"""

        report_dict = self.decode.raw_payload if not self.ok else self.report.to_dict()
        diagnostics = dict(report_dict.get("diagnostics", {}) or {})
        diagnostics.update(self.profile.to_dict())
        diagnostics["roundtrip_status"] = self.decode.status
        report_dict = {**report_dict, "diagnostics": diagnostics}
        return {
            "schema_version": "critic-report/2.0",
            "candidate_id": candidate_id,
            "video_path": Path(video_path).name,
            "prompt": prompt,
            "physics_plan": physics_plan,
            "report": report_dict,
            "codec": self.decode.to_status_dict(),
        }


def decode_eval_payload(payload: dict[str, Any], *, requested_profile: str = "sam2_seeded_rules") -> DecodedEvaluation:
    """把 eval_step 产出的 payload 解码为 DecodedEvaluation（纯函数，可 CPU 测试）。"""

    report_payload = dict(payload.get("report", {}) or {})
    detector_backend = str(payload.get("detector_backend", report_payload.get("diagnostics", {}).get("detector_backend", "unknown")))
    decode = decode_report(report_payload)
    profile = profile_from_backend(
        requested_profile,
        detector_backend,
        diagnostics=report_payload.get("diagnostics", {}),
    )
    return DecodedEvaluation(
        decode=decode,
        profile=profile,
        detector_backend=detector_backend,
        payload=dict(payload),
    )


class V2SubprocessCritic:
    """调用 eval_step 子进程评分，用 codec 无损恢复并持久化完整报告。

    generator/critic 的 vLLM 生命周期沿用注入的 lifecycle 回调（默认 no-op），本类
    不宽泛管理进程；GPU 交接由 resource_coordinator 负责。
    """

    def __init__(
        self,
        *,
        python: str,
        eval_step_path: str,
        requested_profile: str = "sam2_seeded_rules",
        prepare_hook: Callable[[], None] | None = None,
        release_hook: Callable[[], None] | None = None,
    ) -> None:
        self._python = python
        self._eval_step = eval_step_path
        self._requested_profile = requested_profile
        self._prepare_hook = prepare_hook
        self._release_hook = release_hook

    def evaluate_to_payload(self, candidate: Any, *, prompt: str, physics_plan: Any) -> dict[str, Any]:
        """起子进程评分，返回原始 payload（含 report dict + detector_backend）。"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as cf:
            candidate_json = cf.name
            try:
                plan_dict = physics_plan.to_dict() if hasattr(physics_plan, "to_dict") else {}
            except Exception:  # noqa: BLE001
                plan_dict = {}
            json.dump(
                {
                    "candidate_id": candidate.candidate_id,
                    "video_path": candidate.video_path,
                    "prompt": candidate.prompt,
                    "seed": candidate.seed,
                    "metadata": candidate.metadata,
                    "physics_plan": plan_dict,
                },
                cf,
                ensure_ascii=False,
            )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as ef:
            out_json = ef.name

        subprocess.run(
            [self._python, str(self._eval_step), "--candidate-json", candidate_json, "--out-json", out_json],
            check=True,
        )
        with open(out_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        Path(candidate_json).unlink(missing_ok=True)
        Path(out_json).unlink(missing_ok=True)
        return payload

    def evaluate(self, candidate: Any, *, prompt: str, physics_plan: Any) -> DecodedEvaluation:
        if self._prepare_hook is not None:
            self._prepare_hook()
        payload = self.evaluate_to_payload(candidate, prompt=prompt, physics_plan=physics_plan)
        if self._release_hook is not None:
            self._release_hook()
        return decode_eval_payload(payload, requested_profile=self._requested_profile)
