"""无损 CriticReport 跨进程编解码器（V2）。

修复 P0-1 / P0-2：现有 ``generators/wanphysics/sam2_vlm_critic.py`` 在父进程调用
``CriticReport.from_dict`` / ``Violation.from_dict``，而这些方法在
``src/pavg_critic/schemas.py`` 中并不存在，异常被 ``except`` 吞掉后 ``violations``
被静默置空，导致 critical_frames 和 mask 证据在 Repair 前全部丢失。

本模块**不修改** schemas.py，改为用现有 dataclass 构造器从 ``to_dict()`` 产出的
普通字典无损重建对象。解析失败时保留 raw payload 并显式标记状态，禁止静默清空。

约定：
- ``decode_report`` 返回 :class:`ReportDecodeResult`，永远不抛异常，让上层能记录
  ``critic_roundtrip_failed`` 并保存 raw，而不是丢字段继续跑。
- 只重建 Repair 决策真正需要的字段（violations / critical_frames / evidence /
  mask_uri / repair_instruction / evidence_bundles / diagnostics / model_versions /
  score_breakdown）。``graph_evaluation`` / ``node_results`` 属 Critic 内部产物，
  旧链路本就未跨进程传递，这里保持 ``None`` / ``()`` 且不视为字段丢失。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pavg_critic.schemas import (
    CriticReport,
    EvidenceBundle,
    SchemaError,
    Violation,
)

CODEC_SCHEMA_VERSION = "critic-report-codec/1.0"

# graph_evaluation / node_results 在旧跨进程链路中从不传递；这里显式声明为
# "非往返字段"，以免把它们的缺席误报成字段丢失。
_NON_ROUNDTRIP_FIELDS = ("graph_evaluation", "node_results")


@dataclass(frozen=True)
class ReportDecodeResult:
    """一次报告反序列化的结果，永不抛异常。"""

    report: CriticReport | None
    status: str  # "ok" | "roundtrip_failed"
    raw_payload: dict[str, Any]
    error: str | None = None
    recovered_fields: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.report is not None

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "codec_schema_version": CODEC_SCHEMA_VERSION,
            "roundtrip_status": self.status,
            "error": self.error,
            "recovered_fields": dict(self.recovered_fields),
        }


def _as_int_tuple(values: Any) -> tuple[int, ...]:
    if not values:
        return ()
    out: list[int] = []
    for item in values:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def _as_float_tuple(values: Any, length: int) -> tuple[float, ...]:
    return tuple(float(item) for item in values)[:length]


def decode_violation(raw: Mapping[str, Any]) -> Violation:
    """从字典无损重建 :class:`Violation`（含 evidence 里的 mask_uri/mask_uris）。"""

    evidence = dict(raw.get("evidence", {}) or {})
    return Violation(
        object=str(raw["object"]),
        category=str(raw["category"]),
        start_frame=int(raw["start_frame"]),
        peak_frame=int(raw["peak_frame"]),
        end_frame=int(raw["end_frame"]),
        critical_frames=_as_int_tuple(raw.get("critical_frames", ())),
        reason=str(raw.get("reason", "")),
        repair_instruction=str(raw.get("repair_instruction", "")),
        evidence=evidence,
    )


def decode_evidence_bundle(raw: Mapping[str, Any]) -> EvidenceBundle:
    """从字典无损重建 :class:`EvidenceBundle`。"""

    score = raw.get("score", None)
    return EvidenceBundle(
        family=str(raw["family"]),
        source=str(raw["source"]),
        status=str(raw["status"]),
        score=None if score is None else float(score),
        confidence=float(raw.get("confidence", 0.0)),
        coverage=float(raw.get("coverage", 0.0)),
        critical_frames=_as_int_tuple(raw.get("critical_frames", ())),
        details=dict(raw.get("details", {}) or {}),
    )


def decode_report(payload: Mapping[str, Any]) -> ReportDecodeResult:
    """把 ``CriticReport.to_dict()`` 产出的字典无损恢复为 :class:`CriticReport`。

    永不抛异常。任一子结构解析失败即整体判 ``roundtrip_failed`` 并保留 raw，
    绝不返回一个 violations 被静默清空的 "看起来正常" 的报告。
    """

    raw = dict(payload)
    recovered: dict[str, int] = {}
    try:
        raw_violations = raw.get("violations", ()) or ()
        violations = tuple(
            decode_violation(item) for item in raw_violations if isinstance(item, Mapping)
        )
        if len(violations) != len(list(raw_violations)):
            raise SchemaError(
                f"violation count mismatch: decoded {len(violations)} of {len(list(raw_violations))}"
            )
        recovered["violations"] = len(violations)
        recovered["critical_frames"] = sum(len(v.critical_frames) for v in violations)
        recovered["mask_uris"] = sum(
            1 for v in violations if (v.evidence or {}).get("mask_uri")
        )

        raw_bundles = raw.get("evidence_bundles", ()) or ()
        evidence_bundles = tuple(
            decode_evidence_bundle(item) for item in raw_bundles if isinstance(item, Mapping)
        )
        recovered["evidence_bundles"] = len(evidence_bundles)

        decision = raw.get("decision")
        is_physical = bool(raw.get("is_physical", decision == "physical"))

        report = CriticReport(
            is_physical=is_physical,
            physics_score=float(raw.get("physics_score", 0.0)),
            confidence=float(raw.get("confidence", 0.0)),
            violations=violations,
            decision=decision,
            coverage=float(raw.get("coverage", 1.0)),
            score_breakdown={
                str(k): float(v) for k, v in dict(raw.get("score_breakdown", {})).items()
            },
            diagnostics=dict(raw.get("diagnostics", {}) or {}),
            model_versions={
                str(k): str(v) for k, v in dict(raw.get("model_versions", {})).items()
            },
            evidence_bundles=evidence_bundles,
            schema_version=str(raw.get("schema_version", "2.0")),
        )
    except Exception as exc:  # noqa: BLE001 — 编解码器必须吞掉异常并显式上报
        return ReportDecodeResult(
            report=None,
            status="roundtrip_failed",
            raw_payload=raw,
            error=f"{type(exc).__name__}: {exc}",
            recovered_fields=recovered,
        )

    return ReportDecodeResult(
        report=report,
        status="ok",
        raw_payload=raw,
        error=None,
        recovered_fields=recovered,
    )


def encode_report(report: CriticReport) -> dict[str, Any]:
    """报告 → 字典（直接复用 schema 自带 ``to_dict``，集中出口便于审计）。"""

    return report.to_dict()
