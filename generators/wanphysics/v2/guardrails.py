"""接受门禁：physics / confidence / coverage / semantic / quality（V2）。

修复 P0-4(接受门槛) / 方案 §18 / §27：

- ``shadow`` 模式：沿用 legacy ``decision==physical && physics_score>=阈值`` 决定是否
  停止，其余门只记录不改判——用于先观察分布再校准阈值。
- ``enforce`` 模式：physics/confidence/coverage/semantic/quality 全部达标才算接受。
  某个 scorer 不可用时返回明确的 ``*_unavailable``，绝不伪装成 accepted。

quality scorer 首版用纯 CPU 规则指标（黑帧/分辨率/模糊/亮度/帧间突变），semantic
scorer 默认 no-op（返回 None，enforce 时报 unavailable），不得把 physics_score 直接
当成 semantic_score（方案 §18）。cv2 缺失时 quality scorer 返回 None 而非报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

GUARDRAILS_SCHEMA_VERSION = "acceptance-gate/1.0"

MODE_SHADOW = "shadow"
MODE_ENFORCE = "enforce"


@dataclass(frozen=True)
class GateThresholds:
    physics_score: float = 0.80
    confidence: float = 0.60
    coverage: float = 0.50
    semantic_score: float = 0.85
    quality_score: float = 0.75

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "GateThresholds":
        raw = raw or {}
        return cls(
            physics_score=float(raw.get("physics_score", 0.80)),
            confidence=float(raw.get("confidence", 0.60)),
            coverage=float(raw.get("coverage", 0.50)),
            semantic_score=float(raw.get("semantic_score", 0.85)),
            quality_score=float(raw.get("quality_score", 0.75)),
        )


@dataclass(frozen=True)
class GateResult:
    mode: str
    accepted: bool
    physics_score: float | None
    confidence: float | None
    coverage: float | None
    semantic_score: float | None
    quality_score: float | None
    reasons: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    critic_degraded: bool = False
    schema_version: str = GUARDRAILS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "accepted": self.accepted,
            "scores": {
                "physics_score": self.physics_score,
                "confidence": self.confidence,
                "coverage": self.coverage,
                "semantic_score": self.semantic_score,
                "quality_score": self.quality_score,
            },
            "reasons": list(self.reasons),
            "unavailable": list(self.unavailable),
            "critic_degraded": self.critic_degraded,
        }


def evaluate_gate(
    *,
    report: Any,
    mode: str,
    thresholds: GateThresholds,
    semantic_score: float | None = None,
    quality_score: float | None = None,
    critic_degraded: bool = False,
    fail_on_degraded: bool = False,
) -> GateResult:
    """按模式裁决接受。shadow 只用 legacy 条件停机，其余记录；enforce 全门达标。"""

    decision = getattr(report, "decision", None)
    physics = float(getattr(report, "physics_score", 0.0))
    confidence = getattr(report, "confidence", None)
    coverage = getattr(report, "coverage", None)
    confidence = None if confidence is None else float(confidence)
    coverage = None if coverage is None else float(coverage)

    reasons: list[str] = []
    unavailable: list[str] = []

    legacy_ok = decision == "physical" and physics >= thresholds.physics_score

    if mode == MODE_SHADOW:
        accepted = legacy_ok
        if not legacy_ok:
            reasons.append("legacy_not_physical_or_below_physics_threshold")
        # 记录其余门是否满足，但不改判。
        for name, value in (("semantic_score", semantic_score), ("quality_score", quality_score)):
            if value is None:
                unavailable.append(name)
        return GateResult(
            mode=mode,
            accepted=accepted,
            physics_score=physics,
            confidence=confidence,
            coverage=coverage,
            semantic_score=semantic_score,
            quality_score=quality_score,
            reasons=tuple(reasons),
            unavailable=tuple(unavailable),
            critic_degraded=critic_degraded,
        )

    # enforce 模式
    if decision != "physical":
        reasons.append("decision_not_physical")
    if physics < thresholds.physics_score:
        reasons.append("physics_below_threshold")
    if confidence is None:
        unavailable.append("confidence")
    elif confidence < thresholds.confidence:
        reasons.append("confidence_below_threshold")
    if coverage is None:
        unavailable.append("coverage")
    elif coverage < thresholds.coverage:
        reasons.append("coverage_below_threshold")
    if semantic_score is None:
        unavailable.append("semantic_score")
    elif semantic_score < thresholds.semantic_score:
        reasons.append("semantic_below_threshold")
    if quality_score is None:
        unavailable.append("quality_score")
    elif quality_score < thresholds.quality_score:
        reasons.append("quality_below_threshold")
    if critic_degraded and fail_on_degraded:
        reasons.append("critic_degraded")

    accepted = not reasons and not unavailable
    return GateResult(
        mode=mode,
        accepted=accepted,
        physics_score=physics,
        confidence=confidence,
        coverage=coverage,
        semantic_score=semantic_score,
        quality_score=quality_score,
        reasons=tuple(reasons),
        unavailable=tuple(unavailable),
        critic_degraded=critic_degraded,
    )


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def _load_cv2():
    try:
        import cv2  # noqa: PLC0415

        return cv2
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class QualityReport:
    score: float | None
    metrics: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "metrics": dict(self.metrics),
            "available": self.available,
            "reason": self.reason,
        }


class CpuQualityScorer:
    """纯 CPU 规则质量评分：黑帧比例、分辨率、模糊、亮度、帧间突变。

    cv2 缺失或视频不可读时返回 available=False（不伪装成分数）。
    """

    def __init__(self, *, sample_frames: int = 16) -> None:
        self.sample_frames = sample_frames

    def score(self, video_path: str) -> QualityReport:
        cv2 = _load_cv2()
        if cv2 is None:
            return QualityReport(score=None, available=False, reason="cv2_unavailable")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return QualityReport(score=None, available=False, reason="video_unreadable")
        try:
            import numpy as np  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            cap.release()
            return QualityReport(score=None, available=False, reason="numpy_unavailable")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        step = max(1, total // max(1, self.sample_frames)) if total else 1

        read_ok = 0
        read_try = 0
        black_frames = 0
        blur_vals: list[float] = []
        brightness: list[float] = []
        prev_gray = None
        jumps: list[float] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                read_try += 1
                read_ok += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_lum = float(gray.mean())
                brightness.append(mean_lum)
                if mean_lum < 8.0:
                    black_frames += 1
                blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    jumps.append(float(np.abs(gray.astype("float32") - prev_gray.astype("float32")).mean()))
                prev_gray = gray
            idx += 1
        cap.release()

        if read_try == 0:
            return QualityReport(score=None, available=False, reason="no_frames_decoded")

        black_ratio = black_frames / read_try
        avg_blur = sum(blur_vals) / len(blur_vals) if blur_vals else 0.0
        avg_jump = sum(jumps) / len(jumps) if jumps else 0.0
        res_ok = 1.0 if (width >= 256 and height >= 256) else 0.5

        # 归一化到 [0,1]（启发式，shadow 阶段仅记录，不作已验证阈值）。
        blur_score = max(0.0, min(1.0, avg_blur / 200.0))
        jump_penalty = max(0.0, min(1.0, avg_jump / 80.0))
        score = max(
            0.0,
            min(
                1.0,
                0.40 * (1.0 - black_ratio)
                + 0.25 * blur_score
                + 0.20 * res_ok
                + 0.15 * (1.0 - jump_penalty),
            ),
        )
        return QualityReport(
            score=score,
            available=True,
            metrics={
                "black_ratio": black_ratio,
                "avg_blur": avg_blur,
                "avg_frame_jump": avg_jump,
                "width": width,
                "height": height,
                "frames_decoded": read_ok,
            },
        )


class NoOpSemanticScorer:
    """占位 semantic scorer：默认返回 None（不可用）。

    禁止把 physics_score 当 semantic_score。真正的 semantic scorer（独立结构化 VLM
    prompt）在获得 GPU 授权后单独接入；此处保证 enforce 模式会显式报
    ``semantic_score unavailable`` 而非误判 accepted。
    """

    available = False

    def score(self, *, prompt: str, video_path: str) -> float | None:  # noqa: D401
        return None
