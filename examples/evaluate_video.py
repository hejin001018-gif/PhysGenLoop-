"""端到端视频物理违规评估 — 完整 PAVG PhysicsCritic Pipeline。

用法::

    # 无 prompt — VLM 通用检测 + 规则引擎 + VLM 复核
    python examples/evaluate_video.py --video 2n.mp4

    # 有 prompt — VLM 检测 + Planner → Pipeline → VLM 复核
    python examples/evaluate_video.py --video 1n.mp4 --prompt "a red ball falls and bounces"

    # 指定 VLM 复核帧窗口 + 保存结果
    python examples/evaluate_video.py --video 1n.mp4 --pre-frames 5 --post-frames 5 -o result.json -v

配置（``.env`` 或环境变量）::

    API_KEY=sk-your-key
    BASE_URL=https://api.openai.com/v1    # 兼容官方和中转站
    VLM_MODEL=gpt-4o                      # 视觉模型（检测 + 复核）
    TEXT_MODEL=gpt-4o-mini                # 文本模型（Planner + PQSG）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── .env 加载 ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pavg_critic import (
    CriticConfig,
    CriticRequest,
    EvidenceGroundedVLMVerifier,
    OpenAIChatModel,
    PhysicsCritic,
    SAM2ObjectDetector,
    VLMObjectDetector,
)
from pavg_critic.api_models import ModelAPIError
from pavg_critic.schemas import SchemaError


# ── 配置 ────────────────────────────────────────────────────
def load_config() -> dict[str, str]:
    """从环境变量 / .env 加载 API 配置。"""
    api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise ValueError(
            "Set API_KEY (or OPENAI_API_KEY) in .env file or environment"
        )
    base_url = os.getenv(
        "BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    vlm_model = os.getenv("VLM_MODEL", os.getenv("OPENAI_MODEL", ""))
    text_model = os.getenv("TEXT_MODEL", os.getenv("OPENAI_MODEL", ""))

    if not vlm_model:
        raise ValueError(
            "Set VLM_MODEL (or OPENAI_MODEL) in .env file or environment"
        )
    return {
        "api_key": api_key,
        "base_url": base_url,
        "vlm_model": vlm_model,
        "text_model": text_model or vlm_model,
    }


# ── 日志辅助 ────────────────────────────────────────────────
def _print_header(title: str) -> None:
    print(f"\n{'═' * 55}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{'═' * 55}", file=sys.stderr)


def _print_kv(key: str, value: str) -> None:
    print(f"  {key}: {value}", file=sys.stderr)


# ── 视频探测 ────────────────────────────────────────────────
def _probe_video_meta(video_path: str) -> tuple[int, int, int]:
    """探测视频元数据，返回 (总帧数, 宽度, 高度)。失败时返回默认值。"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if total > 0 and width > 0 and height > 0:
                return total, width, height
    except Exception:
        pass
    return 240, 640, 480  # 默认 640×480 @ 8s


def _resolve_sam2_checkpoint() -> Path:
    """Resolve the explicit or repository-frozen SAM2.1 checkpoint."""

    explicit = os.getenv("SAM2_CHECKPOINT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    frozen = Path("evaluation/external/models/sam2.1_hiera_base_plus.pt")
    if frozen.is_file():
        return frozen.resolve()
    legacy = Path("sam2.1_hiera_base_plus.pt")
    return legacy.resolve() if legacy.is_file() else frozen.resolve()


def _configure_sparse_vlm_fallback(
    config: CriticConfig,
    *,
    total_frames: int,
    width: int,
    height: int,
    num_keyframes: int,
) -> CriticConfig:
    """Make sparse keyframe observations explicit instead of faking continuity."""

    diagonal = (width**2 + height**2) ** 0.5
    keyframe_spacing = max(1, total_frames // max(num_keyframes - 1, 1))
    return replace(
        config,
        tracker=replace(
            config.tracker,
            max_match_distance_px=max(
                config.tracker.max_match_distance_px,
                int(diagonal * 0.25),
            ),
            max_missed_frames=max(
                config.tracker.max_missed_frames,
                int(keyframe_spacing * 2),
            ),
        ),
        events=replace(
            config.events,
            min_disappearance_frames=max(
                config.events.min_disappearance_frames,
                int(keyframe_spacing * 2),
            ),
            contact_tolerance_px=max(
                config.events.contact_tolerance_px,
                height * 0.05,
            ),
            penetration_tolerance_px=max(
                config.events.penetration_tolerance_px,
                height * 0.05,
            ),
        ),
        rules=replace(
            config.rules,
            enabled=tuple(
                rule
                for rule in config.rules.enabled
                if rule != "object_disappearance"
            ),
        ),
    )


# ── 核心 ────────────────────────────────────────────────────
def evaluate_video(
    video_path: str,
    *,
    prompt: str | None = None,
    config_path: str | None = None,
    cfg: dict[str, str] | None = None,
    pre_frames: int | None = None,
    post_frames: int | None = None,
    num_keyframes: int = 8,
    verbose: bool = False,
) -> dict[str, Any]:
    """运行完整 PhysicsCritic Pipeline，使用 VLM 通用物体检测。"""

    api_cfg = cfg or load_config()

    # ── 1. 创建模型 ─────────────────────────────────────
    vlm = OpenAIChatModel(
        api_key=api_cfg["api_key"],
        model=api_cfg["vlm_model"],
        base_url=api_cfg["base_url"],
    )
    text_model = OpenAIChatModel(
        api_key=api_cfg["api_key"],
        model=api_cfg["text_model"],
        base_url=api_cfg["base_url"],
        strict_json_schema=True,
    )

    # ── 2. 配置 + VLM 通用检测器 ────────────────────────
    critic_config = CriticConfig()
    if config_path:
        from pavg_critic.config import load_config as load_critic_config
        critic_config = load_critic_config(config_path)
    if pre_frames is not None or post_frames is not None:
        critic_config = replace(
            critic_config,
            temporal=replace(
                critic_config.temporal,
                pre_context_frames=(
                    pre_frames
                    if pre_frames is not None
                    else critic_config.temporal.pre_context_frames
                ),
                post_context_frames=(
                    post_frames
                    if post_frames is not None
                    else critic_config.temporal.post_context_frames
                ),
            ),
        )

    # ── 选择检测器：SAM2 优先 → VLM 降级 ──────────────
    total_frames, vw, vh = _probe_video_meta(video_path)
    detector = None
    sam2_used = False
    sam2_fallback_error: Exception | None = None
    try:
        detector = SAM2ObjectDetector(
            vlm,
            video_path,
            model_ckpt=str(_resolve_sam2_checkpoint()),
            prompt=prompt or "",
        )
        sam2_used = True
        if verbose:
            print(f"SAM2 像素级跟踪 ({vw}×{vh})...", file=sys.stderr)
    except Exception as exc:
        sam2_fallback_error = exc
        # SAM2 不可用（未安装 / 无 GPU / 初始化失败）→ VLM 降级
        if verbose:
            print(f"SAM2 不可用，降级到 VLM 检测器...", file=sys.stderr)
        detector = VLMObjectDetector(vlm, video_path, num_keyframes=num_keyframes)

    # VLM 检测器百分比坐标精度低，需放宽阈值；SAM2 像素级精度用原生配置
    if not sam2_used:
        kf_spacing = max(1, total_frames // max(num_keyframes - 1, 1))
        if verbose:
            print(
                f"视频: {total_frames} 帧, {vw}×{vh}, "
                f"关键帧间隔 ~{kf_spacing} 帧",
                file=sys.stderr,
            )
            print(f"VLM 通用检测 ({num_keyframes} 帧)...", file=sys.stderr)
        critic_config = _configure_sparse_vlm_fallback(
            critic_config,
            total_frames=total_frames,
            width=vw,
            height=vh,
            num_keyframes=num_keyframes,
        )

    # ── 3. 构建 Critic ──────────────────────────────────
    verifier = EvidenceGroundedVLMVerifier(vlm, model_name=vlm.model)
    critic = PhysicsCritic(
        critic_config,
        detector=detector,
        planner_model=text_model,
        question_model=text_model,
        vlm_verifier=verifier,
    )

    # ── 4. 运行 Pipeline ────────────────────────────────
    request = CriticRequest(video_path=video_path, prompt=prompt or "")

    if verbose:
        _print_header(f"PAVG Pipeline: {Path(video_path).name}")
        _print_kv("Prompt", prompt or "(无 — 规则检测)")
        _print_kv("检测器", "SAM2 像素级" if sam2_used else f"VLM ({num_keyframes} 帧)")
        _print_kv("VLM 模型", vlm.model)
        _print_kv("文本模型", text_model.model)
        _print_kv("分辨率", f"{vw}×{vh}, {total_frames} 帧")
        print("─" * 50, file=sys.stderr)

        print("  [1/6] PhysicsPlan 解析...", file=sys.stderr)

    try:
        artifacts = critic.analyze_detailed(request)
    except ModelAPIError as exc:
        if verbose:
            print(f"  ⚠ 模型 API 失败 → 降级为纯规则基线", file=sys.stderr)
        artifacts = PhysicsCritic(critic_config, detector=detector).analyze_detailed(request)

    report = artifacts.report

    if verbose:
        # Planner 结果
        plan = artifacts.resolved_request.physics_plan if artifacts.resolved_request else None
        if plan and (plan.objects or plan.expected_events):
            _print_kv("  Plan 对象", ", ".join(plan.objects) or "(无)")
            _print_kv("  Plan 事件", ", ".join(plan.expected_events) or "(无)")
            meta = plan.planner_metadata
            _print_kv("  Plan 来源", f"{meta.source} (置信度 {meta.confidence:.2f})")
        else:
            print("  └─ 无显式物理计划（空 prompt）", file=sys.stderr)

        # 检测 + 跟踪
        print("  [2/6] 目标检测 + 跟踪...", file=sys.stderr)
        tracks = artifacts.tracks
        if tracks:
            print(f"  ├─ 检测到 {len(tracks)} 条轨迹:", file=sys.stderr)
            for t in tracks:
                visible_frames = sum(1 for s in t.states if s.visible)
                print(
                    f"  │   {t.object} (id={t.track_id}): "
                    f"{len(t.states)} 帧, {visible_frames} 可见",
                    file=sys.stderr,
                )
        else:
            print("  └─ 未检测到任何物体", file=sys.stderr)

        # 事件
        print("  [3/6] 事件检测...", file=sys.stderr)
        events = artifacts.events
        if events:
            event_types = {}
            for e in events:
                event_types[e.event_type] = event_types.get(e.event_type, 0) + 1
            print(f"  ├─ 共 {len(events)} 个事件:", file=sys.stderr)
            for etype, count in sorted(event_types.items()):
                print(f"  │   {etype}: {count}", file=sys.stderr)
        else:
            print("  └─ 无事件", file=sys.stderr)

        # 违规候选
        print("  [4/6] 规则引擎 + 证据家族...", file=sys.stderr)
        if report.violations:
            print(f"  ├─ 违规候选: {len(report.violations)} 个", file=sys.stderr)
            for v in report.violations:
                print(
                    f"  │   [{v.category}] {v.object} "
                    f"(帧 {v.start_frame}-{v.end_frame})",
                    file=sys.stderr,
                )
        else:
            print("  ├─ 规则引擎: 无违规候选", file=sys.stderr)

        # 证据家族状态
        for b in report.evidence_bundles:
            score_str = f"{b.score:.3f}" if b.score is not None else "N/A"
            print(
                f"  │   {b.family:12s} 状态={b.status:14s}  "
                f"分数={score_str}  覆盖={b.coverage:.2f}",
                file=sys.stderr,
            )

        # VLM 复核
        print("  [5/6] VLM 关键帧复核...", file=sys.stderr)
        vlm_reviews = [
            b for b in report.evidence_bundles if b.family == "vlm"
        ]
        if vlm_reviews and vlm_reviews[0].status == "available":
            print(f"  ├─ VLM 复核完成 (模型: {vlm.model})", file=sys.stderr)
        else:
            print("  ├─ 无需 VLM 复核 (无候选/无关键帧)", file=sys.stderr)

        # 融合
        print("  [6/6] 证据融合...", file=sys.stderr)
        print(
            f"  └─ 决策={report.decision}  "
            f"物理分={report.physics_score:.3f}  "
            f"置信度={report.confidence:.3f}  "
            f"覆盖={report.coverage:.3f}",
            file=sys.stderr,
        )

        # Provider 故障
        if "provider_failures" in report.diagnostics:
            failures = report.diagnostics["provider_failures"]
            print(f"  ⚠ {len(failures)} 个 provider 故障 (已降级处理)", file=sys.stderr)

        print("=" * 50, file=sys.stderr)

    # ── 5. 收集输出 ─────────────────────────────────────
    violations = [
        {
            "object": v.object,
            "category": v.category,
            "start_frame": v.start_frame,
            "peak_frame": v.peak_frame,
            "end_frame": v.end_frame,
            "reason": v.reason,
            "repair_instruction": v.repair_instruction,
            "critical_frames": list(v.critical_frames),
        }
        for v in report.violations
    ]

    planner_meta = (
        artifacts.resolved_request.physics_plan.planner_metadata
        if artifacts.resolved_request
        else None
    )

    output: dict[str, Any] = {
        "video_path": str(Path(video_path).resolve()),
        "prompt": prompt or None,
        "decision": report.decision,
        "is_physical": report.is_physical,
        "physics_score": round(report.physics_score, 4),
        "confidence": round(report.confidence, 4),
        "coverage": round(report.coverage, 4),
        "violations_count": len(violations),
        "violations": violations,
        "score_breakdown": report.score_breakdown,
        "planner": {
            "source": planner_meta.source if planner_meta else "unknown",
            "confidence": planner_meta.confidence if planner_meta else 0.0,
            "model": planner_meta.model if planner_meta else None,
        },
        "model_versions": {"vlm": vlm.model, "text": text_model.model},
        "detector": {
            "backend": "sam2" if sam2_used else "sparse_vlm_fallback",
            "sam2_used": sam2_used,
            "fallback_error": (
                None
                if sam2_fallback_error is None
                else {
                    "type": type(sam2_fallback_error).__name__,
                    "message": str(sam2_fallback_error)[:300],
                }
            ),
        },
        "evidence_families": [
            {
                "family": b.family,
                "status": b.status,
                "score": b.score,
                "coverage": round(b.coverage, 4),
            }
            for b in report.evidence_bundles
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    diagnostics = report.diagnostics
    if "provider_failures" in diagnostics:
        output["provider_failures"] = [
            {
                "stage": f["stage"],
                "error": f["error_type"],
                **(
                    {"detail": str(f.get("message", ""))[:300]}
                    if f.get("error_type")
                    in {"QuestionGraphError", "SchemaError"}
                    else {}
                ),
            }
            for f in diagnostics["provider_failures"]
        ]

    tracks = artifacts.tracks
    if tracks:
        output["detected_objects"] = [
            {"track_id": t.track_id, "object": t.object, "frames": len(t.states)}
            for t in tracks
        ]
        output["total_events"] = len(artifacts.events)

    return output


# ── CLI ─────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PAVG 视频物理违规评估（VLM 通用检测 + 完整 Pipeline）",
    )
    parser.add_argument(
        "--video", required=True, type=Path, help="视频文件路径"
    )
    parser.add_argument(
        "--prompt", default=None,
        help="可选的文字描述；不提供时使用纯规则/VLM 检测",
    )
    parser.add_argument(
        "--config", default=None, type=Path,
        help="Critic YAML/JSON 配置文件（可覆盖规则阈值等）",
    )
    parser.add_argument(
        "--pre-frames", type=int, default=None,
        help="VLM 复核时违规前截取帧数 (默认: 3)",
    )
    parser.add_argument(
        "--post-frames", type=int, default=None,
        help="VLM 复核时违规后截取帧数 (默认: 3)",
    )
    parser.add_argument(
        "--keyframes", type=int, default=8,
        help="VLM 检测时采样的关键帧数 (默认: 8)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="输出进度信息到 stderr"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    video_path = str(args.video.resolve())
    if not args.video.is_file():
        print(f"错误: 视频文件不存在: {video_path}", file=sys.stderr)
        return 1

    try:
        api_cfg = load_config()
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    try:
        result = evaluate_video(
            video_path,
            prompt=args.prompt,
            config_path=str(args.config) if args.config else None,
            cfg=api_cfg,
            pre_frames=args.pre_frames,
            post_frames=args.post_frames,
            num_keyframes=args.keyframes,
            verbose=args.verbose,
        )
    except (ModelAPIError, SchemaError, RuntimeError) as exc:
        print(f"Pipeline 错误: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"未预期错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        if args.verbose:
            print(f"结果已写入: {args.output}", file=sys.stderr)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
