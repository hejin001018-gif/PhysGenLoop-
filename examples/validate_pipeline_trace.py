"""Independently validate a PAVG Critic execution trace.

Usage::

    python examples/validate_pipeline_trace.py outputs/video.trace.json \
        --require-sam2 --require-model-planner --fail-on-provider-fallback
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from pavg_critic.execution_trace import (
    TraceValidationPolicy,
    validate_trace,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="校验 PAVG Critic 节点 trace、融合算术和敏感信息边界",
    )
    parser.add_argument("trace_file", type=Path, help="待校验的 trace JSON")
    parser.add_argument(
        "--require-sam2",
        action="store_true",
        help="要求检测后端实际使用 SAM2",
    )
    parser.add_argument(
        "--require-model-planner",
        action="store_true",
        help="要求 Physics Planner 来源为模型而非模板降级",
    )
    parser.add_argument(
        "--fail-on-provider-fallback",
        action="store_true",
        help="任何 provider fallback 或 degraded 节点均视为失败",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = json.loads(args.trace_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"无法读取 trace: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(document, dict):
        print("无法读取 trace: JSON 根节点必须是对象", file=sys.stderr)
        return 2

    report = validate_trace(
        document,
        policy=TraceValidationPolicy(
            require_sam2=args.require_sam2,
            require_model_planner=args.require_model_planner,
            fail_on_provider_fallback=args.fail_on_provider_fallback,
        ),
    )
    for check in report.checks:
        if check.passed:
            marker = "PASS"
        elif check.level == "warning":
            marker = "WARN"
        else:
            marker = "FAIL"
        print(f"[{marker}] {check.code}: {check.message}")
    if report.passed:
        print(f"校验通过：{len(report.checks)} 项检查全部满足必需条件。")
        return 0
    failures = sum(
        not check.passed and check.level == "error" for check in report.checks
    )
    print(f"校验失败：{failures} 项必需检查未通过。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
