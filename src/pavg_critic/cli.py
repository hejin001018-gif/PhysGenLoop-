"""Physics Critic 的命令行入口。

CLI 同时覆盖真实视频和预计算观察值两种工作流，并且只负责参数解析、文件读写和调用
``PhysicsCritic``，不复制任何判定逻辑。这样命令行、Python API 与未来 HTTP 服务可
共享完全相同的流水线结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .pipeline import PhysicsCritic
from .schemas import CriticRequest, load_frame_states


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器；单独封装便于测试帮助信息和嵌入其他入口。"""

    parser = argparse.ArgumentParser(
        prog="physics-critic",
        description="Analyze a video or precomputed frame states for physical inconsistencies.",
    )
    parser.add_argument("--request", required=True, help="Path to a versioned request JSON file.")
    parser.add_argument(
        "--observations",
        help="Optional frame-state JSON; when present, video decoding is skipped.",
    )
    parser.add_argument("--config", help="Optional critic JSON or YAML configuration.")
    parser.add_argument("--floor-y", type=float, help="Support-surface y coordinate in pixels.")
    parser.add_argument("--output", help="Write JSON to this path instead of stdout.")
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include enriched tracks and detected events for debugging.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行一次 CLI 分析并返回进程退出码。

    ``argv=None`` 时 argparse 自动读取系统参数；测试或其他 Python 代码可以传入参数
    序列，避免修改全局 ``sys.argv``。
    """

    args = build_parser().parse_args(argv)
    # 先加载并验证所有外部 JSON，使格式错误在昂贵的视频解码之前失败。
    request = CriticRequest.from_json(args.request)
    config = load_config(args.config)
    observations = load_frame_states(args.observations) if args.observations else None
    artifacts = PhysicsCritic(config).analyze_detailed(
        request, observations=observations, floor_y=args.floor_y
    )
    # 默认只输出稳定报告；显式请求时才加入体积更大的逐帧调试产物。
    payload = artifacts.to_dict() if args.include_artifacts else artifacts.report.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2)
    if args.output:
        output_path = Path(args.output)
        # 输出目录可不存在，CLI 负责创建；已存在文件会被本次完整报告覆盖。
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - ``python -m critic`` 由 __main__ 覆盖测试
    raise SystemExit(main())
