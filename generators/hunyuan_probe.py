"""HunyuanVideo-1.5 探针。

对齐 框架骨干§十：再斩第 3 步。
目标：先跑单次 480p T2V，记录 (显存峰值 / 单条时长 / 输出路径)，供后续
决定 Best-of-K 与反馈轮数上限。

真实模型加载在 _load_pipeline() 中占位，需魔尊本地安装 HunyuanVideo-1.5 后接入。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class ProbeRequest:
    prompt: str
    seed: int = 42
    resolution: str = "480p"
    num_frames: int = 121
    num_inference_steps: int = 50
    image_path: str | None = None
    output_path: str = "outputs/probe.mp4"


@dataclass
class ProbeResult:
    ok: bool
    output_path: str
    elapsed_sec: float
    peak_vram_mb: float | None
    error: str | None = None
    meta: dict[str, Any] | None = None


def _load_pipeline():
    """加载 HunyuanVideo-1.5。真实实现由魔尊补：
        from hunyuan_video import HunyuanVideoPipeline
        return HunyuanVideoPipeline.from_pretrained(...)
    """
    raise NotImplementedError(
        "HunyuanVideo-1.5 未接入。参考 https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5"
    )


def _measure_vram_peak_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        return None
    return None


def run_probe(req: ProbeRequest, dry_run: bool = False) -> ProbeResult:
    Path(req.output_path).parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return ProbeResult(
            ok=True,
            output_path=req.output_path,
            elapsed_sec=0.0,
            peak_vram_mb=None,
            meta={"dry_run": True, "request": asdict(req)},
        )

    try:
        pipe = _load_pipeline()
    except NotImplementedError as e:
        return ProbeResult(
            ok=False,
            output_path=req.output_path,
            elapsed_sec=0.0,
            peak_vram_mb=None,
            error=str(e),
        )

    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t0 = time.time()
    try:
        video = pipe(
            prompt=req.prompt,
            seed=req.seed,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
        )
        # 保存视频接口由 HunyuanVideo 提供，此处占位
        video.save(req.output_path)  # type: ignore[attr-defined]
    except Exception as e:
        return ProbeResult(
            ok=False,
            output_path=req.output_path,
            elapsed_sec=time.time() - t0,
            peak_vram_mb=_measure_vram_peak_mb(),
            error=repr(e),
        )

    return ProbeResult(
        ok=True,
        output_path=req.output_path,
        elapsed_sec=time.time() - t0,
        peak_vram_mb=_measure_vram_peak_mb(),
        meta={"resolution": req.resolution, "num_frames": req.num_frames},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A red ball falls from a table.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", default="480p", choices=["480p", "720p"])
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="outputs/probe.mp4")
    ap.add_argument("--dry-run", action="store_true", help="仅打印请求，不加载模型")
    args = ap.parse_args()

    req = ProbeRequest(
        prompt=args.prompt,
        seed=args.seed,
        resolution=args.resolution,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        output_path=args.out,
    )
    res = run_probe(req, dry_run=args.dry_run)

    report_path = Path("outputs/probe_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(asdict(res), indent=2, ensure_ascii=False))
    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
