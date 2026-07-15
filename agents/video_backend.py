"""视频生成后端抽象——本地 HunyuanVideo 与云 API 二选一。

Repair Agent 只负责产出 generator_request；实际调用视频生成由此模块承担。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class GenResult:
    ok: bool
    output_path: str
    elapsed_sec: float
    error: str | None = None


class VideoBackend(Protocol):
    name: str
    def generate(self, request: dict[str, Any]) -> GenResult: ...


class LocalHunyuanBackend:
    """本地部署 HunyuanVideo-1.5。首次调用加载模型并常驻。"""
    name = "hunyuan-local"

    def __init__(self):
        self._pipe = None

    def _lazy_load(self):
        if self._pipe is None:
            raise NotImplementedError(
                "HunyuanVideo-1.5 本地加载未接入。参考 "
                "https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5"
            )
        return self._pipe

    def generate(self, request: dict[str, Any]) -> GenResult:
        t0 = time.time()
        try:
            pipe = self._lazy_load()
            video = pipe(
                prompt=request["prompt"],
                seed=request["seed"],
                num_frames=request["num_frames"],
                num_inference_steps=request["num_inference_steps"],
            )
            Path(request["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            video.save(request["output_path"])  # type: ignore[attr-defined]
            return GenResult(True, request["output_path"], time.time() - t0)
        except Exception as e:
            return GenResult(False, request["output_path"], time.time() - t0, repr(e))


class ReplicateBackend:
    """Replicate 云 API。需要 REPLICATE_API_TOKEN。

    模型 slug 举例（占位，需按当日可用模型替换）：
        "tencent/hunyuan-video:<version>"
    """
    name = "replicate"

    def __init__(self, model_slug: str = "tencent/hunyuan-video"):
        self.model_slug = model_slug

    def generate(self, request: dict[str, Any]) -> GenResult:
        t0 = time.time()
        try:
            import replicate  # type: ignore
            import requests
            out = replicate.run(
                self.model_slug,
                input={
                    "prompt": request["prompt"],
                    "seed": request["seed"],
                    "num_frames": request["num_frames"],
                    "num_inference_steps": request["num_inference_steps"],
                    "resolution": request.get("resolution", "480p"),
                },
            )
            url = out if isinstance(out, str) else out[0]
            Path(request["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(request["output_path"], "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            return GenResult(True, request["output_path"], time.time() - t0)
        except Exception as e:
            return GenResult(False, request["output_path"], time.time() - t0, repr(e))


class FalBackend:
    """fal.ai 云 API。需要 FAL_KEY。"""
    name = "fal"

    def __init__(self, model_path: str = "fal-ai/hunyuan-video"):
        self.model_path = model_path

    def generate(self, request: dict[str, Any]) -> GenResult:
        t0 = time.time()
        try:
            import fal_client  # type: ignore
            import requests
            result = fal_client.subscribe(
                self.model_path,
                arguments={
                    "prompt": request["prompt"],
                    "seed": request["seed"],
                    "num_frames": request["num_frames"],
                    "num_inference_steps": request["num_inference_steps"],
                },
            )
            url = result["video"]["url"]
            Path(request["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(request["output_path"], "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            return GenResult(True, request["output_path"], time.time() - t0)
        except Exception as e:
            return GenResult(False, request["output_path"], time.time() - t0, repr(e))


class StubBackend:
    """离线开发。只落一个空文件，用于打通闭环。"""
    name = "stub"

    def generate(self, request: dict[str, Any]) -> GenResult:
        out = Path(request["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return GenResult(True, str(out), 0.0)


def make_backend(name: str | None = None) -> VideoBackend:
    name = name or os.environ.get("PAVG_VIDEO_BACKEND", "stub")
    if name == "hunyuan-local":
        return LocalHunyuanBackend()
    if name == "replicate":
        return ReplicateBackend()
    if name == "fal":
        return FalBackend()
    if name == "stub":
        return StubBackend()
    raise ValueError(f"unknown backend: {name}")
