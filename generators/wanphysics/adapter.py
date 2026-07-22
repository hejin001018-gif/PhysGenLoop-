"""把 WanGenerator（diffusers 本机推理）接入 physgenloop.interfaces.VideoGenerator 协议。

提供两种实现：
- WanPhysicsGenerator   常驻式，模型加载一次重复调用，适用于显存充足的环境。
- WanSubprocessGenerator 子进程式，每次 generate() 起一个独立子进程完成推理后退出，
                          进程退出即释放全部 CUDA 显存，适用于需要与 vLLM 等服务交替使用
                          同一张 GPU 的受限环境（如 40GB A100 同时运行 Wan2.2 + Qwen3-VL）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from pavg_critic.schemas import PhysicsPlan

from physgenloop.contracts import GeneratedCandidate

from .wan_generator import WanGenerator

# gen_step.py 的绝对路径，WanSubprocessGenerator 通过它发起子进程推理
_GEN_STEP = Path(__file__).parent.parent.parent / "agents" / "wanphysics" / "gen_step.py"


class WanPhysicsGenerator:
    """常驻加载一次 Wan2.2-TI2V-5B 模型，供 LoopController 重复调用生成候选。"""

    def __init__(
        self,
        model_path: str = "./models/wan2.2_ti2v_5b",
        device: str = "cuda",
        output_root: str = "./outputs",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        fps: int = 24,
        negative_prompt: str | None = None,
    ) -> None:
        self._wan = WanGenerator(model_path=model_path, device=device)
        self._output_root = Path(output_root)
        self._num_frames = num_frames
        self._height = height
        self._width = width
        self._fps = fps
        self._negative_prompt = negative_prompt

    def generate(
        self, *, prompt: str, physics_plan: PhysicsPlan, seed: int
    ) -> GeneratedCandidate:
        candidate_id = self._candidate_id(prompt, seed)
        candidate_dir = self._output_root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        video_path = candidate_dir / f"{candidate_id}-v01.mp4"

        self._wan.generate_video(
            prompt=prompt,
            output_path=str(video_path),
            num_frames=self._num_frames,
            height=self._height,
            width=self._width,
            fps=self._fps,
            seed=seed,
            negative_prompt=self._negative_prompt,
        )

        metadata = {
            "backend": "wan2.2-ti2v-5b",
            "is_real_video": True,
            "seed": seed,
            "num_frames": self._num_frames,
            "height": self._height,
            "width": self._width,
            "fps": self._fps,
        }
        (candidate_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (candidate_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return GeneratedCandidate(
            candidate_id=candidate_id,
            video_path=str(video_path),
            prompt=prompt,
            seed=seed,
            metadata=metadata,
        )

    @staticmethod
    def _candidate_id(prompt: str, seed: int) -> str:
        digest = hashlib.sha256(f"{prompt}\0{seed}".encode("utf-8")).hexdigest()[:6]
        return f"wan-{seed:04d}-{digest}"


class WanSubprocessGenerator:
    """子进程式 VideoGenerator，每次 generate() 在独立子进程中运行 Wan2.2 推理。

    子进程退出后 CUDA 上下文完全释放，主进程（LoopController）无任何显存占用，
    vLLM 等后续步骤可立即使用全部 GPU 显存。接口与 WanPhysicsGenerator 完全相同，
    可直接替换传入 LoopController。
    """

    def __init__(
        self,
        python: str,
        model_path: str = "/root/PhysGenLoop-/models/wan2.2_ti2v_5b",
        output_root: str = "/root/PhysGenLoop-/outputs",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        fps: int = 24,
        negative_prompt: str | None = None,
        gpu_id: str | int | None = None,
    ) -> None:
        self._python = python
        self._model_path = model_path
        self._output_root = output_root
        self._num_frames = num_frames
        self._height = height
        self._width = width
        self._fps = fps
        self._negative_prompt = negative_prompt
        # 双卡角色分工：把 Wan2.2 子进程固定到某张卡（默认由调用方指定 GPU0）。
        self._gpu_id = gpu_id

    def generate(
        self, *, prompt: str, physics_plan: PhysicsPlan, seed: int
    ) -> GeneratedCandidate:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_json = f.name

        cmd = [
            self._python, str(_GEN_STEP),
            "--prompt", prompt,
            "--seed", str(seed),
            "--output-root", self._output_root,
            "--model-path", self._model_path,
            "--num-frames", str(self._num_frames),
            "--height", str(self._height),
            "--width", str(self._width),
            "--fps", str(self._fps),
            "--out-json", out_json,
        ]
        if self._negative_prompt:
            cmd += ["--negative-prompt", self._negative_prompt]
        if self._gpu_id is not None and str(self._gpu_id) != "":
            cmd += ["--gpu-id", str(self._gpu_id)]

        # 通过环境变量传 GPU（gen_step 在 import torch 前读取 WAN_GPU_ID），双保险。
        env = None
        if self._gpu_id is not None and str(self._gpu_id) != "":
            import os as _os

            env = dict(_os.environ)
            env["WAN_GPU_ID"] = str(self._gpu_id)

        subprocess.run(cmd, check=True, env=env)

        with open(out_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        Path(out_json).unlink(missing_ok=True)

        return GeneratedCandidate(
            candidate_id=raw["candidate_id"],
            video_path=raw["video_path"],
            prompt=raw["prompt"],
            seed=raw["seed"],
            metadata=raw.get("metadata", {}),
        )
