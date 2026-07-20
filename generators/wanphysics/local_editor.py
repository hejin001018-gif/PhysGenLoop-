"""ProPainter 局部视频编辑器，实现 LocalEditingExecutor.editor 协议。

接收 GeneratedCandidate + LocalEditTarget，提取需要修复的帧区域，
调用 ProPainter 完成背景修复后重新合成视频，返回新的 GeneratedCandidate。

ProPainter 本身需要帧序列 + mask，本包装负责：
  1. 用 SAM2 mask 信息或 bounding-box 构造 inpainting mask
  2. 将 mp4 分解为帧序列
  3. 调用 ProPainter 子进程
  4. 将输出帧序列重新编码为 mp4
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.contracts import LocalEditTarget


class ProPainterLocalEditor:
    """editor 对象，供 LocalEditingExecutor 注入使用。

    edit() 签名与 LocalEditingExecutor.execute() 中 hasattr(editor, 'edit') 分支一致。
    """

    def __init__(
        self,
        propainter_repo: str = "/root/ProPainter",
        python: str = "/root/PhysGenLoop-/envs/main/bin/python",
        output_root: str = "/root/PhysGenLoop-/outputs",
        fps: int = 24,
    ) -> None:
        self._repo = Path(propainter_repo)
        self._python = python
        self._output_root = Path(output_root)
        self._fps = fps

    def edit(
        self,
        *,
        candidate: GeneratedCandidate,
        target: LocalEditTarget,
        instruction: str,
        critic_report,
        physics_plan,
        seed: int,
    ) -> GeneratedCandidate:
        video_path = Path(candidate.video_path)
        digest = hashlib.sha256(
            f"{candidate.candidate_id}\0{seed}\0{instruction}".encode()
        ).hexdigest()[:12]
        candidate_id = f"propainter-{digest}"
        out_dir = self._output_root / candidate_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"{candidate_id}-v01.mp4"

        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            masks_dir = Path(tmp) / "masks"
            result_dir = Path(tmp) / "result"
            frames_dir.mkdir()
            masks_dir.mkdir()
            result_dir.mkdir()

            self._extract_frames(video_path, frames_dir)
            frame_count = len(list(frames_dir.glob("*.png")))
            self._build_masks(masks_dir, frame_count, target, video_path)
            self._run_propainter(frames_dir, masks_dir, result_dir)
            self._encode_video(result_dir, out_video)

        metadata = {
            **candidate.metadata,
            "backend": "propainter-local-edit",
            "source_candidate_id": candidate.candidate_id,
            "edit_instruction": instruction,
        }
        (out_dir / "prompt.txt").write_text(candidate.prompt, encoding="utf-8")
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return GeneratedCandidate(
            candidate_id=candidate_id,
            video_path=str(out_video),
            prompt=candidate.prompt,
            seed=seed,
            metadata=metadata,
        )

    def _extract_frames(self, video_path: Path, frames_dir: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frames_dir / f"{idx:05d}.png"), frame)
            idx += 1
        cap.release()

    def _build_masks(
        self,
        masks_dir: Path,
        frame_count: int,
        target: LocalEditTarget,
        video_path: Path,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"无法读取视频首帧：{video_path}")
        h, w = frame.shape[:2]

        # 优先用 mask_uri（SAM2 输出的 mask 图片路径）
        if target.mask_uri:
            ref_mask = cv2.imread(str(target.mask_uri), cv2.IMREAD_GRAYSCALE)
            if ref_mask is None:
                raise RuntimeError(f"无法读取 mask_uri：{target.mask_uri}")
        else:
            # 没有 mask_uri 时退化为全帧白色 mask（让 ProPainter 自行决定区域）
            ref_mask = np.ones((h, w), dtype=np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(ref_mask, kernel)

        # critical_frames 指定需要修复的帧索引；为空则修复全部帧
        active_frames: set[int] = set(target.critical_frames) if target.critical_frames else set(range(frame_count))
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(frame_count):
            mask = dilated if i in active_frames else empty_mask
            cv2.imwrite(str(masks_dir / f"{i:05d}.png"), mask)

    def _run_propainter(
        self, frames_dir: Path, masks_dir: Path, result_dir: Path
    ) -> None:
        subprocess.run(
            [
                self._python,
                str(self._repo / "inference_propainter.py"),
                "--video", str(frames_dir),
                "--mask", str(masks_dir),
                "--output", str(result_dir),
                "--fp16",
            ],
            cwd=str(self._repo),
            check=True,
        )

    def _encode_video(self, result_dir: Path, out_video: Path) -> None:
        frames = sorted(result_dir.rglob("*.png"))
        if not frames:
            raise RuntimeError(f"ProPainter 没有输出帧：{result_dir}")
        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (w, h),
        )
        for f in frames:
            writer.write(cv2.imread(str(f)))
        writer.release()
