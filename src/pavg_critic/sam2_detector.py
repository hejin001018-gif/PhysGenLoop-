"""VLM + SAM2 组合检测器 — 像素级精确 + 逐帧跟踪。

VLM 在首帧识别物体并给出提示点，SAM2 将分割掩码传播到全部帧，
每帧的 mask 转换为精确 bbox → :class:`Detection`。

实现 :class:`ObjectDetector` Protocol，精度等同于 ``ColorBlobDetector``，
但无需依赖颜色，适用于任意物体。
"""

from __future__ import annotations

import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

from .interfaces import MultimodalStructuredModel
from .schemas import Detection

# ── VLM 首帧物体识别 prompt ─────────────────────────────────
_VLM_IDENTIFY_SYSTEM = """\
You are identifying physical objects in the first frame of a video.

List every distinct physical object visible in this frame. For each:
- Give a stable snake_case name (e.g. "red_ball", "wooden_plank")
- Estimate its CENTER POINT as percentages of frame width and height
  (0=top/left, 100=bottom/right). Be as precise as possible.
- Include a short description of its appearance.

Only return objects you can clearly see — do not hallucinate."""

_VLM_IDENTIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objects"],
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "x_pct", "y_pct"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "x_pct": {"type": "number"},
                    "y_pct": {"type": "number"},
                },
            },
        },
    },
}


class SAM2ObjectDetector:
    """VLM 语义识别 + SAM2 像素跟踪 → 逐帧精确 Detection。

    VLM 只看首帧识别物体名称和中心点，SAM2 从这些点初始化并逐帧分割
    跟踪，每帧的 mask 转为精确 bbox。

    用法::

        vlm = OpenAIChatModel(...)
        detector = SAM2ObjectDetector(vlm, "video.mp4")
        critic = PhysicsCritic(config, detector=detector)
    """

    def __init__(
        self,
        vlm: MultimodalStructuredModel,
        video_path: str,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        model_ckpt: str = "sam2.1_hiera_base_plus.pt",
        jpeg_quality: int = 85,
    ) -> None:
        self._vlm = vlm
        self._cache: dict[int, list[Detection]] = {}
        self._object_names: dict[int, str] = {}
        self._width: int | None = None
        self._height: int | None = None
        self._precompute(video_path, model_cfg, model_ckpt, jpeg_quality)

    # ── ObjectDetector protocol ──────────────────────────

    def detect(
        self, frame_image: Any, frame_index: int, timestamp_sec: float
    ) -> Sequence[Detection]:
        """返回预缓存的精确 Detection（SAM2 分割结果）。"""
        if self._width is None and frame_image is not None:
            self._height = int(frame_image.shape[0])
            self._width = int(frame_image.shape[1])
        return self._cache.get(frame_index, ())

    # ── 内部实现 ─────────────────────────────────────────

    def _precompute(
        self,
        video_path: str,
        model_cfg: str,
        model_ckpt: str,
        jpeg_quality: int,
    ) -> None:
        """VLM 首帧识别 + SAM2 全视频传播。"""
        # 1. VLM 识别首帧物体
        first_frame_b64, width, height, fps = self._read_first_frame(
            video_path, jpeg_quality
        )
        if first_frame_b64 is None or width <= 0:
            raise ValueError(f"cannot decode first frame for SAM2: {video_path}")

        self._width = width
        self._height = height

        objects = self._identify_objects(first_frame_b64)
        if not objects:
            raise ValueError("VLM produced no object seeds for SAM2 tracking")

        # 2. SAM2 逐帧分割跟踪
        self._track_with_sam2(
            video_path,
            objects,
            model_cfg,
            model_ckpt,
            fps,
            jpeg_quality,
        )

    def _identify_objects(self, frame_b64: str) -> list[dict[str, Any]]:
        """VLM 识别首帧中的物体。"""
        result = self._vlm.generate_json_with_images(
            system_prompt=_VLM_IDENTIFY_SYSTEM,
            user_prompt="Identify all physical objects in this first frame.",
            image_data_urls=[frame_b64],
            schema=_VLM_IDENTIFY_SCHEMA,
        )
        objects = result.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("VLM object seed response must contain an objects array")
        return objects

    def _track_with_sam2(
        self,
        video_path: str,
        objects: list[dict[str, Any]],
        model_cfg: str,
        model_ckpt: str,
        fps: float,
        jpeg_quality: int,
    ) -> None:
        """SAM2 初始化 + 逐帧传播 + 掩码转 bbox。"""
        try:
            import numpy as np
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise RuntimeError("official SAM2 dependencies are not installed") from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = build_sam2_video_predictor(
            model_cfg, model_ckpt, device=device
        )
        with TemporaryDirectory(prefix="pavg_sam2_") as temporary:
            frame_dir = Path(temporary)
            frame_count = self._extract_video_frames(
                video_path,
                frame_dir,
                jpeg_quality,
            )
            inference_state = predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,
            )
            predictor.reset_state(inference_state)

            # 为每个 VLM 识别的物体在首帧添加正向点提示
            for obj_id, obj in enumerate(objects):
                name = str(obj.get("name", f"object_{obj_id}")).strip()
                if not name:
                    raise ValueError("SAM2 object seed name must not be empty")
                x_pct = float(obj["x_pct"])
                y_pct = float(obj["y_pct"])
                if not 0.0 <= x_pct <= 100.0 or not 0.0 <= y_pct <= 100.0:
                    raise ValueError("SAM2 object seed percentages must be in [0, 100]")
                self._object_names[obj_id] = name
                x_px = x_pct / 100.0 * self._width
                y_px = y_pct / 100.0 * self._height
                points = np.array([[x_px, y_px]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)  # 1 = 前景
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

            # 逐帧传播
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(inference_state):
                frame_dets: list[Detection] = []
                for i, obj_id in enumerate(out_obj_ids):
                    obj_id_int = int(obj_id)
                    mask = np.squeeze(
                        (out_mask_logits[i] > 0.0).cpu().numpy()
                    )
                    if mask.ndim != 2:
                        raise ValueError(
                            f"SAM2 mask must be 2D after squeeze, got {mask.shape}"
                        )
                    ys, xs = np.where(mask)
                    if len(xs) < 4:
                        continue  # 掩码太小 → 跳过
                    x_min, x_max = float(xs.min()), float(xs.max())
                    y_min, y_max = float(ys.min()), float(ys.max())
                    bbox = (x_min, y_min, x_max, y_max)
                    center = (
                        (x_min + x_max) / 2.0,
                        (y_min + y_max) / 2.0,
                    )
                    name = self._object_names.get(
                        obj_id_int,
                        f"object_{obj_id_int}",
                    )
                    timestamp = (
                        out_frame_idx / fps
                        if fps > 0
                        else float(out_frame_idx) / 30.0
                    )
                    detection = Detection(
                        frame=out_frame_idx,
                        timestamp_sec=timestamp,
                        object=name,
                        center=center,
                        bbox=bbox,
                        confidence=0.85,
                        track_id=f"sam2:{obj_id_int}",
                    )
                    frame_dets.append(detection)
                if frame_dets:
                    self._cache[out_frame_idx] = frame_dets
            predictor.reset_state(inference_state)
        if not self._cache:
            raise ValueError(
                f"SAM2 produced no detections across {frame_count} video frames"
            )

    @staticmethod
    def _extract_video_frames(
        video_path: str,
        destination: Path,
        jpeg_quality: int,
    ) -> int:
        """Decode a video to SAM2's portable numbered-JPEG input format."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to prepare SAM2 video frames") from exc
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video for SAM2: {video_path}")
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_path = destination / f"{frame_index:05d}.jpg"
                encoded = cv2.imwrite(
                    str(frame_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )
                if not encoded:
                    raise ValueError(f"cannot encode SAM2 frame {frame_index}")
                frame_index += 1
        finally:
            cap.release()
        if frame_index == 0:
            raise ValueError(f"video contains no decodable SAM2 frames: {video_path}")
        return frame_index

    @staticmethod
    def _read_first_frame(
        video_path: str, jpeg_quality: int
    ) -> tuple[str | None, int, int, float]:
        """读取首帧并编码为 base64 data URL。"""
        try:
            import cv2
        except ImportError:
            return None, 0, 0, 30.0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 0, 0, 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None, width, height, fps
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            return None, width, height, fps
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}", width, height, fps
