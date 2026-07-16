"""VLM 通用物体检测器 — 不依赖颜色，可识别任意物体。

用一次 VLM 调用分析均匀采样关键帧，识别所有物体并估算其位置，将模型实际
观察到的关键帧 ``Detection`` 注入现有 PhysicsCritic Pipeline。

实现 :class:`ObjectDetector` Protocol，可直接替换 ``ColorBlobDetector``。
"""

from __future__ import annotations

import base64
from typing import Any, Sequence

from .interfaces import MultimodalStructuredModel
from .schemas import Detection

# ── VLM prompt ──────────────────────────────────────────────
_DETECTOR_SYSTEM = """\
You are an object detector for video frames. Identify every distinct \
physical object visible across the provided keyframes.

For each object:
- Give it a short snake_case name (e.g. "red_ball", "wooden_plank")
- At each keyframe, estimate its bounding box as percentages of frame \
  width and height (0=top/left, 100=bottom/right)
- If the object is not visible in a keyframe, set visible=false

Be precise but don't hallucinate objects that aren't clearly present."""

_DETECTOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objects", "frame_count"],
    "properties": {
        "frame_count": {"type": "integer"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "keyframe_positions"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "keyframe_positions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["frame_index", "visible", "x_pct", "y_pct", "w_pct", "h_pct"],
                            "properties": {
                                "frame_index": {"type": "integer"},
                                "visible": {"type": "boolean"},
                                "x_pct": {"type": "number"},
                                "y_pct": {"type": "number"},
                                "w_pct": {"type": "number"},
                                "h_pct": {"type": "number"},
                            },
                        },
                    },
                },
            },
        },
    },
}


class VLMObjectDetector:
    """用 VLM 识别任意物体并生成逐帧 ``Detection``。

    构造时执行一次 VLM 调用（多帧一起发送），识别视频中所有物体及其
    在关键帧中的大致位置。中间帧不生成虚构检测。

    用法::

        vlm = OpenAIChatModel(...)
        detector = VLMObjectDetector(vlm, "video.mp4", num_keyframes=8)
        critic = PhysicsCritic(config, detector=detector)
    """

    def __init__(
        self,
        vlm: MultimodalStructuredModel,
        video_path: str,
        num_keyframes: int = 8,
        jpeg_quality: int = 85,
    ) -> None:
        self._vlm = vlm
        self._cache: dict[int, list[Detection]] = {}
        self._width: int | None = None
        self._height: int | None = None
        self._fps: float = 30.0
        self._precompute(video_path, num_keyframes, jpeg_quality)

    # ── ObjectDetector protocol ──────────────────────────

    def detect(
        self, frame_image: Any, frame_index: int, timestamp_sec: float
    ) -> Sequence[Detection]:
        """返回预缓存的检测结果；首帧记录画面尺寸。"""
        if self._width is None:
            self._height = int(frame_image.shape[0])
            self._width = int(frame_image.shape[1])
        return self._cache.get(frame_index, ())

    # ── 内部实现 ─────────────────────────────────────────

    def _precompute(
        self, video_path: str, num_keyframes: int, jpeg_quality: int
    ) -> None:
        """一次 VLM 调用完成物体识别，仅缓存关键帧检测结果。

        不在帧间插值 —— 让 tracker 和事件检测器看到真实的帧间间隙，
        从而正确触发瞬移、消失等违规检测。
        """
        # 1. 提取关键帧，同时获取视频真实尺寸和 FPS
        frames, total_frames, width, height, fps = self._extract_keyframes(
            video_path, num_keyframes, jpeg_quality
        )
        if not frames or total_frames <= 0:
            return

        # 用视频真实尺寸做百分比→像素转换
        self._width = width if width > 0 else (self._width or 640)
        self._height = height if height > 0 else (self._height or 480)
        self._fps = fps if fps > 0 else 30.0

        # 2. VLM 识别物体及位置
        user_prompt = (
            f"Total video frames: {total_frames}. "
            f"Keyframes shown are at indices: {frames.indices}. "
            f"Detect all visible objects and their positions at each keyframe."
        )
        try:
            result = self._vlm.generate_json_with_images(
                system_prompt=_DETECTOR_SYSTEM,
                user_prompt=user_prompt,
                image_data_urls=frames.data_urls,
                schema=_DETECTOR_SCHEMA,
            )
        except Exception:
            return  # VLM 调用失败 → 空检测，由上层降级处理

        objects = result.get("objects", [])
        if not objects:
            return

        # 3. 关键帧 → 像素 Detection（仅缓存 VLM 实际观察到的关键帧）
        for obj in objects:
            name = str(obj.get("name", "object"))
            for pos in obj.get("keyframe_positions", []):
                if not pos.get("visible", True):
                    continue
                kf_idx = int(pos["frame_index"])
                bbox = self._percent_to_bbox(
                    float(pos["x_pct"]), float(pos["y_pct"]),
                    float(pos["w_pct"]), float(pos["h_pct"]),
                )
                # 使用视频真实 FPS 计算正确时间戳
                timestamp = kf_idx / self._fps
                detection = Detection(
                    frame=kf_idx,
                    timestamp_sec=timestamp,
                    object=name,
                    center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    bbox=bbox,
                    confidence=0.7,
                )
                self._cache.setdefault(kf_idx, []).append(detection)

    def _percent_to_bbox(
        self, x_pct: float, y_pct: float, w_pct: float, h_pct: float,
    ) -> tuple[float, float, float, float]:
        """百分比坐标 → 像素 bbox [x_min, y_min, x_max, y_max]."""
        w = float(self._width or 640)
        h = float(self._height or 480)
        x_center = (x_pct / 100.0) * w
        y_center = (y_pct / 100.0) * h
        half_w = (w_pct / 100.0) * w / 2.0
        half_h = (h_pct / 100.0) * h / 2.0
        return (
            max(0.0, x_center - half_w),
            max(0.0, y_center - half_h),
            min(w, x_center + half_w),
            min(h, y_center + half_h),
        )

    @staticmethod
    def _extract_keyframes(
        video_path: str,
        num_keyframes: int,
        jpeg_quality: int,
    ) -> tuple["_KeyframeResult", int, int, int, float]:
        """OpenCV 均匀采样关键帧，返回 (关键帧, 总帧数, 宽度, 高度, FPS)。"""
        try:
            import cv2
        except ImportError:
            return _KeyframeResult((), ()), 0, 0, 0, 30.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return _KeyframeResult((), ()), 0, 0, 0, 30.0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        if total <= 0:
            cap.release()
            return _KeyframeResult((), ()), 0, width, height, fps

        if num_keyframes >= total:
            indices = tuple(range(total))
        else:
            indices = tuple(
                int(i * (total - 1) / (num_keyframes - 1))
                for i in range(num_keyframes)
            )

        data_urls: list[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                data_urls.append(f"data:image/jpeg;base64,{b64}")

        cap.release()
        return _KeyframeResult(tuple(data_urls), indices), total, width, height, fps


class _KeyframeResult:
    """关键帧提取结果。"""
    __slots__ = ("data_urls", "indices")

    def __init__(self, data_urls: tuple[str, ...], indices: tuple[int, ...]):
        self.data_urls = data_urls
        self.indices = indices

    def __bool__(self) -> bool:
        return len(self.data_urls) > 0
