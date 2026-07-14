"""视频视觉前端。

当前内置实现是面向受控红球实验的 OpenCV HSV 基线，目标是打通架构和产生可解释
轨迹，而不是替代通用检测/分割模型。OpenCV 和 NumPy 采用延迟导入，因此只处理
Blender 真值 JSON 的用户不需要安装视频依赖。
"""

from __future__ import annotations

from typing import Any, Sequence

from .config import DetectorConfig
from .schemas import Detection


class ColorBlobDetector:
    """使用两段 HSV 红色区间检测连通色块。

    输出的每个轮廓独立成为 ``Detection``。多物体身份不在本类中推断，而交由后续
    Tracker 处理，以保持“单帧观测”和“跨帧身份”职责分离。
    """

    def __init__(self, config: DetectorConfig) -> None:
        """保存已经过 ``CriticConfig.validate`` 校验的检测参数。"""

        self.config = config

    def detect(
        self, frame_image: Any, frame_index: int, timestamp_sec: float
    ) -> Sequence[Detection]:
        """检测一帧中的目标色块并按置信度从高到低返回。

        Args:
            frame_image: OpenCV BGR 图像。
            frame_index: 从 0 开始的帧号。
            timestamp_sec: 由视频 FPS 换算得到的秒级时间戳。
        """

        # 延迟导入让 schema、规则和观察值模式保持零第三方依赖。
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError(
                "Video analysis requires optional dependencies; run "
                "`pip install -e .[video]`."
            ) from exc

        # 红色色相跨越 HSV 环的 0 点，必须合并低色相与高色相两个掩码。
        hsv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        mask_1 = cv2.inRange(
            hsv, np.array(self.config.hsv_lower), np.array(self.config.hsv_upper)
        )
        mask_2 = cv2.inRange(
            hsv, np.array(self.config.hsv_lower_2), np.array(self.config.hsv_upper_2)
        )
        mask = cv2.bitwise_or(mask_1, mask_2)
        # 3x3 开运算先腐蚀后膨胀，用于移除孤立的压缩噪声和红色像素点。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        frame_area = max(float(frame_image.shape[0] * frame_image.shape[1]), 1.0)
        for contour in contours:
            # 面积阈值过滤微小噪声；剩余轮廓转换为统一的 xyxy 包围框。
            area = float(cv2.contourArea(contour))
            if area < self.config.min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    frame=frame_index,
                    timestamp_sec=timestamp_sec,
                    object=self.config.object_label,
                    center=(x + width / 2.0, y + height / 2.0),
                    bbox=(float(x), float(y), float(x + width), float(y + height)),
                    # 基线没有分类网络概率，因此用相对面积构造可复现的启发式置信度。
                    confidence=min(1.0, 0.5 + area / (frame_area * 0.02)),
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)
