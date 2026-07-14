"""轨迹分组、位置平滑与运动学特征提取。

本模块使用图像坐标：x 向右、y 向下，因此 ``velocity_y > 0`` 表示向下运动，
``velocity_y < 0`` 表示向上运动。速度和加速度使用真实时间戳差分，而不是默认相邻帧
间隔固定，从而兼容可变帧率或抽帧后的观察序列。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .config import TrajectoryConfig
from .schemas import FrameState, TrackSequence


class TrajectoryExtractor:
    """将离散 ``FrameState`` 补充为按身份组织的平滑运动轨迹。"""

    def __init__(self, config: TrajectoryConfig) -> None:
        """保存平滑窗口配置。"""

        self.config = config

    def extract(
        self, states: Iterable[FrameState], *, floor_y: float | None = None
    ) -> tuple[TrackSequence, ...]:
        """按 ``track_id`` 分组、排序，并计算运动学及地面相对量。

        缺少 ``track_id`` 的 Blender/人工观察值会回退到 object 名称；这适用于每类仅
        有一个物体的受控场景，多实例场景必须由上游提供显式稳定身份。
        """

        grouped: dict[str, list[FrameState]] = defaultdict(list)
        for state in states:
            track_id = state.track_id or state.object
            grouped[track_id].append(replace(state, track_id=track_id))

        tracks: list[TrackSequence] = []
        for track_id, raw_states in sorted(grouped.items()):
            # 同一轨迹先按帧号和时间戳排序，保证差分方向与视频播放方向一致。
            raw_states.sort(key=lambda item: (item.frame, item.timestamp_sec))
            enriched = self._enrich(raw_states, floor_y=floor_y)
            tracks.append(
                TrackSequence(track_id=track_id, object=enriched[0].object, states=tuple(enriched))
            )
        return tuple(tracks)

    def _enrich(
        self, states: list[FrameState], *, floor_y: float | None
    ) -> list[FrameState]:
        """对一条轨迹执行平滑、一阶/二阶差分和地面几何量计算。"""

        centers = self._smooth_centers(states)
        velocities: list[tuple[float, float] | None] = [None] * len(states)
        accelerations: list[tuple[float, float] | None] = [None] * len(states)

        # 速度只在可见状态之间计算，避免用跟踪器复制的缺失位置制造零速度假证据。
        previous_visible: int | None = None
        for index, state in enumerate(states):
            if not state.visible:
                continue
            if state.velocity is not None:
                # Blender 或外部系统提供的真值优先于二维差分估计。
                velocities[index] = state.velocity
            elif previous_visible is not None:
                dt = state.timestamp_sec - states[previous_visible].timestamp_sec
                if dt > 0:
                    velocities[index] = (
                        (centers[index][0] - centers[previous_visible][0]) / dt,
                        (centers[index][1] - centers[previous_visible][1]) / dt,
                    )
            previous_visible = index

        # 加速度是速度对时间的一阶差分；同样优先保留上游提供的真值。
        previous_velocity: int | None = None
        for index, state in enumerate(states):
            velocity = velocities[index]
            if state.acceleration is not None:
                accelerations[index] = state.acceleration
            elif velocity is not None and previous_velocity is not None:
                dt = state.timestamp_sec - states[previous_velocity].timestamp_sec
                prior = velocities[previous_velocity]
                if dt > 0 and prior is not None:
                    accelerations[index] = (
                        (velocity[0] - prior[0]) / dt,
                        (velocity[1] - prior[1]) / dt,
                    )
            if velocity is not None:
                previous_velocity = index

        result: list[FrameState] = []
        for index, state in enumerate(states):
            distance = state.distance_to_floor
            overlap = state.overlap_with_floor
            if floor_y is not None:
                # 有符号距离 = 地面 y - bbox 底边；负值即底边已经进入地面下方。
                if distance is None:
                    distance = floor_y - state.bbox[3]
                if overlap is None:
                    # 以 bbox 高度归一化穿过地面的深度，得到 [0, 1] 的近似重叠比例。
                    height = max(state.bbox[3] - state.bbox[1], 1.0)
                    overlap = min(1.0, max(0.0, (state.bbox[3] - floor_y) / height))
            result.append(
                replace(
                    state,
                    center=centers[index],
                    distance_to_floor=distance,
                    overlap_with_floor=overlap,
                    velocity=velocities[index],
                    acceleration=accelerations[index],
                )
            )
        return result

    def _smooth_centers(self, states: list[FrameState]) -> list[tuple[float, float]]:
        """使用居中滑动平均抑制检测抖动，同时保持缺失帧的原始预测位置。"""

        radius = self.config.smoothing_window // 2
        if radius == 0:
            return [state.center for state in states]
        result: list[tuple[float, float]] = []
        for index, state in enumerate(states):
            if not state.visible:
                # 缺失帧不参与相邻可见点平均，否则会将复制位置引入真实轨迹。
                result.append(state.center)
                continue
            nearby = [
                item.center
                for item in states[max(0, index - radius) : index + radius + 1]
                if item.visible
            ]
            result.append(
                (
                    sum(point[0] for point in nearby) / len(nearby),
                    sum(point[1] for point in nearby) / len(nearby),
                )
            )
        return result
