"""用于基线视频路径的轻量质心跟踪器。

算法按物体类别和中心点欧氏距离进行贪心匹配，适合目标数量少、相互遮挡有限的受控
场景。它不是通用 MOT 算法；复杂场景应实现更可靠的外部跟踪器，并直接导出统一的
``FrameState``。未匹配轨迹会短暂保留并产生 ``visible=False`` 状态，为物体恒存规则
提供连续的缺失证据。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Iterable, Sequence

from .config import TrackerConfig
from .schemas import Detection, FrameState


@dataclass
class _ActiveTrack:
    """跟踪器内部的可变状态，不会越过模块边界暴露给调用方。"""

    track_id: str
    object: str
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    confidence: float
    missed: int = 0


class CentroidTracker:
    """按类别约束的贪心质心跟踪器。"""

    def __init__(self, config: TrackerConfig) -> None:
        """初始化跟踪阈值；每次 ``track*`` 调用都会创建全新的轨迹状态。"""

        self.config = config

    def track(self, frames: Iterable[Sequence[Detection]]) -> tuple[FrameState, ...]:
        """跟踪每帧均至少有一个检测结果的序列。

        空检测帧本身不携带帧号和时间戳，无法产生缺失状态；包含空帧的正常视频流程
        应使用 ``track_timed``，流水线也始终调用后者。
        """

        active: dict[str, _ActiveTrack] = {}
        next_id = 0
        states: list[FrameState] = []

        for detections in frames:
            detections = list(detections)
            if not detections and not active:
                continue
            frame = detections[0].frame if detections else None
            timestamp = detections[0].timestamp_sec if detections else None
            if frame is None or timestamp is None:
                raise ValueError(
                    "Empty detection frames must carry timing through track_timed(); "
                    "the pipeline uses that method."
                )
            frame_states, next_id = self._step(
                active, detections, frame, timestamp, next_id
            )
            states.extend(frame_states)
        return tuple(states)

    def track_timed(
        self, frames: Iterable[tuple[int, float, Sequence[Detection]]]
    ) -> tuple[FrameState, ...]:
        """跟踪显式携带 ``(frame, timestamp, detections)`` 的完整视频序列。"""

        active: dict[str, _ActiveTrack] = {}
        next_id = 0
        states: list[FrameState] = []
        for frame, timestamp, detections in frames:
            frame_states, next_id = self._step(
                active, list(detections), frame, timestamp, next_id
            )
            states.extend(frame_states)
        return tuple(states)

    def _step(
        self,
        active: dict[str, _ActiveTrack],
        detections: list[Detection],
        frame: int,
        timestamp: float,
        next_id: int,
    ) -> tuple[list[FrameState], int]:
        """推进一帧跟踪状态，并返回该帧状态与下一个可用的轨迹编号。"""

        assignments: dict[str, int] = {}
        available = set(range(len(detections)))

        # 跟踪型后端已经维护的稳定身份优先于二次质心匹配。
        explicit_ids: set[str] = set()
        for index, detection in enumerate(detections):
            if detection.track_id is None:
                continue
            if detection.track_id in explicit_ids:
                raise ValueError(
                    "duplicate explicit detection track_id in one frame: "
                    f"{detection.track_id}"
                )
            explicit_ids.add(detection.track_id)
            if detection.track_id in active:
                track = active[detection.track_id]
                if track.object != detection.object:
                    raise ValueError(
                        f"explicit track_id {detection.track_id!r} changed object "
                        f"from {track.object!r} to {detection.object!r}"
                    )
                assignments[detection.track_id] = index
                available.remove(index)

        # 枚举同类别轨迹-检测组合。不同类别绝不匹配，避免身份在类别间跳变。
        candidates: list[tuple[float, str, int]] = []
        for track_id, track in active.items():
            for index, detection in enumerate(detections):
                if (
                    track_id not in assignments
                    and index in available
                    and detection.track_id is None
                    and detection.object == track.object
                ):
                    candidates.append((dist(track.center, detection.center), track_id, index))
        # 全局按距离从小到大贪心选择一对一匹配；每个检测和轨迹最多使用一次。
        for distance, track_id, index in sorted(candidates):
            if distance > self.config.max_match_distance_px:
                continue
            if track_id in assignments or index not in available:
                continue
            assignments[track_id] = index
            available.remove(index)

        result: list[FrameState] = []
        for track_id in list(active):
            track = active[track_id]
            if track_id in assignments:
                # 命中检测后使用真实观测更新中心、框和置信度，并清零丢失计数。
                detection = detections[assignments[track_id]]
                track.center = detection.center
                track.bbox = detection.bbox
                track.confidence = detection.confidence
                track.missed = 0
                result.append(_state(detection, track_id))
            else:
                # 短时丢失期间保留最后位置，并按 0.8^missed 衰减观测置信度。
                track.missed += 1
                if track.missed <= self.config.max_missed_frames:
                    result.append(
                        FrameState(
                            frame=frame,
                            timestamp_sec=timestamp,
                            object=track.object,
                            center=track.center,
                            bbox=track.bbox,
                            visible=False,
                            confidence=max(0.0, track.confidence * (0.8**track.missed)),
                            track_id=track_id,
                        )
                    )
                else:
                    # 超过容忍窗口后终止轨迹；之后重新出现的目标会获得新 track_id。
                    del active[track_id]

        # 尚未被任何已有轨迹认领的检测会启动新身份。
        for index in sorted(available):
            detection = detections[index]
            if detection.track_id is None:
                track_id = f"{detection.object}:{next_id}"
                next_id += 1
            else:
                track_id = detection.track_id
            if track_id in active:
                raise ValueError(f"detection track_id collision: {track_id!r}")
            active[track_id] = _ActiveTrack(
                track_id=track_id,
                object=detection.object,
                center=detection.center,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
            result.append(_state(detection, track_id))

        return result, next_id


def _state(detection: Detection, track_id: str) -> FrameState:
    """把命中的单帧检测转换成带稳定身份的可见状态。"""

    return FrameState(
        frame=detection.frame,
        timestamp_sec=detection.timestamp_sec,
        object=detection.object,
        center=detection.center,
        bbox=detection.bbox,
        confidence=detection.confidence,
        track_id=track_id,
    )
