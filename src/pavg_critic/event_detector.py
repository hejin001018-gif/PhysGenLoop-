"""将连续轨迹状态转换为可解释的视觉/物理事件。

事件检测只陈述“观察到了什么”，不直接判断“是否违反物理”。例如反弹事件仅表示
y 方向速度由正变负；它发生在接触前还是接触后，由后续规则引擎结合事件顺序判断。
这种分层设计让每条违规结论都能追溯到独立的轨迹与事件证据。
"""

from __future__ import annotations

from math import hypot
from typing import Callable, Iterable

from .config import EventConfig
from .schemas import Event, FrameState, TrackSequence


class EventDetector:
    """在每条轨迹上检测运动、接触、恒存与连续性事件。"""

    def __init__(self, config: EventConfig) -> None:
        """保存事件阈值。"""

        self.config = config

    def detect(self, tracks: Iterable[TrackSequence]) -> tuple[Event, ...]:
        """检测全部轨迹并按发生帧排序，确保规则结果与输入顺序无关。"""

        events: list[Event] = []
        for track in tracks:
            # 各检测器只关注一种证据族，便于后续独立替换或做消融实验。
            events.extend(self._motion_events(track))
            events.extend(self._contact_events(track))
            events.extend(self._disappearance_events(track))
            events.extend(self._teleport_events(track))
        return tuple(sorted(events, key=lambda item: (item.start_frame, item.event_type)))

    def _motion_events(self, track: TrackSequence) -> list[Event]:
        """检测下落、向上运动以及由向下转为向上的反弹。"""

        epsilon = self.config.velocity_epsilon_px_s
        visible = [state for state in track.states if state.visible and state.velocity is not None]
        result: list[Event] = []
        for previous, current in zip(visible, visible[1:]):
            assert previous.velocity is not None and current.velocity is not None
            # 图像 y 轴向下：正速度表示下落，负速度表示上升。
            if previous.velocity[1] > epsilon and current.velocity[1] < -epsilon:
                result.append(
                    _event(
                        "rebound",
                        track,
                        current.frame,
                        confidence=min(previous.confidence, current.confidence),
                        evidence={
                            "velocity_y_before": previous.velocity[1],
                            "velocity_y_after": current.velocity[1],
                            "distance_to_floor": current.distance_to_floor,
                        },
                    )
                )

        # 连续区间事件比逐帧事件更适合输出 start/peak/end 和选择关键证据帧。
        result.extend(
            self._runs(
                track,
                "upward_motion",
                lambda state: bool(
                    state.visible and state.velocity and state.velocity[1] < -epsilon
                ),
                self.config.min_upward_frames,
            )
        )
        result.extend(
            self._runs(
                track,
                "fall",
                lambda state: bool(
                    state.visible and state.velocity and state.velocity[1] > epsilon
                ),
                1,
            )
        )
        return result

    def _contact_events(self, track: TrackSequence) -> list[Event]:
        """依据地面有符号距离区分可容忍接触和明显表面穿透。"""

        # 接触允许少量负距离，以吸收检测框抖动；更深的负距离单独归为穿透。
        contact = self._runs(
            track,
            "floor_contact",
            lambda state: bool(
                state.visible
                and state.distance_to_floor is not None
                and -self.config.penetration_tolerance_px
                <= state.distance_to_floor
                <= self.config.contact_tolerance_px
            ),
            1,
        )
        penetration = self._runs(
            track,
            "surface_penetration",
            lambda state: bool(
                state.visible
                and state.distance_to_floor is not None
                and state.distance_to_floor < -self.config.penetration_tolerance_px
            ),
            1,
        )
        return contact + penetration

    def _disappearance_events(self, track: TrackSequence) -> list[Event]:
        """把达到最小长度的连续不可见状态标记为消失事件。"""

        return self._runs(
            track,
            "disappearance",
            lambda state: not state.visible,
            self.config.min_disappearance_frames,
        )

    def _teleport_events(self, track: TrackSequence) -> list[Event]:
        """用表观速度上限发现相邻观测之间的瞬移候选。"""

        result: list[Event] = []
        for state in track.states:
            if not state.visible or state.velocity is None:
                continue
            speed = hypot(*state.velocity)
            if speed > self.config.teleport_speed_px_s:
                result.append(
                    _event(
                        "teleport",
                        track,
                        state.frame,
                        confidence=state.confidence,
                        evidence={"speed_px_s": speed},
                    )
                )
        return result

    def _runs(
        self,
        track: TrackSequence,
        event_type: str,
        predicate: Callable[[FrameState], bool],
        minimum_length: int,
    ) -> list[Event]:
        """将满足谓词的连续状态压缩成区间事件。

        连续性以输入状态序列为准；调用前轨迹已按时间排序。短于 ``minimum_length``
        的区间视为检测抖动而丢弃。
        """

        runs: list[list[FrameState]] = []
        current: list[FrameState] = []
        for state in track.states:
            if predicate(state):
                current.append(state)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)

        result: list[Event] = []
        for run in runs:
            if len(run) < minimum_length:
                continue
            # 不同事件使用不同的“最强证据”定义，统一封装在 _peak_state 中。
            peak = _peak_state(event_type, run)
            result.append(
                Event(
                    event_type=event_type,
                    object=track.object,
                    track_id=track.track_id,
                    start_frame=run[0].frame,
                    peak_frame=peak.frame,
                    end_frame=run[-1].frame,
                    confidence=sum(state.confidence for state in run) / len(run),
                    evidence={"frame_count": len(run)},
                )
            )
        return result


def _event(
    event_type: str,
    track: TrackSequence,
    frame: int,
    *,
    confidence: float,
    evidence: dict[str, object],
) -> Event:
    """构造发生在单帧上的点事件，令 start/peak/end 使用同一帧。"""

    return Event(
        event_type=event_type,
        object=track.object,
        track_id=track.track_id,
        start_frame=frame,
        peak_frame=frame,
        end_frame=frame,
        confidence=confidence,
        evidence=evidence,
    )


def _peak_state(event_type: str, states: list[FrameState]) -> FrameState:
    """按事件语义选择证据最强帧。

    穿透选择有符号距离最负的状态，运动事件选择绝对垂直速度最大的状态，其余事件
    使用区间中点，避免所有事件都机械地把开始帧当作峰值。
    """

    if event_type == "surface_penetration":
        return min(states, key=lambda item: item.distance_to_floor or 0.0)
    if event_type in {"fall", "upward_motion"}:
        return max(states, key=lambda item: abs(item.velocity[1]) if item.velocity else 0.0)
    return states[len(states) // 2]
