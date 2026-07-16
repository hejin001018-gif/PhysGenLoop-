"""将粗粒度规则命中细化为异常 start/peak/end 区间。

多数区间型规则已经从 Event 继承完整范围；点状的反弹规则只知道速度反转帧，需要
结合随后持续的向上运动事件补全异常结果区间。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .schemas import Event, ViolationCandidate


class TemporalLocalizer:
    """根据同轨迹相关事件扩展候选异常的时间范围。"""

    def localize(
        self, candidate: ViolationCandidate, events: Iterable[Event]
    ) -> ViolationCandidate:
        """细化单个候选；当前仅提前反弹需要跨事件扩展结束帧。"""

        if candidate.category != "premature_rebound":
            return candidate
        # 只关联同一 track 且覆盖反弹峰值的向上运动，避免串用其他物体的事件。
        upward = [
            event
            for event in events
            if event.track_id == candidate.track_id
            and event.event_type == "upward_motion"
            and event.start_frame <= candidate.peak_frame <= event.end_frame
        ]
        if not upward:
            return candidate
        interval = upward[0]
        # dataclass 是冻结对象，通过 replace 返回新候选而不修改原始规则证据。
        return replace(candidate, end_frame=max(candidate.end_frame, interval.end_frame))
