"""事件驱动的证据关键帧选择。

关键帧不是均匀抽样，而是围绕异常前最后正常状态、开始、峰值和结果帧选择。返回帧
始终来自实际轨迹，因而不会要求下游解码不存在的帧号。
"""

from __future__ import annotations

from typing import Iterable

from .config import TemporalConfig
from .schemas import TrackSequence, ViolationCandidate


class KeyframeSelector:
    """为一个异常候选选择去重且稳定排序的证据帧。"""

    def __init__(self, config: TemporalConfig) -> None:
        """保存异常前证据帧的最大搜索窗口。"""

        self.config = config

    def select(
        self, candidate: ViolationCandidate, tracks: Iterable[TrackSequence]
    ) -> tuple[int, ...]:
        """选择 ``F_pre/F_start/F_peak/F_post`` 对应的可用轨迹帧。"""

        track = next((item for item in tracks if item.track_id == candidate.track_id), None)
        if track is None or not track.states:
            # 极端情况下轨迹已不可用，仍保留规则明确给出的证据帧。
            return _unique(
                (
                    *candidate.evidence_frames,
                    candidate.start_frame,
                    candidate.peak_frame,
                    candidate.end_frame,
                )
            )

        frames = sorted({state.frame for state in track.states})
        # F_pre 取搜索窗口内最接近异常开始的上一帧，即最后一个正常证据状态。
        before = [
            frame
            for frame in frames
            if candidate.start_frame - self.config.pre_context_frames
            <= frame
            < candidate.start_frame
        ]
        targets: list[int] = []
        if before:
            targets.append(before[-1])
        targets.extend(
            (candidate.start_frame, candidate.peak_frame, candidate.end_frame)
        )
        targets.extend(candidate.evidence_frames)
        # 规则可能给出稀疏或抽帧前的帧号，先映射到最近实际帧再稳定去重。
        return _unique(_nearest(frame, frames) for frame in targets)


def _nearest(target: int, frames: list[int]) -> int:
    """选择数值上最近的帧；距离相同时优先更早帧以保证确定性。"""

    return min(frames, key=lambda frame: (abs(frame - target), frame))


def _unique(frames: Iterable[int]) -> tuple[int, ...]:
    """保持首次出现顺序去重，避免 start 与 peak 相同时重复输出。"""

    return tuple(dict.fromkeys(int(frame) for frame in frames))
