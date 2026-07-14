"""首期物理异常分类的可解释规则引擎。

规则层比较事件顺序、物理计划与局部几何证据，产出 ``ViolationCandidate``，但不
直接决定最终报告。候选仍需经过时序细化、关键帧选择、可选 VLM 复核和分数融合。
每条规则都输出稳定的规则 ID，便于审计、统计和后续消融实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import EventConfig, RuleConfig
from .schemas import CriticRequest, Event, TrackSequence, ViolationCandidate


# 问题图执行器通过稳定 rule ID 判断某个 Physics 节点是否具备可用验证器，同时将
# RuleConfig 中的类别开关映射回来。该注册表也避免模板生成器依赖私有规则方法名。
RULE_ID_TO_CATEGORY = {
    "velocity_reversal_before_contact": "premature_rebound",
    "solid_surface_non_penetration": "surface_penetration",
    "object_persistence": "object_disappearance",
    "unsupported_objects_accelerate_downward": "reverse_gravity",
    "bounded_interframe_displacement": "teleportation",
}


@dataclass(frozen=True)
class RuleContext:
    """一次规则评估所需的只读请求、轨迹和事件集合。"""

    request: CriticRequest
    tracks: tuple[TrackSequence, ...]
    events: tuple[Event, ...]


class PhysicsRuleEngine:
    """执行配置中启用的规则，并返回尚未融合的异常候选。"""

    def __init__(self, config: RuleConfig, event_config: EventConfig) -> None:
        """保存规则与事件阈值；事件阈值可用于保持跨层语义一致。"""

        self.config = config
        self.event_config = event_config

    def evaluate(self, context: RuleContext) -> tuple[ViolationCandidate, ...]:
        """按固定顺序运行启用规则，保证相同输入得到稳定输出顺序。"""

        enabled = set(self.config.enabled)
        candidates: list[ViolationCandidate] = []
        if "premature_rebound" in enabled:
            candidates.extend(self._premature_rebound(context))
        if "surface_penetration" in enabled:
            candidates.extend(self._surface_penetration(context))
        if "object_disappearance" in enabled:
            candidates.extend(self._object_disappearance(context))
        if "reverse_gravity" in enabled:
            candidates.extend(self._reverse_gravity(context))
        if "teleportation" in enabled:
            candidates.extend(self._teleportation(context))
        return tuple(candidates)

    def _premature_rebound(self, context: RuleContext) -> list[ViolationCandidate]:
        """识别没有近期地面接触作为前因的速度反转。"""

        result: list[ViolationCandidate] = []
        for rebound in _events(context.events, "rebound"):
            contacts = _events(context.events, "floor_contact", rebound.track_id)
            # 合法接触必须发生在反弹之前，且间隔不能超过配置的回看窗口。
            has_contact = any(
                0 <= rebound.start_frame - contact.end_frame <= self.config.contact_lookback_frames
                for contact in contacts
            )
            if has_contact:
                continue
            distance = rebound.evidence.get("distance_to_floor")
            # 反弹点离地越远，提前反弹证据越强；增益封顶以避免单一几何量支配分数。
            distance_bonus = (
                min(max(float(distance), 0.0) / 100.0, 0.15) if distance is not None else 0.0
            )
            score = min(0.99, 0.78 * rebound.confidence + 0.12 + distance_bonus)
            result.append(
                ViolationCandidate(
                    object=rebound.object,
                    track_id=rebound.track_id,
                    category="premature_rebound",
                    start_frame=rebound.start_frame,
                    peak_frame=rebound.peak_frame,
                    end_frame=rebound.end_frame,
                    reason="The object reverses its downward motion before visible floor contact.",
                    repair_instruction=(
                        "Keep the object moving downward until visible floor contact, "
                        "then apply rebound."
                    ),
                    detector_score=score,
                    rules=("velocity_reversal_before_contact",),
                    evidence_frames=(rebound.start_frame,),
                    evidence={"rebound_event": rebound.evidence},
                )
            )
        return result

    def _surface_penetration(self, context: RuleContext) -> list[ViolationCandidate]:
        """把超过容忍深度的地面穿透事件转换成实体性违规。"""

        result: list[ViolationCandidate] = []
        for event in _events(context.events, "surface_penetration"):
            state = _state_at(context.tracks, event.track_id, event.peak_frame)
            # 取最深帧的负距离绝对值作为严重程度，50 像素后分数达到上限。
            depth = abs(min(state.distance_to_floor or 0.0, 0.0)) if state else 0.0
            score = min(0.99, 0.72 + depth / 50.0)
            result.append(
                ViolationCandidate(
                    object=event.object,
                    track_id=event.track_id,
                    category="surface_penetration",
                    start_frame=event.start_frame,
                    peak_frame=event.peak_frame,
                    end_frame=event.end_frame,
                    reason="The object visibly penetrates the configured support surface.",
                    repair_instruction=(
                        "Preserve solid contact: stop the object at the surface and "
                        "prevent interpenetration."
                    ),
                    detector_score=score,
                    rules=("solid_surface_non_penetration",),
                    evidence_frames=(event.start_frame, event.peak_frame, event.end_frame),
                    evidence={"penetration_depth_px": depth},
                )
            )
        return result

    def _object_disappearance(self, context: RuleContext) -> list[ViolationCandidate]:
        """将持续缺失区间映射为物体恒存违规候选。"""

        return [
            ViolationCandidate(
                object=event.object,
                track_id=event.track_id,
                category="object_disappearance",
                start_frame=event.start_frame,
                peak_frame=event.peak_frame,
                end_frame=event.end_frame,
                reason=(
                    "The tracked object disappears without a visible exit or "
                    "explained occlusion."
                ),
                repair_instruction=(
                    "Keep the object's identity and appearance continuous, or show a "
                    "clear occluder/exit."
                ),
                detector_score=min(0.95, 0.7 + 0.04 * int(event.evidence["frame_count"])),
                rules=("object_persistence",),
                evidence_frames=(event.start_frame, event.end_frame),
                evidence={"missing_frame_count": event.evidence["frame_count"]},
            )
            for event in _events(context.events, "disappearance")
        ]

    def _reverse_gravity(self, context: RuleContext) -> list[ViolationCandidate]:
        """识别计划要求下落、但物体无接触前因而持续上移的情况。"""

        expected = set(context.request.physics_plan.expected_events)
        # 没有“下落/离开支撑面”预期时，上移可能来自抛掷或主动运动，不能据此判错。
        if not expected.intersection({"fall", "leave_support"}):
            return []

        result: list[ViolationCandidate] = []
        for upward in _events(context.events, "upward_motion"):
            rebounds = _events(context.events, "rebound", upward.track_id)
            # 已被识别为速度反转的上移交给碰撞规则，避免同时报告反重力和提前反弹。
            if any(
                upward.start_frame - 1 <= rebound.start_frame <= upward.end_frame
                for rebound in rebounds
            ):
                continue
            contacts = _events(context.events, "floor_contact", upward.track_id)
            # 近期合法接触可以解释向上运动，因此不属于反重力。
            if any(
                0 <= upward.start_frame - contact.end_frame
                <= self.config.gravity_contact_lookback_frames
                for contact in contacts
            ):
                continue
            result.append(
                ViolationCandidate(
                    object=upward.object,
                    track_id=upward.track_id,
                    category="reverse_gravity",
                    start_frame=upward.start_frame,
                    peak_frame=upward.peak_frame,
                    end_frame=upward.end_frame,
                    reason=(
                        "An object expected to fall moves upward without a preceding "
                        "contact event."
                    ),
                    repair_instruction=(
                        "Maintain downward acceleration while the object is unsupported "
                        "and falling."
                    ),
                    detector_score=min(0.94, 0.75 + 0.03 * upward.evidence["frame_count"]),
                    rules=("unsupported_objects_accelerate_downward",),
                    evidence_frames=(upward.start_frame, upward.peak_frame, upward.end_frame),
                )
            )
        return result

    def _teleportation(self, context: RuleContext) -> list[ViolationCandidate]:
        """将超过表观速度阈值的点事件映射为空间连续性违规。"""

        return [
            ViolationCandidate(
                object=event.object,
                track_id=event.track_id,
                category="teleportation",
                start_frame=event.start_frame,
                peak_frame=event.peak_frame,
                end_frame=event.end_frame,
                reason="The object's apparent speed exceeds the configured continuity limit.",
                repair_instruction="Keep position changes continuous across adjacent frames.",
                detector_score=min(0.98, 0.78 * event.confidence + 0.16),
                rules=("bounded_interframe_displacement",),
                evidence_frames=(event.start_frame,),
                evidence=event.evidence,
            )
            for event in _events(context.events, "teleport")
        ]


def _events(
    events: Iterable[Event], event_type: str, track_id: str | None = None
) -> list[Event]:
    """按事件类型及可选轨迹身份筛选事件，集中维护过滤语义。"""

    return [
        event
        for event in events
        if event.event_type == event_type and (track_id is None or event.track_id == track_id)
    ]


def _state_at(tracks: Iterable[TrackSequence], track_id: str, frame: int):
    """获取目标帧最近的轨迹状态，用于读取规则严重程度证据。"""

    for track in tracks:
        if track.track_id == track_id:
            return min(track.states, key=lambda state: abs(state.frame - frame), default=None)
    return None
