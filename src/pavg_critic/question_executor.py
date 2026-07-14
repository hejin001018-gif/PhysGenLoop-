"""问题图的依赖执行器与第一阶段确定性节点验证器。

执行器严格按拓扑序处理节点：只有全部父节点为 ``yes`` 时才调用本节点验证逻辑；
任何 ``no/blocked/unknown`` 父节点都会让子节点进入 ``blocked``。这种 gating 防止在
对象或动作前提不存在时凭空判断物理过程，同时保留 ``blocked_by`` 因果链供 Repair
Agent 定位真正根因。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .physics_rules import RULE_ID_TO_CATEGORY
from .question_graph import QuestionGraphValidator
from .schemas import (
    Event,
    NodeResult,
    QuestionGraph,
    QuestionNode,
    TrackSequence,
    ViolationCandidate,
)


# EventDetector 当前没有独立 leave_support 事件；受控下落场景中，可见 fall 是物体已
# 离开支撑并开始运动的可审计代理。未知事件不强制判 No，而返回 unknown 表示能力缺口。
_EVENT_ALIASES = {
    "leave_support": ("fall",),
    "fall": ("fall",),
    "floor_contact": ("floor_contact",),
    "rebound": ("rebound",),
}

_SUPPORT_SURFACE_TOKENS = ("floor", "ground", "surface")


@dataclass(frozen=True)
class QuestionExecutionContext:
    """节点验证所需的只读证据集合。

    ``candidate_keyframes`` 以候选在 ``candidates`` 中的索引为键，沿用现有流水线的
    稳定关联方式，避免相同类别在不同时间段发生时互相覆盖。
    """

    tracks: tuple[TrackSequence, ...]
    events: tuple[Event, ...]
    candidates: tuple[ViolationCandidate, ...]
    candidate_keyframes: Mapping[int, Sequence[int]]


class QuestionGraphExecutor:
    """执行问题 DAG，并路由到对象、事件或规则验证器。"""

    def __init__(
        self,
        *,
        enabled_rule_categories: Sequence[str],
        rule_pass_confidence: float,
    ) -> None:
        """记录规则可用性和规则未命中时的保守通过置信度。"""

        self.enabled_rule_categories = frozenset(enabled_rule_categories)
        self.rule_pass_confidence = rule_pass_confidence
        self.validator = QuestionGraphValidator()

    def execute(
        self,
        graph: QuestionGraph,
        context: QuestionExecutionContext,
    ) -> tuple[NodeResult, ...]:
        """按稳定拓扑序执行全部节点，并返回与拓扑序一致的结果。"""

        ordered_nodes = self.validator.topological_order(graph)
        results_by_id: dict[str, NodeResult] = {}
        ordered_results: list[NodeResult] = []

        for node in ordered_nodes:
            blocked_by = tuple(
                parent_id
                for parent_id in node.parent_ids
                if results_by_id[parent_id].status != "yes"
            )
            if blocked_by:
                # blocked 不是对节点内容的 No 判断，而是“没有执行”的结构化记录。
                parent_statuses = {
                    parent_id: results_by_id[parent_id].status for parent_id in blocked_by
                }
                result = NodeResult(
                    node_id=node.id,
                    category=node.category,
                    status="blocked",
                    direct_score=None,
                    confidence=1.0,
                    reason="One or more prerequisite questions were not satisfied.",
                    verifier="dependency_graph",
                    blocked_by=blocked_by,
                    evidence={"parent_statuses": parent_statuses},
                )
            else:
                result = self._verify(node, context)
            results_by_id[node.id] = result
            ordered_results.append(result)

        return tuple(ordered_results)

    def _verify(
        self,
        node: QuestionNode,
        context: QuestionExecutionContext,
    ) -> NodeResult:
        """按节点类别选择第一阶段确定性验证器。"""

        if node.category == "object":
            return self._verify_object(node, context)
        if node.category == "action":
            return self._verify_action(node, context)
        return self._verify_physics(node, context)

    def _verify_object(
        self,
        node: QuestionNode,
        context: QuestionExecutionContext,
    ) -> NodeResult:
        """用可见轨迹验证对象；检测器未覆盖的类别返回 unknown 而非误判缺失。"""

        targets = set(node.target_objects)
        matching_tracks = [track for track in context.tracks if track.object in targets]
        visible_states = [
            state for track in matching_tracks for state in track.states if state.visible
        ]
        if visible_states:
            frames = _unique_frames(
                (visible_states[0].frame, visible_states[-1].frame)
            )
            return NodeResult(
                node_id=node.id,
                category=node.category,
                status="yes",
                direct_score=1.0,
                confidence=max(state.confidence for state in visible_states),
                reason="The target object has visible tracked observations.",
                verifier="track_observation",
                critical_frames=frames,
                evidence={"visible_observation_count": len(visible_states)},
            )

        # 当前红球检测器不会输出 floor track，但已有 distance_to_floor 证明流水线建立了
        # 一个支撑面坐标；仅对 floor/ground/surface 使用这一几何推断，不扩展到 table。
        normalized_targets = {target.lower() for target in targets}
        is_support_surface = any(
            token in target
            for target in normalized_targets
            for token in _SUPPORT_SURFACE_TOKENS
        )
        surface_evidence = [
            state
            for track in context.tracks
            for state in track.states
            if state.distance_to_floor is not None
        ]
        if is_support_surface and surface_evidence:
            return NodeResult(
                node_id=node.id,
                category=node.category,
                status="yes",
                direct_score=1.0,
                confidence=self.rule_pass_confidence,
                reason="A support surface is inferred from signed distance-to-floor evidence.",
                verifier="surface_geometry",
                critical_frames=(surface_evidence[0].frame,),
                evidence={"distance_observation_count": len(surface_evidence)},
            )

        return NodeResult(
            node_id=node.id,
            category=node.category,
            status="unknown",
            direct_score=None,
            confidence=0.0,
            reason=(
                "No matching observation is available, but the active detector does not "
                "declare complete category coverage."
            ),
            verifier="track_observation",
            evidence={"target_objects": sorted(targets)},
        )

    def _verify_action(
        self,
        node: QuestionNode,
        context: QuestionExecutionContext,
    ) -> NodeResult:
        """把机器可读 expected_events 映射到 EventDetector 的离散事件。"""

        if not node.expected_events:
            return _unknown(node, "The action node does not declare expected_events.")

        aliases: list[str] = []
        unsupported: list[str] = []
        for expected_event in node.expected_events:
            mapped = _EVENT_ALIASES.get(expected_event)
            if mapped is None:
                unsupported.append(expected_event)
            else:
                aliases.extend(mapped)
        if unsupported:
            return _unknown(
                node,
                f"No deterministic event verifier is registered for {unsupported}.",
            )

        targets = set(node.target_objects)
        matches = [
            event
            for event in context.events
            if event.event_type in aliases
            and (not targets or event.object in targets)
        ]
        if matches:
            frames = _unique_frames(
                frame
                for event in matches
                for frame in (event.start_frame, event.peak_frame, event.end_frame)
            )
            return NodeResult(
                node_id=node.id,
                category=node.category,
                status="yes",
                direct_score=1.0,
                confidence=max(event.confidence for event in matches),
                reason="The expected action is supported by one or more detected events.",
                verifier="event_detector",
                critical_frames=frames,
                evidence={"matched_event_types": sorted({event.event_type for event in matches})},
            )

        return NodeResult(
            node_id=node.id,
            category=node.category,
            status="no",
            direct_score=0.0,
            confidence=self.rule_pass_confidence,
            reason="No matching event was detected for the expected action.",
            verifier="event_detector",
            evidence={"expected_event_types": list(node.expected_events)},
        )

    def _verify_physics(
        self,
        node: QuestionNode,
        context: QuestionExecutionContext,
    ) -> NodeResult:
        """用规则候选回答 Physics 节点；规则不可用时返回 unknown。"""

        if not node.rule_ids:
            return _unknown(node, "The physics node does not declare rule_ids.")

        unavailable = [
            rule_id
            for rule_id in node.rule_ids
            if rule_id not in RULE_ID_TO_CATEGORY
            or RULE_ID_TO_CATEGORY[rule_id] not in self.enabled_rule_categories
        ]
        if unavailable:
            return _unknown(
                node,
                f"No enabled deterministic rule verifier is available for {unavailable}.",
            )

        targets = set(node.target_objects)
        matches: list[tuple[int, ViolationCandidate]] = []
        for index, candidate in enumerate(context.candidates):
            if targets and candidate.object not in targets:
                continue
            if set(candidate.rules).intersection(node.rule_ids):
                matches.append((index, candidate))

        if matches:
            candidate_index, strongest = max(
                matches,
                key=lambda item: item[1].detector_score,
            )
            frames = tuple(
                int(frame)
                for frame in context.candidate_keyframes.get(
                    candidate_index,
                    strongest.evidence_frames,
                )
            )
            return NodeResult(
                node_id=node.id,
                category=node.category,
                status="no",
                direct_score=0.0,
                confidence=strongest.detector_score,
                reason=strongest.reason,
                verifier="physics_rule",
                critical_frames=frames,
                rule_ids=strongest.rules,
                evidence={
                    "violation_category": strongest.category,
                    "candidate_count": len(matches),
                },
            )

        # 只有确认所有声明规则都已注册且启用后，规则未命中才可解释为保守的 Yes。
        return NodeResult(
            node_id=node.id,
            category=node.category,
            status="yes",
            direct_score=1.0,
            confidence=self.rule_pass_confidence,
            reason="All declared deterministic rules ran without producing a violation.",
            verifier="physics_rule",
            rule_ids=node.rule_ids,
        )


def _unknown(node: QuestionNode, reason: str) -> NodeResult:
    """统一构造能力或证据不足的 unknown 结果。"""

    return NodeResult(
        node_id=node.id,
        category=node.category,
        status="unknown",
        direct_score=None,
        confidence=0.0,
        reason=reason,
        verifier="unavailable",
        rule_ids=node.rule_ids,
    )


def _unique_frames(frames) -> tuple[int, ...]:
    """按首次出现顺序去重证据帧，保留事件的时间阅读顺序。"""

    return tuple(dict.fromkeys(int(frame) for frame in frames))

