"""从现有 ``PhysicsPlan`` 生成确定性第一阶段问题图。

模板生成器不是论文中 VLM QG 的替代品，而是无需外部模型即可验证图执行架构的
bootstrap。它把计划中的对象和事件转换为稳定的 Object/Action 节点，再为当前规则
引擎能够回答的约束生成 Physics 节点。未来接入 VLM 时可替换该组件，图校验、执行
与评分层无需改动。
"""

from __future__ import annotations

from .config import QuestionGraphConfig
from .question_graph import QuestionGraphValidator
from .schemas import CriticRequest, QuestionGraph, QuestionNode


# 模板只依赖机器可读事件名，不从自然语言问题反向推断规则。每个映射包含问题模板、
# 稳定规则 ID 和对应物理领域；未列出的事件仍会生成 Action 节点，但不会虚构 Physics
# 检查能力。
_PHYSICS_EVENT_TEMPLATES = {
    "fall": (
        "Does {object} keep physically plausible downward motion while unsupported?",
        "unsupported_objects_accelerate_downward",
        "solid_mechanics",
    ),
    "floor_contact": (
        "Does {object} remain outside the solid floor during contact?",
        "solid_surface_non_penetration",
        "solid_mechanics",
    ),
    "rebound": (
        "Does {object} rebound only after visible floor contact?",
        "velocity_reversal_before_contact",
        "solid_mechanics",
    ),
}

_ACTION_QUESTION_TEMPLATES = {
    "leave_support": "Does {object} leave its support?",
    "fall": "Does {object} move downward after leaving support?",
    "floor_contact": "Does {object} make contact with the floor?",
    "rebound": "Does {object} rebound after the downward motion?",
}

# 支撑面通常不是主要运动对象。该启发式只决定模板问题的主语，不改变原始对象列表；
# 复杂多物体场景应由后续 Planner 或 VLM QG 显式提供每个节点的 target_objects。
_STATIC_OBJECT_TOKENS = ("floor", "ground", "table", "wall", "support", "surface")


class TemplateQuestionGraphGenerator:
    """把扁平 PhysicsPlan 转换成可执行且可审计的问题 DAG。"""

    def __init__(self, config: QuestionGraphConfig) -> None:
        """保存模板开关，并复用统一图校验器检查生成结果。"""

        self.config = config
        self.validator = QuestionGraphValidator()

    def generate(self, request: CriticRequest) -> QuestionGraph:
        """为一次请求生成稳定问题图；空计划会得到合法空图。"""

        objects = tuple(dict.fromkeys(request.physics_plan.objects))
        events = tuple(dict.fromkeys(request.physics_plan.expected_events))
        primary_object = _select_primary_object(objects)
        display_name = _display(primary_object) if primary_object else "the target object"

        nodes: list[QuestionNode] = []
        object_node_ids: dict[str, str] = {}
        for index, object_name in enumerate(objects, start=1):
            node_id = f"O{index}"
            object_node_ids[object_name] = node_id
            nodes.append(
                QuestionNode(
                    id=node_id,
                    category="object",
                    question=f"Is {_display(object_name)} present in the video?",
                    target_objects=(object_name,),
                    verifier_hint="observation",
                )
            )

        # Action 节点只依赖主要运动对象的存在。模板不会把“物理上应先发生的事件”当作
        # 逻辑前置条件，否则恰好违反事件顺序的视频会被 blocked，无法暴露顺序错误。
        action_node_ids: dict[str, str] = {}
        primary_parent = (
            (object_node_ids[primary_object],)
            if primary_object is not None and primary_object in object_node_ids
            else ()
        )
        for index, event_type in enumerate(events, start=1):
            node_id = f"A{index}"
            action_node_ids[event_type] = node_id
            template = _ACTION_QUESTION_TEMPLATES.get(
                event_type,
                "Does {object} exhibit the expected event '" + event_type + "'?",
            )
            nodes.append(
                QuestionNode(
                    id=node_id,
                    category="action",
                    question=template.format(object=display_name),
                    parent_ids=primary_parent,
                    target_objects=((primary_object,) if primary_object else ()),
                    expected_events=(event_type,),
                    verifier_hint="event",
                )
            )

        physics_index = 1
        emitted_rule_ids: set[str] = set()
        for event_type in events:
            definition = _PHYSICS_EVENT_TEMPLATES.get(event_type)
            if definition is None:
                continue
            question_template, rule_id, domain = definition
            emitted_rule_ids.add(rule_id)
            nodes.append(
                QuestionNode(
                    id=f"P{physics_index}",
                    category="physics",
                    question=question_template.format(object=display_name),
                    parent_ids=(action_node_ids[event_type],),
                    target_objects=((primary_object,) if primary_object else ()),
                    expected_events=(event_type,),
                    physics_domain=domain,
                    verifier_hint="rule",
                    rule_ids=(rule_id,),
                )
            )
            physics_index += 1

        if self.config.include_generic_physics and action_node_ids:
            # 通用恒存/连续性检查依附首个计划动作：对象存在但计划动作未发生时，这些
            # Physics 问题会被 blocked，而观测通道仍可独立报告消失或瞬移 violation。
            generic_parent = (next(iter(action_node_ids.values())),)
            generic_definitions = (
                (
                    "Does {object} preserve its identity and remain visible through the event?",
                    "object_persistence",
                ),
                (
                    "Does {object} move continuously without teleporting between frames?",
                    "bounded_interframe_displacement",
                ),
            )
            for question_template, rule_id in generic_definitions:
                if rule_id in emitted_rule_ids:
                    continue
                nodes.append(
                    QuestionNode(
                        id=f"P{physics_index}",
                        category="physics",
                        question=question_template.format(object=display_name),
                        parent_ids=generic_parent,
                        target_objects=((primary_object,) if primary_object else ()),
                        physics_domain="solid_mechanics",
                        verifier_hint="rule",
                        rule_ids=(rule_id,),
                    )
                )
                physics_index += 1

        graph = QuestionGraph(nodes=tuple(nodes), source="physics_plan_template")
        self.validator.validate(graph)
        return graph


def _select_primary_object(objects: tuple[str, ...]) -> str | None:
    """选择第一个不像静态支撑面的对象作为模板动作主语。"""

    for object_name in objects:
        normalized = object_name.lower()
        if not any(token in normalized for token in _STATIC_OBJECT_TOKENS):
            return object_name
    return objects[0] if objects else None


def _display(identifier: str) -> str:
    """把机器友好的 ``red_ball`` 转为问题文本中的 ``red ball``。"""

    return identifier.replace("_", " ")

