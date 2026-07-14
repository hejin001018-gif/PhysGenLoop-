"""PQSG 启发的问题图生成、PAVG 混合和两阶段问答。

PQSG 在这里是 PAVG 内部的一种问题生成/问答策略，而不是替代 Critic 主流水线。
模板图保证已有 PhysicsPlan 与规则覆盖，模型图补充长尾对象、动作和物理约束；两者
合并后仍由 PAVG 的统一 DAG 校验器、证据执行器和覆盖感知评分器处理。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .interfaces import QuestionGraphGenerator, StructuredTextModel
from .question_graph import QuestionGraphValidator
from .schemas import CriticRequest, QuestionGraph, QuestionNode, SchemaError


_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["nodes"],
    "additionalProperties": False,
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "category",
                    "question",
                    "parent_ids",
                    "target_objects",
                    "expected_events",
                    "physics_domain",
                    "verifier_hint",
                    "rule_ids",
                    "weight",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "category": {"enum": ["object", "action", "physics"]},
                    "question": {"type": "string"},
                    "parent_ids": {"type": "array", "items": {"type": "string"}},
                    "target_objects": {"type": "array", "items": {"type": "string"}},
                    "expected_events": {"type": "array", "items": {"type": "string"}},
                    "physics_domain": {"type": ["string", "null"]},
                    "verifier_hint": {
                        "enum": ["observation", "event", "rule", "hybrid"]
                    },
                    "rule_ids": {"type": "array", "items": {"type": "string"}},
                    "weight": {"type": "number", "exclusiveMinimum": 0},
                },
            },
        }
    },
}


class PQSGQuestionGraphGenerator:
    """调用结构化文本模型，将 prompt/计划转换成 O/A/P 问题 DAG。"""

    def __init__(self, model: StructuredTextModel) -> None:
        self.model = model
        self.validator = QuestionGraphValidator()

    def generate(self, request: CriticRequest) -> QuestionGraph:
        """生成并严格校验模型图；非法边、悬空引用和环路都会立即失败。"""

        payload = self.model.generate_json(
            system_prompt=(
                "Generate a minimal physics scene graph as atomic yes/no questions. "
                "Use Object, Action, Physics categories and only forward dependencies. "
                "Do not answer the questions."
            ),
            user_prompt=json.dumps(
                {
                    "prompt": request.prompt,
                    "objects": request.physics_plan.objects,
                    "expected_events": request.physics_plan.expected_events,
                },
                ensure_ascii=False,
            ),
            schema=_GRAPH_SCHEMA,
        )
        nodes_raw = payload.get("nodes")
        if not isinstance(nodes_raw, list):
            raise SchemaError("PQSG response must contain a nodes array")
        graph = QuestionGraph(
            nodes=tuple(_parse_node(item) for item in nodes_raw),
            source="pqsg_model",
        )
        self.validator.validate(graph)
        return graph


class HybridQuestionGraphGenerator:
    """合并 PAVG 模板图与 PQSG 模型图，并为模型节点建立独立命名空间。"""

    def __init__(
        self,
        template: QuestionGraphGenerator,
        pqsg: QuestionGraphGenerator,
    ) -> None:
        self.template = template
        self.pqsg = pqsg
        self.validator = QuestionGraphValidator()

    def generate(self, request: CriticRequest) -> QuestionGraph:
        template_graph = self.template.generate(request)
        pqsg_graph = self.pqsg.generate(request)

        # 模型经常也从 O1 开始编号；统一加 Q_ 前缀可避免覆盖模板节点，同时保持
        # PQSG 图内部父子引用不变。图仍作为一个 DAG 交给后续执行器。
        id_map = {node.id: f"Q_{node.id}" for node in pqsg_graph.nodes}
        pqsg_nodes = tuple(
            QuestionNode(
                id=id_map[node.id],
                category=node.category,
                question=node.question,
                parent_ids=tuple(id_map[parent] for parent in node.parent_ids),
                target_objects=node.target_objects,
                expected_events=node.expected_events,
                physics_domain=node.physics_domain,
                verifier_hint=node.verifier_hint,
                rule_ids=node.rule_ids,
                weight=node.weight,
            )
            for node in pqsg_graph.nodes
        )
        graph = QuestionGraph(
            nodes=template_graph.nodes + pqsg_nodes,
            source="pavg_hybrid_template_pqsg",
        )
        self.validator.validate(graph)
        return graph


@dataclass(frozen=True)
class PQSGAnswer:
    """PQSG 两阶段 QA 的最终可审计结果。"""

    answer: str
    confidence: float
    reasoning: str

    def __post_init__(self) -> None:
        if self.answer not in {"yes", "no"}:
            raise SchemaError("PQSG answer must be yes or no")
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaError("PQSG answer confidence must be in [0, 1]")


class TwoPassQuestionAnswerer:
    """先开放推理、再强制 yes/no，复现 PQSG 的两阶段 QA 边界。"""

    def __init__(self, model: StructuredTextModel) -> None:
        self.model = model

    def answer(self, *, question: str, evidence: str) -> PQSGAnswer:
        reasoning_payload = self.model.generate_json(
            system_prompt=(
                "Analyze only the supplied evidence. Explain whether it supports the "
                "question, but do not emit a final yes/no token."
            ),
            user_prompt=f"question={question}\nevidence={evidence}",
            schema={
                "type": "object",
                "required": ["reasoning"],
                "additionalProperties": False,
                "properties": {"reasoning": {"type": "string"}},
            },
        )
        reasoning = str(reasoning_payload.get("reasoning", "")).strip()
        if not reasoning:
            raise SchemaError("PQSG reasoning pass returned empty reasoning")

        answer_payload = self.model.generate_json(
            system_prompt="Return a forced yes/no answer grounded in the supplied reasoning.",
            user_prompt=(
                f"question={question}\nevidence={evidence}\nreasoning={reasoning}"
            ),
            schema={
                "type": "object",
                "required": ["answer", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "answer": {"enum": ["yes", "no"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        )
        return PQSGAnswer(
            answer=str(answer_payload.get("answer", "")).lower(),
            confidence=float(answer_payload.get("confidence", 0.0)),
            reasoning=reasoning,
        )


def _parse_node(raw: Any) -> QuestionNode:
    """把不可信模型 JSON 规范化为冻结 schema；未知字段不会泄漏到核心对象。"""

    if not isinstance(raw, Mapping):
        raise SchemaError("PQSG node must be an object")
    try:
        return QuestionNode(
            id=str(raw["id"]),
            category=str(raw["category"]),
            question=str(raw["question"]),
            parent_ids=tuple(str(value) for value in raw.get("parent_ids", ())),
            target_objects=tuple(str(value) for value in raw.get("target_objects", ())),
            expected_events=tuple(str(value) for value in raw.get("expected_events", ())),
            physics_domain=(
                None if raw.get("physics_domain") is None else str(raw["physics_domain"])
            ),
            verifier_hint=str(raw.get("verifier_hint", "hybrid")),
            rule_ids=tuple(str(value) for value in raw.get("rule_ids", ())),
            weight=float(raw.get("weight", 1.0)),
        )
    except KeyError as exc:
        raise SchemaError(f"PQSG node missing field: {exc.args[0]}") from exc
