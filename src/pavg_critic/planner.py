"""Prompt 到 PhysicsPlan 的确定性与模型规划入口。

模板实现只提取 prompt 明确表达的对象、事件和定性约束。它不猜测质量、速度、
重力常数等数值，因此可作为无 API 环境的安全默认值及模型失败时的降级实现。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .api_models import ModelAPIError
from .interfaces import PhysicsPlanner, StructuredTextModel
from .schemas import (
    CriticRequest,
    PhysicsConstraint,
    PhysicsPlan,
    PhysicsRelation,
    PlannerMetadata,
    SchemaError,
)


PLAN_EVENT_VOCABULARY = (
    "leave_support",
    "fall",
    "floor_contact",
    "rebound",
    "projectile",
    "collision",
    "roll_down_slope",
)


@dataclass(frozen=True)
class _ObjectPattern:
    name: str
    pattern: re.Pattern[str]


class TemplatePhysicsPlanner:
    """用有序双语词表生成保守、可复现的 PhysicsPlan。"""

    _OBJECT_PATTERNS = (
        _ObjectPattern("red_ball", re.compile(r"\bred\s+ball\b|红球", re.IGNORECASE)),
        _ObjectPattern("ball", re.compile(r"\bballs?\b|球", re.IGNORECASE)),
        _ObjectPattern("table", re.compile(r"\btable\b|桌子|桌面", re.IGNORECASE)),
        _ObjectPattern(
            "floor", re.compile(r"\bfloor\b|\bground\b|地面|地板", re.IGNORECASE)
        ),
        _ObjectPattern("wall", re.compile(r"\bwall\b|墙", re.IGNORECASE)),
        _ObjectPattern(
            "rock", re.compile(r"\b(?:rock|boulder)s?\b|石头|岩石|巨石", re.IGNORECASE)
        ),
        _ObjectPattern(
            "slope", re.compile(r"\b(?:slope|hill)s?\b|斜坡|山坡|坡", re.IGNORECASE)
        ),
    )
    _EVENT_PATTERNS = {
        "fall": re.compile(
            r"\bfall(?:s|ing|en)?\b|\bdrop(?:s|ped|ping)?\b|下落|掉落|坠落",
            re.IGNORECASE,
        ),
        "floor_contact": re.compile(
            r"\bhit(?:s|ting)?\s+(?:the\s+)?(?:floor|ground)\b|"
            r"\bcontact(?:s|ed|ing)?\s+(?:the\s+)?(?:floor|ground)\b|"
            r"落地|接触地面|撞击地面",
            re.IGNORECASE,
        ),
        "rebound": re.compile(
            r"\bbounc(?:e|es|ed|ing)\b|\brebound(?:s|ed|ing)?\b|反弹|回弹",
            re.IGNORECASE,
        ),
        "projectile": re.compile(
            r"\bthrow(?:s|n|ing)?\b|\bthrown\b|\blaunch(?:es|ed|ing)?\b|抛出|投掷|发射",
            re.IGNORECASE,
        ),
        "collision": re.compile(
            r"\bcollid(?:e|es|ed|ing)\b|\bcollision\b|相撞|碰撞",
            re.IGNORECASE,
        ),
        "roll_down_slope": re.compile(
            r"\broll(?:s|ed|ing)?\b.*\b(?:downhill|slope|hill)\b|"
            r"滚下坡|滚下斜坡|沿坡滚动",
            re.IGNORECASE,
        ),
    }
    _EVENT_ORDER = (
        "leave_support",
        "fall",
        "floor_contact",
        "rebound",
        "projectile",
        "collision",
        "roll_down_slope",
    )
    _STATIC_OBJECTS = frozenset({"table", "floor", "wall", "slope"})

    def generate(
        self, prompt: str, partial_plan: PhysicsPlan | None = None
    ) -> PhysicsPlan:
        # partial_plan 由 Resolver 统一合并；模板只需要 prompt 的确定性词表结果。
        del partial_plan
        text = str(prompt or "").strip()
        if not text:
            return PhysicsPlan()

        objects = self._extract_objects(text)
        events = self._extract_events(text)
        if not objects and not events:
            return PhysicsPlan()

        primary = next((item for item in objects if item not in self._STATIC_OBJECTS), None)
        relations = self._derive_relations(objects, events, primary)
        constraints = self._derive_constraints(objects, events, primary)
        plan = PhysicsPlan(
            objects=objects,
            expected_events=events,
            relations=relations,
            physics_constraints=constraints,
            planner_metadata=PlannerMetadata(source="template", confidence=0.55),
        )
        plan.validate_references()
        return plan

    def _extract_objects(self, text: str) -> tuple[str, ...]:
        objects: list[str] = []
        has_red_ball = bool(self._OBJECT_PATTERNS[0].pattern.search(text))
        for item in self._OBJECT_PATTERNS:
            # “red ball/红球”是一个对象，不同时再输出泛化的 ball。
            if item.name == "ball" and has_red_ball:
                continue
            if item.pattern.search(text):
                objects.append(item.name)
        return tuple(objects)

    def _extract_events(self, text: str) -> tuple[str, ...]:
        found: set[str] = set()
        if self._EVENT_PATTERNS["fall"].search(text):
            found.update(("leave_support", "fall"))
        if self._EVENT_PATTERNS["floor_contact"].search(text):
            found.add("floor_contact")
        if self._EVENT_PATTERNS["rebound"].search(text):
            found.update(("floor_contact", "rebound"))
        if self._EVENT_PATTERNS["projectile"].search(text):
            found.add("projectile")
        if self._EVENT_PATTERNS["collision"].search(text):
            found.add("collision")
        if self._EVENT_PATTERNS["roll_down_slope"].search(text):
            found.add("roll_down_slope")
        return tuple(item for item in self._EVENT_ORDER if item in found)

    @staticmethod
    def _derive_relations(
        objects: tuple[str, ...], events: tuple[str, ...], primary: str | None
    ) -> tuple[PhysicsRelation, ...]:
        relations: list[PhysicsRelation] = []
        if primary and "table" in objects and "leave_support" in events:
            relations.append(
                PhysicsRelation("R1", primary, "initially_supported_by", "table")
            )
        if primary and "floor" in objects and "floor_contact" in events:
            relations.append(
                PhysicsRelation(
                    f"R{len(relations) + 1}",
                    primary,
                    "expected_to_collide_with",
                    "floor",
                )
            )
        if primary and "slope" in objects and "roll_down_slope" in events:
            relations.append(
                PhysicsRelation(
                    f"R{len(relations) + 1}",
                    primary,
                    "moves_down",
                    "slope",
                )
            )
        return tuple(relations)

    @staticmethod
    def _derive_constraints(
        objects: tuple[str, ...], events: tuple[str, ...], primary: str | None
    ) -> tuple[PhysicsConstraint, ...]:
        constraints: list[PhysicsConstraint] = []

        def add(
            domain: str,
            subjects: tuple[str, ...],
            expectation: str,
            condition: str | None = None,
        ) -> None:
            if subjects:
                constraints.append(
                    PhysicsConstraint(
                        id=f"C{len(constraints) + 1}",
                        domain=domain,
                        subjects=subjects,
                        expectation=expectation,
                        condition=condition,
                    )
                )

        dynamic_subject = (primary,) if primary else ()
        if "fall" in events:
            add("gravity", dynamic_subject, "downward_acceleration", "after_leave_support")
        if "floor_contact" in events:
            contact_subjects = dynamic_subject + (("floor",) if "floor" in objects else ())
            add("contact", contact_subjects, "no_interpenetration", "during_contact")
        if "rebound" in events:
            add(
                "rebound",
                dynamic_subject,
                "velocity_reversal_without_energy_gain",
                "after_contact",
            )
        if "collision" in events:
            collision_subjects = tuple(item for item in objects if item not in {"floor", "table"})
            add("collision", collision_subjects, "momentum_consistency", "during_collision")
        if "projectile" in events:
            add(
                "projectile",
                dynamic_subject,
                "parabolic_vertical_motion",
                "during_free_flight",
            )
        if "roll_down_slope" in events:
            rolling_subjects = dynamic_subject + (
                ("slope",) if "slope" in objects else ()
            )
            add(
                "rolling",
                rolling_subjects,
                "continuous_downslope_motion_without_unexplained_disappearance",
                "while_on_slope",
            )
        return tuple(constraints)


PHYSICS_PLAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["objects", "expected_events", "relations", "physics_constraints"],
    "properties": {
        "objects": {"type": "array", "items": {"type": "string"}},
        "expected_events": {
            "type": "array",
            "items": {"type": "string", "enum": list(PLAN_EVENT_VOCABULARY)},
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "subject", "relation", "object"],
                "properties": {
                    "id": {"type": "string"},
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"},
                },
            },
        },
        "physics_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "domain", "subjects", "condition", "expectation"],
                "properties": {
                    "id": {"type": "string"},
                    "domain": {"type": "string"},
                    "subjects": {"type": "array", "items": {"type": "string"}},
                    "condition": {"type": ["string", "null"]},
                    "expectation": {"type": "string"},
                },
            },
        },
    },
}


class ModelPhysicsPlanner:
    """通过现有结构化文本模型生成 PhysicsPlan，并在边界执行严格校验。"""

    def __init__(self, model: StructuredTextModel):
        self.model = model
        self.model_name = str(getattr(model, "model", type(model).__name__))

    def generate(
        self, prompt: str, partial_plan: PhysicsPlan | None = None
    ) -> PhysicsPlan:
        partial_context = (partial_plan or PhysicsPlan()).to_dict()
        partial_context.pop("planner_metadata", None)
        normalized_prompt = str(prompt or "")
        require_semantic_content = bool(normalized_prompt.strip()) and not any(
            partial_context.get(key)
            for key in (
                "objects",
                "expected_events",
                "relations",
                "physics_constraints",
            )
        )
        payload = self._generate_payload(
            prompt=normalized_prompt,
            partial_context=partial_context,
        )
        try:
            return self._parse_plan(
                payload,
                require_semantic_content=require_semantic_content,
            )
        except SchemaError as error:
            repaired = self._generate_payload(
                prompt=normalized_prompt,
                partial_context=partial_context,
                repair_feedback=str(error)[:300],
                previous_plan=payload,
            )
            try:
                return self._parse_plan(
                    repaired,
                    require_semantic_content=require_semantic_content,
                )
            except SchemaError:
                return self._parse_plan(
                    repaired,
                    require_semantic_content=require_semantic_content,
                    prune_unknown_references=True,
                )

    def _generate_payload(
        self,
        *,
        prompt: str,
        partial_context: Mapping[str, Any],
        repair_feedback: str | None = None,
        previous_plan: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        user_payload: dict[str, Any] = {
            "prompt": prompt,
            "partial_physics_plan": partial_context,
        }
        if repair_feedback is not None:
            user_payload.update(
                {
                    "repair_feedback": repair_feedback,
                    "previous_plan": previous_plan,
                    "repair_policy": (
                        "Correct schema, reference, or empty-plan errors. Preserve "
                        "supported objects and events; omit relations or constraints "
                        "that cannot use exact object identifiers."
                    ),
                }
            )
        return self.model.generate_json(
            system_prompt=(
                "Convert the video-generation prompt into a conservative physics plan. "
                "Treat every non-empty field in partial_physics_plan as authoritative and "
                "use its object identifiers when generating relations or constraints. "
                "Use stable snake_case identifiers. Include only objects and qualitative "
                "events/constraints supported by the prompt. Never invent masses, speeds, "
                "gravity constants, dimensions, coefficients, or other numeric parameters. "
                "For a non-empty prompt with no authoritative partial-plan content, "
                "extract at least one physically relevant visible entity and must not "
                "return all plan arrays empty. "
                "Every relation subject/object and every constraint subject must exactly "
                "equal one entry in objects; omit an extension if no exact reference exists. "
                "expected_events must use only these implemented IDs: "
                + ", ".join(PLAN_EVENT_VOCABULARY)
                + "."
            ),
            user_prompt=json.dumps(user_payload, ensure_ascii=False),
            schema=PHYSICS_PLAN_SCHEMA,
        )

    def _parse_plan(
        self,
        payload: Mapping[str, Any],
        *,
        require_semantic_content: bool = False,
        prune_unknown_references: bool = False,
    ) -> PhysicsPlan:
        self._validate_payload_shape(payload)
        plan = PhysicsPlan.from_dict(payload)
        if prune_unknown_references:
            known_objects = set(plan.objects)
            plan = PhysicsPlan(
                objects=plan.objects,
                expected_events=plan.expected_events,
                relations=tuple(
                    relation
                    for relation in plan.relations
                    if relation.subject in known_objects
                    and relation.object in known_objects
                ),
                physics_constraints=tuple(
                    constraint
                    for constraint in plan.physics_constraints
                    if set(constraint.subjects).issubset(known_objects)
                ),
            )
        plan.validate_references()
        if require_semantic_content and not _has_semantic_content(plan):
            raise SchemaError(
                "physics planner returned an empty plan for a non-empty prompt"
            )
        metadata = (
            PlannerMetadata(
                source="model", confidence=0.8, fallback_used=False, model=self.model_name
            )
            if _has_semantic_content(plan)
            else PlannerMetadata(source="empty", confidence=0.0, model=self.model_name)
        )
        plan = _with_metadata(
            plan,
            metadata,
        )
        return plan

    @staticmethod
    def _validate_payload_shape(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise SchemaError("physics planner model output root must be an object")
        error = next(Draft202012Validator(PHYSICS_PLAN_SCHEMA).iter_errors(payload), None)
        if error is not None:
            path = ".".join(str(item) for item in error.absolute_path)
            location = f" at {path}" if path else ""
            raise SchemaError(
                f"physics planner model output violates schema{location}: {error.message}"
            )
        required = ("objects", "expected_events", "relations", "physics_constraints")
        for key in required:
            if key not in payload:
                raise SchemaError(f"physics planner model output is missing {key!r}")
            if not isinstance(payload[key], (list, tuple)):
                raise SchemaError(f"physics planner field {key!r} must be an array")


@dataclass(frozen=True)
class PhysicsPlanResolution:
    """解析后的计划以及可选的模型失败审计记录。"""

    plan: PhysicsPlan
    provider_failure: dict[str, object] | None = None


_PROVIDER_ERRORS = (
    ModelAPIError,
    TimeoutError,
    ConnectionError,
    OSError,
    SchemaError,
    KeyError,
    ValueError,
    TypeError,
)


class PhysicsPlanResolver:
    """执行显式计划优先、空字段补全和模型失败降级。"""

    def __init__(
        self,
        planner: PhysicsPlanner,
        fallback: PhysicsPlanner | None = None,
        fallback_on_provider_error: bool = False,
    ):
        if fallback_on_provider_error and fallback is None:
            raise ValueError("fallback planner is required when provider fallback is enabled")
        self.planner = planner
        self.fallback = fallback
        self.fallback_on_provider_error = fallback_on_provider_error

    def resolve(self, request: CriticRequest) -> PhysicsPlanResolution:
        explicit = request.physics_plan
        explicit.validate_references()
        if explicit.objects and explicit.expected_events:
            return PhysicsPlanResolution(
                _with_metadata(
                    explicit,
                    PlannerMetadata(source="explicit", confidence=1.0),
                )
            )

        try:
            generated = self.planner.generate(request.prompt, partial_plan=explicit)
        except _PROVIDER_ERRORS as error:
            if not self.fallback_on_provider_error:
                raise
            assert self.fallback is not None
            fallback_plan = self.fallback.generate(
                request.prompt, partial_plan=explicit
            )
            model_name = getattr(self.planner, "model_name", None)
            fallback_metadata = (
                PlannerMetadata(
                    source="template_fallback",
                    confidence=0.4,
                    fallback_used=True,
                    model=None if model_name is None else str(model_name),
                )
                if _has_semantic_content(fallback_plan)
                else PlannerMetadata(
                    source="empty",
                    confidence=0.0,
                    fallback_used=True,
                    model=None if model_name is None else str(model_name),
                )
            )
            generated = _with_metadata(
                fallback_plan,
                fallback_metadata,
            )
            return PhysicsPlanResolution(
                plan=_merge_plans(explicit, generated),
                provider_failure=_failure_record(error),
            )

        return PhysicsPlanResolution(plan=_merge_plans(explicit, generated))


def _with_metadata(plan: PhysicsPlan, metadata: PlannerMetadata) -> PhysicsPlan:
    return PhysicsPlan(
        objects=plan.objects,
        expected_events=plan.expected_events,
        relations=plan.relations,
        physics_constraints=plan.physics_constraints,
        planner_metadata=metadata,
    )


def _merge_plans(explicit: PhysicsPlan, generated: PhysicsPlan) -> PhysicsPlan:
    """按字段合并，显式非空字段和同 ID 扩展始终优先。"""

    objects = explicit.objects or generated.objects
    events = explicit.expected_events or generated.expected_events
    known = set(objects)
    generated_relations = tuple(
        item
        for item in generated.relations
        if item.subject in known and item.object in known
    )
    generated_constraints = tuple(
        item
        for item in generated.physics_constraints
        if set(item.subjects).issubset(known)
    )
    relations = _merge_extensions(generated_relations, explicit.relations)
    constraints = _merge_extensions(
        generated_constraints, explicit.physics_constraints
    )

    explicit_has_content = _has_semantic_content(explicit)
    generated_contributed = bool(
        (not explicit.objects and generated.objects)
        or (not explicit.expected_events and generated.expected_events)
        or generated_relations
        or generated_constraints
    )
    if not explicit_has_content:
        metadata = generated.planner_metadata
    elif generated_contributed:
        metadata = PlannerMetadata(
            source="merged",
            confidence=generated.planner_metadata.confidence,
            fallback_used=generated.planner_metadata.fallback_used,
            model=generated.planner_metadata.model,
        )
    else:
        metadata = PlannerMetadata(
            source="explicit",
            confidence=1.0,
            fallback_used=generated.planner_metadata.fallback_used,
            model=(
                generated.planner_metadata.model
                if generated.planner_metadata.fallback_used
                else None
            ),
        )

    result = PhysicsPlan(
        objects=objects,
        expected_events=events,
        relations=relations,
        physics_constraints=constraints,
        planner_metadata=metadata,
    )
    result.validate_references()
    return result


def _merge_extensions(generated: tuple[Any, ...], explicit: tuple[Any, ...]) -> tuple[Any, ...]:
    """保持生成顺序；显式同 ID 项替换生成项，显式新项追加到末尾。"""

    explicit_by_id = {item.id: item for item in explicit}
    result = [explicit_by_id.get(item.id, item) for item in generated]
    generated_ids = {item.id for item in generated}
    result.extend(item for item in explicit if item.id not in generated_ids)
    return tuple(result)


def _has_semantic_content(plan: PhysicsPlan) -> bool:
    return bool(
        plan.objects
        or plan.expected_events
        or plan.relations
        or plan.physics_constraints
    )


def _failure_record(error: Exception) -> dict[str, object]:
    return {
        "stage": "physics_planner",
        "error_type": type(error).__name__,
        "message": str(error)[:300],
        "fallback": "template_physics_planner",
    }
