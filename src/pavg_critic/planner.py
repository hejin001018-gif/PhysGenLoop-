"""Prompt 到 PhysicsPlan 的确定性与模型规划入口。

模板实现只提取 prompt 明确表达的对象、事件和定性约束。它不猜测质量、速度、
重力常数等数值，因此可作为无 API 环境的安全默认值及模型失败时的降级实现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import (
    PhysicsConstraint,
    PhysicsPlan,
    PhysicsRelation,
    PlannerMetadata,
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
    }
    _EVENT_ORDER = (
        "leave_support",
        "fall",
        "floor_contact",
        "rebound",
        "projectile",
        "collision",
    )
    _STATIC_OBJECTS = frozenset({"table", "floor", "wall"})

    def generate(self, prompt: str) -> PhysicsPlan:
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
        return tuple(constraints)
