"""Deterministic prompt repair driven only by the current CriticReport."""

from __future__ import annotations

import re
from typing import Any


_BEGIN = "[PHYSICS_CORRECTION_BEGIN]"
_END = "[PHYSICS_CORRECTION_END]"
_CATEGORY_FALLBACK = {
    "gravity_violation": "Preserve consistent downward gravity and physically plausible acceleration.",
    "friction_violation": "Use physically plausible friction and gradual deceleration at contact.",
    "contact_violation": "Maintain non-penetrating contact and continuous support forces.",
    "collision_violation": "Use momentum-consistent collision response without interpenetration.",
    "trajectory_violation": "Keep the object's trajectory continuous with bounded acceleration.",
    "continuity_violation": "Preserve temporal continuity of object shape, position, and motion.",
}


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


class InstructionPromptRepairer:
    """Replace one bounded correction block without changing the user's task."""

    def __init__(self, *, max_instructions: int = 4, max_chars: int = 900) -> None:
        self.max_instructions = max(1, int(max_instructions))
        self.max_chars = max(128, int(max_chars))

    def repair(self, *, prompt: str, report: Any) -> str:
        original = str(prompt).strip()
        violations = tuple(_value(report, "violations", ()) or ())
        ranked = sorted(
            violations,
            key=lambda item: (
                -float(_value(item, "confidence", 0.0) or 0.0),
                str(_value(item, "object", "")),
                str(_value(item, "category", "")),
            ),
        )
        instructions: list[str] = []
        for violation in ranked:
            instruction = str(_value(violation, "repair_instruction", "") or "").strip()
            category = str(
                _value(violation, "category", _value(violation, "type", "")) or ""
            ).strip()
            if not instruction:
                instruction = _CATEGORY_FALLBACK.get(category, "")
            if not instruction:
                continue
            obj = str(_value(violation, "object", "") or "").strip()
            frames = tuple(_value(violation, "critical_frames", ()) or ())
            phase = ""
            if frames:
                phase = f" around frames {min(frames)}-{max(frames)}"
            text = f"{obj}{phase}: {instruction}" if obj else f"{instruction}{phase}"
            normalized = " ".join(text.split())
            if normalized and normalized not in instructions:
                instructions.append(normalized)
            if len(instructions) >= self.max_instructions:
                break
        if not instructions:
            return original

        base = re.sub(
            rf"\n?{re.escape(_BEGIN)}.*?{re.escape(_END)}",
            "",
            original,
            flags=re.DOTALL,
        ).strip()
        block = (
            f"{_BEGIN}\n"
            "Keep the original objects, scene, action, camera, and visual style. "
            "Apply only these physical corrections:\n- "
            + "\n- ".join(instructions)
            + f"\n{_END}"
        )
        repaired = f"{base}\n{block}".strip()
        if len(repaired) > self.max_chars:
            repaired = repaired[: self.max_chars].rsplit(" ", 1)[0].rstrip()
            if _END not in repaired:
                repaired = f"{repaired}\n{_END}"
        return repaired
