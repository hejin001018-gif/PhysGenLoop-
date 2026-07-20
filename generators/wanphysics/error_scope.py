"""基于 critical_frames 占比的错误范围启发式。"""

from __future__ import annotations

from typing import Any, Mapping


def _violations(report: Any) -> tuple[Any, ...]:
    return tuple(getattr(report, "violations", ()) or ())


def extract_critical_frames(report: Any) -> tuple[int, ...]:
    frames: set[int] = set()
    for violation in _violations(report):
        for frame in getattr(violation, "critical_frames", ()) or ():
            try:
                value = int(frame)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                frames.add(value)
    return tuple(sorted(frames))


def has_local_editing_evidence(report: Any) -> bool:
    for violation in _violations(report):
        evidence = getattr(violation, "evidence", {}) or {}
        if getattr(violation, "critical_frames", ()) or (
            isinstance(evidence, Mapping) and evidence.get("mask_uri")
        ):
            return True
    return False


def classify_error_scope(report: Any, total_frames: int, local_threshold: float = 0.4) -> str:
    if total_frames <= 0:
        return "global"
    critical_frames = extract_critical_frames(report)
    ratio = len(critical_frames) / float(total_frames)
    if ratio == 0.0 or ratio >= float(local_threshold):
        return "global"
    return "local"
