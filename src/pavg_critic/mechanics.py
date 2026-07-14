"""Morpheus 启发的轻量力学评估器。

Morpheus 的关键思想被拆成两条可解释分数：滑窗/守恒性质的 invariance score，以及
基于动力学方程拟合残差的 dynamical score。这里首批覆盖自由落体、抛体、反弹和
碰撞；所有评估器先做现象门控，证据不够时返回 not_applicable 而不是伪造低分。
"""

from __future__ import annotations

from math import hypot
from statistics import fmean

from .config import MechanicsConfig
from .schemas import (
    MECHANICS_EVALUATORS,
    CriticRequest,
    Event,
    FrameState,
    MechanicsResult,
    MechanicsSummary,
    TrackSequence,
)


class MechanicsEvaluator:
    """按固定顺序运行四个力学假设并生成覆盖感知摘要。"""

    def __init__(self, config: MechanicsConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        request: CriticRequest,
        tracks: tuple[TrackSequence, ...],
        events: tuple[Event, ...],
    ) -> tuple[tuple[MechanicsResult, ...], MechanicsSummary]:
        methods = (
            ("freefall", self._freefall),
            ("projectile", self._projectile),
            ("rebound", self._rebound),
            ("collision", self._collision),
        )
        results: list[MechanicsResult] = []
        for name, method in methods:
            try:
                results.append(method(request, tracks, events))
            except (ArithmeticError, ValueError) as exc:
                # 数值退化属于该评估器失败，不应使规则/PQSG 主流水线整体崩溃。
                results.append(
                    MechanicsResult(
                        evaluator=name,
                        applicability="failed",
                        invariance_score=None,
                        dynamical_score=None,
                        score=None,
                        is_plausible=None,
                        reason=f"Mechanics computation failed: {exc}",
                    )
                )
        applicable = [item for item in results if item.applicability == "applicable"]
        summary = MechanicsSummary(
            score=(fmean(item.score for item in applicable) if applicable else None),
            coverage=len(applicable) / len(MECHANICS_EVALUATORS),
            applicable=len(applicable),
            not_applicable=sum(
                item.applicability == "not_applicable" for item in results
            ),
            failed=sum(item.applicability == "failed" for item in results),
        )
        return tuple(results), summary

    def _freefall(self, request, tracks, events) -> MechanicsResult:
        expected = set(request.physics_plan.expected_events)
        if not expected.intersection({"fall", "leave_support"}):
            return _not_applicable("freefall", "The plan does not declare free fall.")
        states = _longest_visible_track(tracks, self.config.min_points)
        if states is None:
            return _not_applicable("freefall", "Too few visible trajectory points.")
        times = [state.timestamp_sec for state in states]
        y = [state.center[1] for state in states]
        coefficients, nmse = _polynomial_fit_score(times, y, degree=2)
        accelerations = _second_derivatives(times, y)
        invariance = _constancy_score(accelerations)
        # 图像坐标向下为正；负二次项与重力下落方向矛盾。
        dynamical = 1.0 - min(nmse, 1.0) if coefficients[0] > 0 else 0.0
        return self._result(
            "freefall",
            invariance,
            dynamical,
            "Quadratic vertical motion and acceleration invariance were evaluated.",
            states,
            {
                "quadratic_coefficient": coefficients[0],
                "nmse": nmse,
                "acceleration_samples": accelerations,
            },
        )

    def _projectile(self, request, tracks, events) -> MechanicsResult:
        expected = set(request.physics_plan.expected_events)
        if not expected.intersection({"projectile", "throw", "parabolic_motion"}):
            return _not_applicable("projectile", "The plan does not declare projectile motion.")
        states = _longest_visible_track(tracks, self.config.min_points)
        if states is None:
            return _not_applicable("projectile", "Too few visible trajectory points.")
        times = [state.timestamp_sec for state in states]
        x = [state.center[0] for state in states]
        y = [state.center[1] for state in states]
        if max(x) - min(x) <= 1e-9:
            return _not_applicable("projectile", "No measurable horizontal displacement.")
        _, x_nmse = _polynomial_fit_score(times, x, degree=1)
        y_coefficients, y_nmse = _polynomial_fit_score(times, y, degree=2)
        horizontal_velocities = _first_derivatives(times, x)
        invariance = _constancy_score(horizontal_velocities)
        dynamical = 1.0 - min((x_nmse + y_nmse) / 2.0, 1.0)
        if y_coefficients[0] <= 0:
            dynamical = 0.0
        return self._result(
            "projectile",
            invariance,
            dynamical,
            "Horizontal invariance and a parabolic vertical trajectory were evaluated.",
            states,
            {"horizontal_nmse": x_nmse, "vertical_nmse": y_nmse},
        )

    def _rebound(self, request, tracks, events) -> MechanicsResult:
        if "rebound" not in set(request.physics_plan.expected_events):
            return _not_applicable("rebound", "The plan does not declare a rebound.")
        rebounds = [event for event in events if event.event_type == "rebound"]
        if not rebounds:
            return _not_applicable("rebound", "No rebound event was detected.")
        rebound = rebounds[0]
        track = next((item for item in tracks if item.track_id == rebound.track_id), None)
        if track is None:
            return _not_applicable("rebound", "The rebound track is unavailable.")
        before = [
            state
            for state in track.states
            if state.frame < rebound.start_frame
            and state.velocity is not None
            and state.velocity[1] > 0
        ]
        after = [
            state
            for state in track.states
            if state.frame >= rebound.start_frame
            and state.velocity is not None
            and state.velocity[1] < 0
        ]
        if not before or not after:
            return _not_applicable("rebound", "Pre/post impact velocities are unavailable.")
        speed_before = abs(before[-1].velocity[1])
        speed_after = abs(after[0].velocity[1])
        if speed_before <= 1e-12:
            raise ValueError("pre-impact speed is zero")
        restitution = speed_after / speed_before
        invariance = 1.0 if restitution <= 1.0 else max(0.0, 2.0 - restitution)
        contacts = [
            event
            for event in events
            if event.event_type == "floor_contact"
            and event.track_id == rebound.track_id
            and 0
            <= rebound.start_frame - event.end_frame
            <= self.config.contact_lookback_frames
        ]
        dynamical = 1.0 if contacts else 0.0
        states = (before[-1], after[0])
        return self._result(
            "rebound",
            invariance,
            dynamical,
            "Restitution bounds, velocity reversal and contact ordering were evaluated.",
            states,
            {"restitution_ratio": restitution, "contact_count": len(contacts)},
        )

    def _collision(self, request, tracks, events) -> MechanicsResult:
        if "collision" not in set(request.physics_plan.expected_events):
            return _not_applicable("collision", "The plan does not declare a collision.")
        if len(tracks) < 2:
            return _not_applicable("collision", "At least two tracked objects are required.")
        first, second = tracks[:2]
        pairs = _same_frame_pairs(first.states, second.states)
        if len(pairs) < 3:
            return _not_applicable("collision", "Too few synchronized two-object states.")
        collision_index = min(
            range(len(pairs)),
            key=lambda index: hypot(
                pairs[index][0].center[0] - pairs[index][1].center[0],
                pairs[index][0].center[1] - pairs[index][1].center[1],
            ),
        )
        if collision_index == 0 or collision_index >= len(pairs) - 1:
            return _not_applicable("collision", "No pre/post closest-approach window exists.")
        closest = pairs[collision_index]
        has_collision_event = any(
            event.event_type == "collision"
            and event.track_id in {first.track_id, second.track_id}
            for event in events
        )
        if not has_collision_event and not _boxes_overlap(closest[0], closest[1]):
            return _not_applicable(
                "collision",
                "The trajectories have no collision event or bounding-box contact/overlap.",
            )
        pre = pairs[collision_index - 1]
        post = pairs[collision_index + 1]
        if any(state.velocity is None for state in pre + post):
            return _not_applicable("collision", "Pre/post collision velocities are unavailable.")
        momentum_before = (
            pre[0].velocity[0] + pre[1].velocity[0],
            pre[0].velocity[1] + pre[1].velocity[1],
        )
        momentum_after = (
            post[0].velocity[0] + post[1].velocity[0],
            post[0].velocity[1] + post[1].velocity[1],
        )
        residual = (momentum_after[0] - momentum_before[0]) ** 2 + (
            momentum_after[1] - momentum_before[1]
        ) ** 2
        scale = momentum_before[0] ** 2 + momentum_before[1] ** 2 + 1e-9
        momentum_nmse = residual / scale
        invariance = 1.0 - min(momentum_nmse, 1.0)
        rel_before = (
            pre[0].velocity[0] - pre[1].velocity[0],
            pre[0].velocity[1] - pre[1].velocity[1],
        )
        rel_after = (
            post[0].velocity[0] - post[1].velocity[0],
            post[0].velocity[1] - post[1].velocity[1],
        )
        dynamical = 1.0 if rel_before != rel_after else 0.5
        states = (pre[0], pre[1], post[0], post[1])
        return self._result(
            "collision",
            invariance,
            dynamical,
            "Equal-mass image-plane momentum and relative velocity change were evaluated.",
            states,
            {"momentum_nmse": momentum_nmse},
        )

    def _result(self, evaluator, invariance, dynamical, reason, states, metrics):
        score = (invariance + dynamical) / 2.0
        return MechanicsResult(
            evaluator=evaluator,
            applicability="applicable",
            invariance_score=invariance,
            dynamical_score=dynamical,
            score=score,
            is_plausible=score >= self.config.plausible_threshold,
            reason=reason,
            critical_frames=tuple(dict.fromkeys(state.frame for state in states)),
            metrics=metrics,
        )


def _not_applicable(evaluator: str, reason: str) -> MechanicsResult:
    return MechanicsResult(
        evaluator=evaluator,
        applicability="not_applicable",
        invariance_score=None,
        dynamical_score=None,
        score=None,
        is_plausible=None,
        reason=reason,
    )


def _longest_visible_track(
    tracks: tuple[TrackSequence, ...], minimum: int
) -> tuple[FrameState, ...] | None:
    candidates = [tuple(state for state in track.states if state.visible) for track in tracks]
    candidates = [states for states in candidates if len(states) >= minimum]
    return max(candidates, key=len) if candidates else None


def _first_derivatives(times: list[float], values: list[float]) -> list[float]:
    result = []
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if dt <= 0:
            raise ValueError("trajectory timestamps must increase")
        result.append((values[index] - values[index - 1]) / dt)
    return result


def _second_derivatives(times: list[float], values: list[float]) -> list[float]:
    velocities = _first_derivatives(times, values)
    result = []
    for index in range(1, len(velocities)):
        dt = (times[index + 1] - times[index - 1]) / 2.0
        if dt <= 0:
            raise ValueError("trajectory timestamps must increase")
        result.append((velocities[index] - velocities[index - 1]) / dt)
    return result


def _constancy_score(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = fmean(values)
    scale = max(fmean(abs(value) for value in values), 1e-9)
    dispersion = fmean(abs(value - mean) for value in values) / scale
    return 1.0 - min(dispersion, 1.0)


def _polynomial_fit_score(
    times: list[float], values: list[float], *, degree: int
) -> tuple[list[float], float]:
    # 系数按高次到常数项返回；正规方程规模最多 3x3，标准库即可稳定处理受控短轨迹。
    powers = list(range(degree, -1, -1))
    matrix = [[sum(t ** (left + right) for t in times) for right in powers] for left in powers]
    vector = [sum(value * t**power for t, value in zip(times, values)) for power in powers]
    coefficients = _solve_linear_system(matrix, vector)
    predicted = [
        sum(coefficient * t**power for coefficient, power in zip(coefficients, powers))
        for t in times
    ]
    residual = sum((actual - estimate) ** 2 for actual, estimate in zip(values, predicted))
    mean = fmean(values)
    variance = sum((value - mean) ** 2 for value in values)
    nmse = residual / max(variance, 1e-9)
    return coefficients, nmse


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("trajectory fit is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _same_frame_pairs(
    first: tuple[FrameState, ...], second: tuple[FrameState, ...]
) -> list[tuple[FrameState, FrameState]]:
    second_by_frame = {state.frame: state for state in second if state.visible}
    return [
        (state, second_by_frame[state.frame])
        for state in first
        if state.visible and state.frame in second_by_frame
    ]


def _boxes_overlap(first: FrameState, second: FrameState) -> bool:
    """将 bbox 接触（边界相等）也视为候选碰撞门控证据。"""

    return (
        max(first.bbox[0], second.bbox[0]) <= min(first.bbox[2], second.bbox[2])
        and max(first.bbox[1], second.bbox[1]) <= min(first.bbox[3], second.bbox[3])
    )
