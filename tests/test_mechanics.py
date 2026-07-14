"""Morpheus 启发的四类力学评估与适用性门控。"""

from __future__ import annotations

from pavg_critic.mechanics import MechanicsEvaluator
from pavg_critic import PhysicsCritic
from pavg_critic.config import CriticConfig, MechanicsConfig, TrajectoryConfig
from pavg_critic.schemas import (
    CriticRequest,
    Event,
    FrameState,
    PhysicsPlan,
    TrackSequence,
)


def _track(
    points: tuple[tuple[float, float], ...],
    velocities: tuple[tuple[float, float] | None, ...] | None = None,
    track_id: str = "ball-1",
) -> TrackSequence:
    velocities = velocities or (None,) * len(points)
    states = tuple(
        FrameState(
            frame=index,
            timestamp_sec=float(index),
            object="ball",
            track_id=track_id,
            center=point,
            bbox=(point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1),
            velocity=velocities[index],
        )
        for index, point in enumerate(points)
    )
    return TrackSequence(track_id=track_id, object="ball", states=states)


def test_freefall_quadratic_motion_has_high_dynamical_score():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("ball",), expected_events=("fall",)),
    )
    track = _track(((0, 0), (0, 1), (0, 4), (0, 9), (0, 16)))

    results, summary = MechanicsEvaluator(MechanicsConfig()).evaluate(
        request=request,
        tracks=(track,),
        events=(),
    )

    freefall = next(item for item in results if item.evaluator == "freefall")
    assert freefall.applicability == "applicable"
    assert freefall.dynamical_score is not None
    assert freefall.dynamical_score > 0.95
    assert freefall.is_plausible is True
    assert summary.score is not None


def test_irregular_fall_scores_below_quadratic_motion():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(expected_events=("fall",)),
    )
    physical = _track(((0, 0), (0, 1), (0, 4), (0, 9), (0, 16)))
    irregular = _track(((0, 0), (0, 8), (0, 2), (0, 20), (0, 7)))
    evaluator = MechanicsEvaluator(MechanicsConfig())

    physical_result = evaluator.evaluate(request=request, tracks=(physical,), events=())[0][0]
    irregular_result = evaluator.evaluate(request=request, tracks=(irregular,), events=())[0][0]

    assert physical_result.score > irregular_result.score


def test_projectile_linear_x_and_quadratic_y_scores_high():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(expected_events=("projectile",)),
    )
    track = _track(((0, 0), (2, 1), (4, 4), (6, 9), (8, 16)))

    results, _ = MechanicsEvaluator(MechanicsConfig()).evaluate(
        request=request,
        tracks=(track,),
        events=(),
    )

    projectile = next(item for item in results if item.evaluator == "projectile")
    assert projectile.applicability == "applicable"
    assert projectile.score > 0.95


def test_rebound_uses_contact_and_velocity_reversal():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(expected_events=("rebound",)),
    )
    track = _track(
        ((0, 0), (0, 5), (0, 9), (0, 6)),
        velocities=((0, 5), (0, 5), (0, -4), (0, -4)),
    )
    events = (
        Event("floor_contact", "ball", "ball-1", 1, 1, 1, 0.9),
        Event("rebound", "ball", "ball-1", 2, 2, 2, 0.9),
    )

    results, _ = MechanicsEvaluator(MechanicsConfig()).evaluate(
        request=request,
        tracks=(track,),
        events=events,
    )

    rebound = next(item for item in results if item.evaluator == "rebound")
    assert rebound.applicability == "applicable"
    assert rebound.score > 0.7
    assert rebound.metrics["restitution_ratio"] == 0.8


def test_collision_is_not_applicable_without_two_tracks():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(expected_events=("collision",)),
    )

    results, _ = MechanicsEvaluator(MechanicsConfig()).evaluate(
        request=request,
        tracks=(_track(((0, 0), (1, 0), (2, 0))),),
        events=(),
    )

    collision = next(item for item in results if item.evaluator == "collision")
    assert collision.applicability == "not_applicable"
    assert collision.score is None


def test_equal_mass_collision_conserves_image_plane_momentum():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(expected_events=("collision",)),
    )
    first = _track(
        ((0, 0), (1, 0), (2, 0), (1, 0), (0, 0)),
        velocities=((1, 0), (1, 0), (-1, 0), (-1, 0), (-1, 0)),
        track_id="ball-1",
    )
    second = _track(
        ((4, 0), (3, 0), (2, 0), (3, 0), (4, 0)),
        velocities=((-1, 0), (-1, 0), (1, 0), (1, 0), (1, 0)),
        track_id="ball-2",
    )

    results, _ = MechanicsEvaluator(MechanicsConfig()).evaluate(
        request=request,
        tracks=(first, second),
        events=(),
    )

    collision = next(item for item in results if item.evaluator == "collision")
    assert collision.applicability == "applicable"
    assert collision.invariance_score == 1.0
    assert collision.is_plausible is True


def test_pipeline_attaches_mechanics_diagnostics():
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("ball",), expected_events=("fall",)),
    )
    observations = _track(((0, 0), (0, 1), (0, 4), (0, 9), (0, 16))).states
    critic = PhysicsCritic(
        CriticConfig(trajectory=TrajectoryConfig(smoothing_window=1))
    )

    artifacts = critic.analyze_detailed(
        request,
        observations=observations,
        floor_y=100,
    )

    assert len(artifacts.mechanics_results) == 4
    assert artifacts.mechanics_summary is not None
    assert "morpheus_mechanics" in artifacts.report.diagnostics
