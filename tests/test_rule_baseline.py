"""冻结 PAVG 原有确定性规则 Critic 的行为。

这些测试使用显式速度和几何真值，避免检测器、视频编码和轨迹平滑掩盖规则语义。
后续问题图、力学模型和 VLM 只能增强证据，不能破坏这里的五类基础能力。
"""

from __future__ import annotations

import pytest

from pavg_critic import CriticRequest, FrameState, PhysicsCritic
from pavg_critic.config import CriticConfig, QuestionGraphConfig, TrajectoryConfig
from pavg_critic.schemas import PhysicsPlan


def _state(
    frame: int,
    y: float,
    *,
    velocity: tuple[float, float] | None = None,
    visible: bool = True,
    bottom: float | None = None,
) -> FrameState:
    """构造一个带稳定身份的红球状态；测试坐标中的地面位于 y=100。"""

    bottom = y + 5 if bottom is None else bottom
    return FrameState(
        frame=frame,
        timestamp_sec=frame / 10,
        object="red_ball",
        track_id="ball-1",
        center=(50.0, y),
        bbox=(45.0, bottom - 10, 55.0, bottom),
        visible=visible,
        velocity=velocity,
    )


def _critic() -> PhysicsCritic:
    """关闭问题图，仅刻画迁移前的规则基线。"""

    return PhysicsCritic(
        CriticConfig(
            trajectory=TrajectoryConfig(smoothing_window=1),
            question_graph=QuestionGraphConfig(enabled=False),
        )
    )


@pytest.mark.parametrize(
    ("expected_category", "states", "expected_events"),
    [
        (
            "premature_rebound",
            (
                _state(0, 40, velocity=(0, 100)),
                _state(1, 55, velocity=(0, 100)),
                _state(2, 48, velocity=(0, -100)),
                _state(3, 38, velocity=(0, -100)),
            ),
            ("fall",),
        ),
        (
            "surface_penetration",
            (_state(0, 103, velocity=(0, 20), bottom=108),),
            ("floor_contact",),
        ),
        (
            "object_disappearance",
            (
                _state(0, 50, visible=True),
                _state(1, 50, visible=False),
                _state(2, 50, visible=False),
                _state(3, 50, visible=False),
            ),
            (),
        ),
        (
            "reverse_gravity",
            (
                _state(0, 70, velocity=(0, -100)),
                _state(1, 60, velocity=(0, -100)),
                _state(2, 50, velocity=(0, -100)),
            ),
            ("fall",),
        ),
        (
            "teleportation",
            (_state(0, 50, velocity=(2000, 0)),),
            (),
        ),
    ],
)
def test_existing_rule_detects_violation(expected_category, states, expected_events):
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=expected_events),
    )

    report = _critic().analyze(request, observations=states, floor_y=100)

    assert expected_category in {item.category for item in report.violations}
    assert report.is_physical is False


def test_contact_then_rebound_remains_physical():
    states = (
        _state(0, 85, velocity=(0, 100), bottom=90),
        _state(1, 95, velocity=(0, 100), bottom=100),
        _state(2, 92, velocity=(0, -100), bottom=100),
        _state(3, 80, velocity=(0, -100), bottom=85),
    )
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=("fall", "rebound")),
    )

    report = _critic().analyze(request, observations=states, floor_y=100)

    assert report.is_physical is True
    assert report.violations == ()


def test_unplanned_motion_reversal_is_not_a_rebound_violation():
    states = (
        _state(0, 40, velocity=(0, 100)),
        _state(1, 55, velocity=(0, 100)),
        _state(2, 48, velocity=(0, -100)),
        _state(3, 38, velocity=(0, -100)),
    )
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=()),
    )

    report = _critic().analyze(request, observations=states, floor_y=100)

    assert "premature_rebound" not in {
        item.category for item in report.violations
    }


def test_unplanned_floor_geometry_is_not_surface_penetration_evidence():
    states = (_state(0, 103, velocity=(0, 20), bottom=108),)
    request = CriticRequest(
        video_path="unused.mp4",
        physics_plan=PhysicsPlan(objects=("red_ball",), expected_events=()),
    )

    report = _critic().analyze(request, observations=states, floor_y=100)

    assert "surface_penetration" not in {
        item.category for item in report.violations
    }
