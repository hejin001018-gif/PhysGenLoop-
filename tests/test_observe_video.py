import numpy as np
import pytest

from pavg_critic.pipeline import PhysicsCritic
from pavg_critic.schemas import Detection


class StableDetector:
    def detect(self, frame_image, frame_index, timestamp_sec):
        x = 10.0 + frame_index * 20.0
        return (
            Detection(
                frame=frame_index,
                timestamp_sec=timestamp_sec,
                object="ball",
                center=(x, 20.0),
                bbox=(x - 5, 15.0, x + 5, 25.0),
                track_id="backend:0",
            ),
        )


def test_observe_video_returns_raw_backend_tracked_states(tmp_path):
    import cv2

    path = tmp_path / "three.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 64),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for _ in range(3):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    states, floor_y = PhysicsCritic(detector=StableDetector()).observe_video(str(path))

    assert [state.track_id for state in states] == ["backend:0"] * 3
    assert all(state.velocity is None for state in states)
    assert all(state.distance_to_floor is None for state in states)
    assert floor_y > 0
