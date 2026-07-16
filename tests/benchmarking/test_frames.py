import pytest

from pavg_critic.benchmarking.frames import sample_video_frames, uniform_indices


@pytest.mark.parametrize(
    ("total", "count", "expected"),
    [
        (1, 8, (0,)),
        (5, 5, (0, 1, 2, 3, 4)),
        (10, 3, (0, 4, 9)),
        (10, 1, (0,)),
    ],
)
def test_uniform_indices_cover_endpoints(total, count, expected):
    assert uniform_indices(total, count) == expected


def test_uniform_indices_reject_invalid_counts():
    with pytest.raises(ValueError, match="positive"):
        uniform_indices(10, 0)


def test_sample_video_frames_decodes_requested_frames(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "five.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 64),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for value in range(5):
        writer.write(np.full((64, 64, 3), value * 40, dtype=np.uint8))
    writer.release()

    result = sample_video_frames(str(path), count=3)
    assert result.indices == (0, 2, 4)
    assert len(result.data_urls) == 3
    assert all(item.startswith("data:image/jpeg;base64,") for item in result.data_urls)
    assert result.fps > 0
