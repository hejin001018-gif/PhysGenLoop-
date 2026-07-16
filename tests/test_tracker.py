from pavg_critic.config import TrackerConfig
from pavg_critic.schemas import Detection
from pavg_critic.tracker import CentroidTracker


def _detection(frame: int, x: float, *, track_id: str | None = None) -> Detection:
    return Detection(
        frame=frame,
        timestamp_sec=frame / 10,
        object="ball",
        center=(x, 20.0),
        bbox=(x - 5, 15.0, x + 5, 25.0),
        track_id=track_id,
    )


def test_backend_track_id_survives_jump_beyond_centroid_threshold():
    tracker = CentroidTracker(TrackerConfig(max_match_distance_px=10.0))
    states = tracker.track_timed(
        (
            (0, 0.0, (_detection(0, 10.0, track_id="sam2:0"),)),
            (1, 0.1, (_detection(1, 100.0, track_id="sam2:0"),)),
        )
    )
    assert [state.track_id for state in states] == ["sam2:0", "sam2:0"]
    assert all(state.visible for state in states)


def test_backend_track_id_rejoins_after_short_missing_interval():
    tracker = CentroidTracker(TrackerConfig(max_missed_frames=3))
    states = tracker.track_timed(
        (
            (0, 0.0, (_detection(0, 10.0, track_id="sam2:0"),)),
            (1, 0.1, ()),
            (2, 0.2, (_detection(2, 100.0, track_id="sam2:0"),)),
        )
    )
    assert [state.track_id for state in states] == ["sam2:0"] * 3
    assert [state.visible for state in states] == [True, False, True]
