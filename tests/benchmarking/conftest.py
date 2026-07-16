import pytest

from pavg_critic.benchmarking.contracts import BenchmarkPrediction, BenchmarkSample
from pavg_critic.schemas import FrameState


@pytest.fixture
def sample_factory(tmp_path):
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"fixture")

    def make(*, index: int, physical: bool, generator: str):
        return BenchmarkSample(
            sample_id=str(index),
            benchmark="videophy2",
            split="test",
            prompt=f"prompt {index}",
            video_path=str(video),
            prompt_group_id=f"action-{index // 2}",
            generator=generator,
            semantic_label="adherent",
            physics_label="physical" if physical else "violation",
            semantic_score=5.0,
            physics_score=5.0 if physical else 2.0,
        )

    return make


@pytest.fixture
def prediction_factory():
    def make(sample_id: str, label: str, score: float | None, *, method_id: str = "D0_DIRECT_VLM"):
        return BenchmarkPrediction(
            sample_id=sample_id,
            method_id=method_id,
            model_id="fake",
            semantic_score=5.0,
            physics_score=score,
            semantic_label="adherent",
            physics_label=label,
            confidence=0.8 if label != "unknown" else 0.0,
            coverage=1.0 if label != "unknown" else 0.0,
            latency_sec=0.1,
            visible_frame_count=4,
        )

    return make


@pytest.fixture
def frame_state_factory():
    def make():
        return FrameState(
            frame=0,
            timestamp_sec=0.0,
            object="ball",
            center=(10.0, 10.0),
            bbox=(5.0, 5.0, 15.0, 15.0),
            track_id="ball-1",
        )

    return make
