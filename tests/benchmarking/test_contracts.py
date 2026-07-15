from pathlib import Path

import pytest

from pavg_critic.benchmarking.contracts import BenchmarkPrediction, BenchmarkSample


def test_sample_requires_existing_local_video(tmp_path: Path):
    with pytest.raises(ValueError, match="video does not exist"):
        BenchmarkSample(
            sample_id="vp2-1",
            benchmark="videophy2",
            split="test",
            prompt="A ball rolls down a ramp.",
            video_path=str(tmp_path / "missing.mp4"),
            prompt_group_id="rolling_ball",
            generator="model-a",
            semantic_label="adherent",
            physics_label="physical",
        )


def test_prediction_round_trip_preserves_unknown_and_failure():
    prediction = BenchmarkPrediction(
        sample_id="vp2-1",
        method_id="D0_DIRECT_VLM",
        model_id="fake-vlm",
        semantic_score=None,
        physics_score=None,
        semantic_label="unknown",
        physics_label="unknown",
        confidence=0.0,
        coverage=0.0,
        latency_sec=1.25,
        visible_frame_count=8,
        violation_categories=(),
        evidence_frames=(),
        repair_instruction=None,
        failure={"type": "TimeoutError", "message": "timed out"},
    )
    assert BenchmarkPrediction.from_dict(prediction.to_dict()) == prediction
