"""关键帧约束的证据化 VLM 复核。"""

from __future__ import annotations

from pavg_critic.schemas import CriticRequest, ViolationCandidate
from pavg_critic.vlm_verifier import (
    CategoryGroupedVLMVerifier,
    EvidenceGroundedVLMVerifier,
)


class FakeMultimodalModel:
    def __init__(self):
        self.calls = []

    def generate_json_with_images(
        self, *, system_prompt, user_prompt, image_data_urls, schema
    ):
        self.calls.append((user_prompt, image_data_urls, schema))
        return {
            "violation_score": 0.8,
            "reason": "The ball reverses before the floor-contact frame.",
            "repair_instruction": "Continue falling until contact.",
        }


class FakeFrameLoader:
    def load(self, video_path, frame_indices):
        assert video_path == "video.mp4"
        assert frame_indices == (4, 5, 6)
        return ("data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB")


def _candidate():
    return ViolationCandidate(
        object="red_ball",
        track_id="ball-1",
        category="premature_rebound",
        start_frame=4,
        peak_frame=5,
        end_frame=6,
        reason="rule reason",
        repair_instruction="rule repair",
        detector_score=0.9,
        rules=("velocity_reversal_before_contact",),
    )


def test_vlm_verifier_sends_only_selected_evidence_frames():
    model = FakeMultimodalModel()
    verifier = EvidenceGroundedVLMVerifier(
        model,
        frame_loader=FakeFrameLoader(),
        model_name="openai-test-model",
    )

    review = verifier.verify(
        CriticRequest(video_path="video.mp4", prompt="A red ball falls."),
        _candidate(),
        (4, 5, 6),
    )

    assert review is not None
    assert review.score == 0.8
    assert review.model == "openai-test-model"
    assert len(model.calls[0][1]) == 2


def test_vlm_verifier_skips_api_when_no_keyframes_exist():
    model = FakeMultimodalModel()
    verifier = EvidenceGroundedVLMVerifier(
        model,
        frame_loader=FakeFrameLoader(),
        model_name="openai-test-model",
    )

    review = verifier.verify(
        CriticRequest(video_path="video.mp4"),
        _candidate(),
        (),
    )

    assert review is None
    assert model.calls == []


def test_grouped_verifier_uses_one_call_per_category():
    class FlexibleLoader:
        def load(self, video_path, frame_indices):
            return tuple(
                f"data:image/jpeg;base64,{frame}" for frame in frame_indices
            )

    model = FakeMultimodalModel()
    verifier = CategoryGroupedVLMVerifier(
        model,
        frame_loader=FlexibleLoader(),
        model_name="grouped-test",
    )
    rebound_a = _candidate()
    rebound_b = ViolationCandidate(
        object="blue_ball",
        track_id="ball-2",
        category="premature_rebound",
        start_frame=7,
        peak_frame=8,
        end_frame=9,
        reason="second rebound",
        repair_instruction="repair",
        detector_score=0.7,
        rules=("velocity_reversal_before_contact",),
    )
    disappearance = ViolationCandidate(
        object="ball",
        track_id="ball-3",
        category="object_disappearance",
        start_frame=10,
        peak_frame=11,
        end_frame=12,
        reason="missing",
        repair_instruction="repair",
        detector_score=0.8,
        rules=("object_persistence",),
    )

    reviews = verifier.verify_many(
        CriticRequest(video_path="video.mp4", prompt="A ball falls."),
        (rebound_a, rebound_b, disappearance),
        {0: (4, 5, 6), 1: (7, 8, 9), 2: (10, 11, 12)},
    )

    assert set(reviews) == {0, 1, 2}
    assert len(model.calls) == 2
    assert reviews[0] == reviews[1]
    assert reviews[2].model == "grouped-test"
