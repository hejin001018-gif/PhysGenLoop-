"""关键帧约束的证据化 VLM 复核。"""

from __future__ import annotations

import json

from pavg_critic.schemas import CriticRequest, FrameState, TrackSequence, ViolationCandidate
from pavg_critic.vlm_verifier import (
    CategoryGroupedVLMVerifier,
    EvidenceGroundedVLMVerifier,
    with_track_evidence,
)


class FakeMultimodalModel:
    def __init__(self):
        self.calls = []
        self.system_prompts = []

    def generate_json_with_images(
        self, *, system_prompt, user_prompt, image_data_urls, schema
    ):
        self.system_prompts.append(system_prompt)
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


def test_grouped_verifier_separates_object_category_and_time_segments():
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
    assert len(model.calls) == 3
    grouped_objects = {
        json.loads(call[0])["candidates"][0]["object"]
        for call in model.calls
    }
    assert grouped_objects == {"red_ball", "blue_ball", "ball"}
    assert reviews[2].model == "grouped-test"


def test_grouped_verifier_bounds_context_images_and_preserves_temporal_endpoints():
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

    verifier.verify_many(
        CriticRequest(video_path="video.mp4", prompt="A ball moves."),
        (_candidate(),),
        {0: tuple(range(20))},
    )

    images = model.calls[0][1]
    assert len(images) == 8
    assert images[0].endswith(",0")
    assert images[-1].endswith(",19")


def test_verifier_payload_contains_sam2_track_evidence_and_expected_event_policy():
    model = FakeMultimodalModel()
    verifier = EvidenceGroundedVLMVerifier(
        model,
        frame_loader=FakeFrameLoader(),
        model_name="evidence-test",
    )
    candidate = ViolationCandidate(
        **{
            **_candidate().__dict__,
            "evidence": {
                "sam2_track": {
                    "track_id": "ball-1",
                    "states": [
                        {"frame": 4, "visible": True, "center": [10, 20]},
                        {"frame": 5, "visible": False, "center": [11, 21]},
                    ],
                }
            },
        }
    )
    verifier.verify(
        CriticRequest(video_path="video.mp4", prompt="A red ball falls."),
        candidate,
        (4, 5, 6),
    )
    payload = json.loads(model.calls[0][0])
    assert payload["candidate"]["evidence"]["sam2_track"]["track_id"] == "ball-1"
    assert payload["expected_event_policy"] == "do_not_reject_prompt_expected_events"
    assert "tracking or segmentation loss" in model.system_prompts[0]
    assert "prompt-relevant physical actor" in model.system_prompts[0]


def test_verifier_retains_optional_claim_status():
    class StatusModel(FakeMultimodalModel):
        def generate_json_with_images(self, **kwargs):
            super().generate_json_with_images(**kwargs)
            return {
                "violation_score": 0.2,
                "reason": "The candidate is not supported.",
                "repair_instruction": "Keep the expected event.",
                "claim_status": "rejected",
            }

    model = StatusModel()
    verifier = EvidenceGroundedVLMVerifier(
        model,
        frame_loader=FakeFrameLoader(),
        model_name="status-test",
    )
    review = verifier.verify(
        CriticRequest(video_path="video.mp4"),
        _candidate(),
        (4, 5, 6),
    )
    assert review is not None
    assert review.claim_status == "rejected"


def test_track_evidence_serializes_bounded_sam2_states():
    states = tuple(
        FrameState(
            frame=frame,
            timestamp_sec=frame / 10,
            object="red_ball",
            center=(frame, frame + 1),
            bbox=(frame, frame + 1, frame + 2, frame + 3),
            track_id="ball-1",
            visible=frame % 2 == 0,
        )
        for frame in range(40)
    )
    candidate = with_track_evidence(
        _candidate(),
        (TrackSequence(track_id="ball-1", object="red_ball", states=states),),
    )
    track = candidate.evidence["sam2_track"]
    assert track["track_id"] == "ball-1"
    assert track["state_count"] == 40
    assert track["visible_count"] == 20
    assert len(track["states"]) <= 24
