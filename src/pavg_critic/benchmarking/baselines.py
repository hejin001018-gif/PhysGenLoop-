"""Matched direct and checklist-prompt VLM benchmark baselines."""

from __future__ import annotations

from time import perf_counter

from pavg_critic.interfaces import MultimodalStructuredModel

from .contracts import BenchmarkPrediction, BenchmarkSample
from .frames import sample_video_frames


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "semantic_score",
        "physics_score",
        "confidence",
        "violation_categories",
        "reason",
    ],
    "properties": {
        "semantic_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "physics_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "violation_categories": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reason": {"type": "string"},
    },
}

DIRECT_SYSTEM = """You are evaluating a generated video. Score semantic adherence and physical commonsense from 1 to 5. Use only visible evidence. If evidence is insufficient, lower confidence. Return only the requested JSON object."""

STRUCTURED_SYSTEM = """You are evaluating a generated video with a fixed physical checklist. Check object permanence, gravity and support, contact and collision order, conservation of mass/momentum when visible, material behavior, and temporal continuity. Then score semantic adherence and physical commonsense from 1 to 5. Do not assume events that are not visible. Return only the requested JSON object."""


class DirectVLMJudge:
    def __init__(
        self,
        model: MultimodalStructuredModel,
        *,
        model_id: str,
        structured: bool,
        frame_count: int = 16,
    ):
        self.model = model
        self.model_id = model_id
        self.structured = structured
        self.frame_count = frame_count
        self.method_id = "D1_STRUCTURED_VLM" if structured else "D0_DIRECT_VLM"

    def evaluate(self, sample: BenchmarkSample) -> BenchmarkPrediction:
        started = perf_counter()
        visible_frame_count = 0
        try:
            frames = sample_video_frames(sample.video_path, count=self.frame_count)
            visible_frame_count = len(frames.data_urls)
            result = self.model.generate_json_with_images(
                system_prompt=STRUCTURED_SYSTEM if self.structured else DIRECT_SYSTEM,
                user_prompt=(
                    f"Prompt: {sample.prompt}\n"
                    f"Frame indices: {list(frames.indices)}"
                ),
                image_data_urls=frames.data_urls,
                schema=JUDGE_SCHEMA,
            )
            semantic_score = float(result["semantic_score"])
            physics_score = float(result["physics_score"])
            return BenchmarkPrediction(
                sample_id=sample.sample_id,
                method_id=self.method_id,
                model_id=self.model_id,
                semantic_score=semantic_score,
                physics_score=physics_score,
                semantic_label=(
                    "adherent" if semantic_score >= 4 else "not_adherent"
                ),
                physics_label="physical" if physics_score >= 4 else "violation",
                confidence=float(result["confidence"]),
                coverage=1.0,
                latency_sec=perf_counter() - started,
                visible_frame_count=visible_frame_count,
                violation_categories=tuple(
                    str(item) for item in result["violation_categories"]
                ),
                evidence_frames=frames.indices,
            )
        except Exception as exc:
            return BenchmarkPrediction(
                sample_id=sample.sample_id,
                method_id=self.method_id,
                model_id=self.model_id,
                semantic_score=None,
                physics_score=None,
                semantic_label="unknown",
                physics_label="unknown",
                confidence=0.0,
                coverage=0.0,
                latency_sec=perf_counter() - started,
                visible_frame_count=visible_frame_count,
                failure={
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                },
            )
