"""Non-secret, schema-stable diagnostics for complete PAVG predictions."""

from __future__ import annotations

import json
from dataclasses import replace

from pavg_critic.benchmarking.model_cache import ModelCallEvent
from pavg_critic.benchmarking.pavg_diagnostics import (
    build_pavg_diagnostics,
    build_pavg_failure_diagnostics,
)
from pavg_critic.config import QuestionGraphConfig, TrajectoryConfig
from pavg_critic.evaluation import build_ablation_config
from pavg_critic.pipeline import PhysicsCritic
from pavg_critic.schemas import CriticRequest, FrameState, PhysicsPlan


def _state(frame, *, visible):
    return FrameState(
        frame=frame,
        timestamp_sec=frame / 10,
        object="ball",
        track_id="ball-1",
        center=(50.0, 50.0),
        bbox=(45.0, 45.0, 55.0, 55.0),
        visible=visible,
    )


def test_diagnostics_cover_every_evidence_family_without_raw_plan(sample_factory):
    sample = sample_factory(index=1, physical=False, generator="g")
    config = build_ablation_config("M3_MECHANICS")
    config = replace(
        config,
        trajectory=TrajectoryConfig(smoothing_window=1),
        question_graph=QuestionGraphConfig(enabled=False),
    )
    states = (
        _state(0, visible=True),
        _state(1, visible=False),
        _state(2, visible=False),
        _state(3, visible=False),
    )
    artifacts = PhysicsCritic(config).analyze_detailed(
        CriticRequest(
            video_path=sample.video_path,
            prompt=sample.prompt,
            physics_plan=PhysicsPlan(objects=("ball",)),
        ),
        observations=states,
        floor_y=100,
    )
    event = ModelCallEvent(
        namespace="planner",
        model_id="qwen",
        cache_key="a" * 64,
        prompt_sha256="b" * 64,
        schema_sha256="c" * 64,
        input_evidence_sha256=None,
        cache_hit=False,
        latency_sec=0.2,
    )

    diagnostics = build_pavg_diagnostics(
        sample=sample,
        method_id="M3_MECHANICS",
        artifacts=artifacts,
        config=config,
        stage_events={"planner": (event,)},
        analysis_latency_sec=0.5,
        total_latency_sec=0.6,
        visible_frame_count=4,
    )

    assert diagnostics["key"] == {
        "sample_id": sample.sample_id,
        "method_id": "M3_MECHANICS",
    }
    assert set(diagnostics["evidence_families"]) == {
        "rules",
        "pqsg",
        "checklist",
        "mechanics",
        "vlm",
    }
    assert diagnostics["hard_violation_override"] is True
    assert diagnostics["rules"]["candidate_count"] >= 1
    assert diagnostics["model_calls"]["planner"]["call_count"] == 1
    serialized = json.dumps(diagnostics, allow_nan=False)
    assert "resolved_plan" not in serialized
    assert "image_data" not in serialized


def test_failure_diagnostics_are_keyed_and_do_not_store_error_message(sample_factory):
    sample = sample_factory(index=1, physical=False, generator="g")

    diagnostics = build_pavg_failure_diagnostics(
        sample=sample,
        method_id="M5_FULL",
        stage_events={},
        total_latency_sec=0.3,
        visible_frame_count=0,
        error=RuntimeError("Authorization: Bearer secret"),
    )

    assert diagnostics["key"] == {
        "sample_id": sample.sample_id,
        "method_id": "M5_FULL",
    }
    assert diagnostics["failure"] == {"error_type": "RuntimeError"}
    assert "Bearer secret" not in json.dumps(diagnostics)
