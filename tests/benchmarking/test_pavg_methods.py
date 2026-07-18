import json
from dataclasses import replace

import pytest

from pavg_critic.benchmarking.pavg_methods import (
    CachedObservationProvider,
    PAVGMethod,
)
from pavg_critic.benchmarking.model_cache import AuditedCachedModel


def test_two_pavg_methods_share_one_observation_production(
    tmp_path, sample_factory, frame_state_factory
):
    calls = []

    def producer(sample):
        calls.append(sample.sample_id)
        return (frame_state_factory(),)

    sample = sample_factory(index=1, physical=False, generator="g")
    provider = CachedObservationProvider(tmp_path / "observations", producer)
    PAVGMethod("B1_RULE", provider, model_id="fake-frontend").evaluate(sample)
    PAVGMethod("M3_MECHANICS", provider, model_id="fake-frontend").evaluate(sample)
    assert calls == [sample.sample_id]


def test_cache_round_trip_avoids_producer_across_provider_instances(
    tmp_path, sample_factory, frame_state_factory
):
    sample = sample_factory(index=1, physical=False, generator="g")
    cache_dir = tmp_path / "observations"
    first = CachedObservationProvider(
        cache_dir,
        lambda ignored: (frame_state_factory(),),
    )
    assert first.get(sample) == (frame_state_factory(),)
    metadata = json.loads((cache_dir / "1.meta.json").read_text(encoding="utf-8"))
    assert metadata["observed_frame_count"] == 1
    assert metadata["track_count"] == 1
    assert metadata["propagation_failure"] is None
    assert metadata["production_latency_sec"] >= 0.0

    def should_not_run(ignored):
        raise AssertionError("cache was not reused")

    second = CachedObservationProvider(cache_dir, should_not_run)
    assert second.get(sample) == (frame_state_factory(),)


def test_empty_observation_output_is_rejected(
    tmp_path, sample_factory
):
    sample = sample_factory(index=1, physical=False, generator="g")
    provider = CachedObservationProvider(tmp_path / "observations", lambda ignored: ())
    with pytest.raises(ValueError, match="produced no states"):
        provider.get(sample)
    assert not (tmp_path / "observations" / "1.json").exists()
    metadata = json.loads(
        (tmp_path / "observations" / "1.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["propagation_failure"]["type"] == "ValueError"


def test_audited_prediction_failure_does_not_store_exception_message(
    tmp_path, sample_factory
):
    sample = sample_factory(index=1, physical=False, generator="g")

    def fail_with_secret(ignored):
        raise RuntimeError("Authorization: Bearer secret-value")

    method = PAVGMethod(
        "B1_RULE",
        CachedObservationProvider(tmp_path / "observations", fail_with_secret),
        model_id=None,
    )

    prediction, diagnostics = method.evaluate_audited(sample)

    assert prediction.failure == {"type": "RuntimeError"}
    assert diagnostics["failure"] == {"error_type": "RuntimeError"}
    assert "secret-value" not in json.dumps(
        {"prediction": prediction.to_dict(), "diagnostics": diagnostics}
    )


def test_m5_requires_all_three_explicit_models(tmp_path):
    provider = CachedObservationProvider(tmp_path, lambda ignored: ())
    with pytest.raises(ValueError, match="planner, question, and verifier"):
        PAVGMethod("M5_FULL", provider, model_id=None)


class FakeStructuredModel:
    model = "fake-qwen"

    def generate_json(self, *, system_prompt, user_prompt, schema):
        if "nodes" in schema.get("required", ()):
            return {"nodes": []}
        return {
            "objects": ["ball"],
            "expected_events": ["fall"],
            "relations": [],
            "physics_constraints": [],
        }

    def generate_json_with_images(
        self, *, system_prompt, user_prompt, image_data_urls, schema
    ):
        return {
            "violation_score": 0.1,
            "reason": "The candidate is not supported.",
            "repair_instruction": "Keep the expected motion.",
            "claim_status": "rejected",
        }


def test_m5_injects_planner_pqsg_and_verifier_stages(
    tmp_path, sample_factory, frame_state_factory
):
    sample = sample_factory(index=1, physical=True, generator="g")
    provider = CachedObservationProvider(
        tmp_path / "observations", lambda ignored: (frame_state_factory(),)
    )
    raw_model = FakeStructuredModel()
    stages = {
        name: AuditedCachedModel(
            raw_model,
            cache_dir=tmp_path / "model-cache",
            namespace=name,
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            model_revision="a" * 64,
        )
        for name in ("planner", "pqsg", "verifier")
    }
    method = PAVGMethod(
        "M5_FULL",
        provider,
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        planner_model=stages["planner"],
        question_model=stages["pqsg"],
        verifier_model=stages["verifier"],
        model_stages=stages,
    )

    prediction, diagnostics = method.evaluate_audited(sample)

    assert prediction.method_id == "M5_FULL"
    assert prediction.failure is None
    assert diagnostics["model_calls"]["planner"]["call_count"] == 1
    assert diagnostics["model_calls"]["pqsg"]["call_count"] == 1
    assert diagnostics["planner"]["source"] == "model"
    assert diagnostics["planner"]["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert diagnostics["question_graph"]["source"] == (
        "pavg_hybrid_template_pqsg"
    )


def test_m4_accepts_explicit_verifier_model(tmp_path):
    provider = CachedObservationProvider(tmp_path, lambda ignored: ())
    method = PAVGMethod(
        "M4_VLM",
        provider,
        model_id="verifier-model",
        verifier_model=object(),
        verifier_detector_weight=0.4,
    )
    assert method.method_id == "M4_VLM"


def test_m4_defaults_to_detector_dominant_fusion(tmp_path):
    provider = CachedObservationProvider(tmp_path, lambda ignored: ())
    method = PAVGMethod(
        "M4_VLM",
        provider,
        model_id="verifier-model",
        verifier_model=object(),
    )
    assert method.verifier_detector_weight == 0.7


def test_m4_verifier_failure_becomes_explicit_unknown(
    tmp_path, sample_factory, frame_state_factory
):
    first = frame_state_factory()
    states = (
        first,
        replace(first, frame=1, timestamp_sec=0.1, visible=False),
        replace(first, frame=2, timestamp_sec=0.2, visible=False),
        replace(first, frame=3, timestamp_sec=0.3, visible=False),
    )
    sample = sample_factory(index=1, physical=False, generator="g")
    provider = CachedObservationProvider(tmp_path, lambda ignored: states)
    prediction = PAVGMethod(
        "M4_VLM",
        provider,
        model_id="verifier-model",
        verifier_model=object(),
    ).evaluate(sample)
    assert prediction.physics_label == "unknown"
    assert prediction.failure is not None
