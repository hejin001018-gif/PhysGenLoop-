import json

import pytest

from pavg_critic.benchmarking.pavg_methods import (
    CachedObservationProvider,
    PAVGMethod,
)


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


def test_unsupported_pavg_mode_is_rejected(tmp_path):
    provider = CachedObservationProvider(tmp_path, lambda ignored: ())
    with pytest.raises(ValueError, match="not supported"):
        PAVGMethod("M5_FULL", provider, model_id=None)
