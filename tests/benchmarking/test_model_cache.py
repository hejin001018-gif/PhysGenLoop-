"""Deterministic, stage-separated benchmark model response caching."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from pavg_critic.benchmarking.model_cache import AuditedCachedModel


SCHEMA = {
    "type": "object",
    "required": ["value"],
    "properties": {"value": {"type": "integer"}},
}


class FakeModel:
    def __init__(self):
        self.text_calls = 0
        self.image_calls = 0

    def generate_json(self, *, system_prompt, user_prompt, schema):
        self.text_calls += 1
        return {"value": self.text_calls}

    def generate_json_with_images(
        self, *, system_prompt, user_prompt, image_data_urls, schema
    ):
        self.image_calls += 1
        return {"value": self.image_calls}


def _cached(fake, cache_dir, namespace="planner", **kwargs):
    model = AuditedCachedModel(
        fake,
        cache_dir=cache_dir,
        namespace=namespace,
        model_id="qwen",
        model_revision="snapshot-a",
        **kwargs,
    )
    model.bind_sample("sample-a")
    return model


def test_cache_identity_binds_sample_and_model_revision(tmp_path):
    fake = FakeModel()
    model = AuditedCachedModel(
        fake,
        cache_dir=tmp_path,
        namespace="planner",
        model_id="qwen",
        model_revision="snapshot-a",
    )
    model.bind_sample("sample-a")
    first = model.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
    model.bind_sample("sample-b")
    second = model.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
    model.bind_sample("sample-a")
    reused = model.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
    other_revision = AuditedCachedModel(
        fake,
        cache_dir=tmp_path,
        namespace="planner",
        model_id="qwen",
        model_revision="snapshot-b",
    )
    other_revision.bind_sample("sample-a")
    revised = other_revision.generate_json(
        system_prompt="s", user_prompt="u", schema=SCHEMA
    )

    assert (first, second, reused, revised) == (
        {"value": 1},
        {"value": 2},
        {"value": 1},
        {"value": 3},
    )
    assert fake.text_calls == 3
    assert model.events_since(0)[0].sample_id == "sample-a"
    assert model.events_since(0)[0].model_revision == "snapshot-a"


def test_identical_text_call_is_reused_without_recording_prompt(tmp_path):
    fake = FakeModel()
    model = _cached(fake, tmp_path)

    first = model.generate_json(
        system_prompt="system secret-free",
        user_prompt="sensitive user prompt",
        schema=SCHEMA,
    )
    second = model.generate_json(
        system_prompt="system secret-free",
        user_prompt="sensitive user prompt",
        schema=SCHEMA,
    )

    assert first == second == {"value": 1}
    assert fake.text_calls == 1
    events = model.events_since(0)
    assert [event.cache_hit for event in events] == [False, True]
    assert all(len(event.cache_key) == 64 for event in events)
    assert "sensitive user prompt" not in json.dumps(
        [event.to_dict() for event in events]
    )


def test_namespace_prompt_and_schema_are_part_of_cache_identity(tmp_path):
    fake = FakeModel()
    planner = _cached(fake, tmp_path)
    pqsg = _cached(fake, tmp_path, "pqsg")

    planner.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
    pqsg.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
    planner.generate_json(system_prompt="s", user_prompt="changed", schema=SCHEMA)
    planner.generate_json(
        system_prompt="s",
        user_prompt="u",
        schema={**SCHEMA, "additionalProperties": False},
    )

    assert fake.text_calls == 4


def test_image_order_is_hashed_and_image_payload_is_not_in_telemetry(tmp_path):
    fake = FakeModel()
    model = _cached(fake, tmp_path, "verifier")
    images = ("data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB")

    first = model.generate_json_with_images(
        system_prompt="s", user_prompt="u", image_data_urls=images, schema=SCHEMA
    )
    second = model.generate_json_with_images(
        system_prompt="s", user_prompt="u", image_data_urls=images, schema=SCHEMA
    )
    reversed_result = model.generate_json_with_images(
        system_prompt="s",
        user_prompt="u",
        image_data_urls=tuple(reversed(images)),
        schema=SCHEMA,
    )

    assert first == second == {"value": 1}
    assert reversed_result == {"value": 2}
    assert fake.image_calls == 2
    telemetry = json.dumps([event.to_dict() for event in model.events_since(0)])
    assert "data:image" not in telemetry
    assert all(event.input_evidence_sha256 for event in model.events_since(0))


def test_corrupted_cache_metadata_is_rejected(tmp_path):
    fake = FakeModel()
    model = _cached(fake, tmp_path)
    model.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
    cache_path = next((tmp_path / "planner").rglob("*.json"))
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["model_id"] = "wrong-model"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata mismatch"):
        _cached(fake, tmp_path).generate_json(
            system_prompt="s", user_prompt="u", schema=SCHEMA
        )


def test_provider_failure_is_retried_but_never_cached(tmp_path, monkeypatch):
    class FlakyModel(FakeModel):
        def generate_json(self, *, system_prompt, user_prompt, schema):
            self.text_calls += 1
            if self.text_calls <= 2:
                raise TimeoutError("temporary outage")
            return {"value": self.text_calls}

    monkeypatch.setattr("pavg_critic.benchmarking.model_cache.sleep", lambda _: None)
    fake = FlakyModel()
    model = _cached(fake, tmp_path, retries=3)

    assert model.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA) == {
        "value": 3
    }
    assert fake.text_calls == 3
    assert len(tuple((tmp_path / "planner").rglob("*.json"))) == 1
    assert model.events_since(0)[0].error_type is None


def test_terminal_provider_failure_leaves_no_cache_file(tmp_path):
    class ToggleModel(FakeModel):
        failing = True

        def generate_json(self, *, system_prompt, user_prompt, schema):
            self.text_calls += 1
            if self.failing:
                raise TimeoutError("offline")
            return {"value": self.text_calls}

    fake = ToggleModel()
    model = _cached(fake, tmp_path, retries=1)
    with pytest.raises(TimeoutError, match="offline"):
        model.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
    assert not tuple(tmp_path.rglob("*.json"))
    assert model.events_since(0)[0].error_type == "TimeoutError"

    fake.failing = False
    assert model.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA) == {
        "value": 2
    }


def test_concurrent_identical_calls_use_one_provider_request(tmp_path):
    class SlowModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()

        def generate_json(self, *, system_prompt, user_prompt, schema):
            with self.lock:
                self.text_calls += 1
                value = self.text_calls
            time.sleep(0.05)
            return {"value": value}

    fake = SlowModel()
    models = (_cached(fake, tmp_path), _cached(fake, tmp_path))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda model: model.generate_json(
                    system_prompt="s", user_prompt="u", schema=SCHEMA
                ),
                models,
            )
        )

    assert results == ({"value": 1}, {"value": 1})
    assert fake.text_calls == 1
    assert sorted(
        event.cache_hit for model in models for event in model.events_since(0)
    ) == [False, True]
