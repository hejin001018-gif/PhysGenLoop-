from pavg_critic.benchmarking.baselines import DirectVLMJudge


class ScriptedModel:
    def __init__(self):
        self.calls = []

    def generate_json_with_images(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "semantic_score": 5,
            "physics_score": 2,
            "confidence": 0.8,
            "violation_categories": ["gravity"],
            "reason": "The ball accelerates upward without a force.",
        }


def test_direct_and_structured_judges_use_same_images(sample_factory, monkeypatch):
    model = ScriptedModel()
    frames = type(
        "Frames",
        (),
        {
            "data_urls": ("data:image/jpeg;base64,AA==",) * 4,
            "indices": (0, 1, 2, 3),
        },
    )()
    monkeypatch.setattr(
        "pavg_critic.benchmarking.baselines.sample_video_frames",
        lambda *args, **kwargs: frames,
    )
    sample = sample_factory(index=1, physical=False, generator="g")
    d0 = DirectVLMJudge(model, model_id="fake", structured=False).evaluate(sample)
    d1 = DirectVLMJudge(model, model_id="fake", structured=True).evaluate(sample)
    assert d0.physics_label == d1.physics_label == "violation"
    assert model.calls[0]["image_data_urls"] == model.calls[1]["image_data_urls"]
    assert "checklist" not in model.calls[0]["system_prompt"].lower()
    assert "checklist" in model.calls[1]["system_prompt"].lower()


class TimeoutModel:
    def generate_json_with_images(self, **kwargs):
        raise TimeoutError("provider timeout")


def test_timeout_becomes_explicit_unknown(sample_factory, monkeypatch):
    frames = type(
        "Frames",
        (),
        {"data_urls": ("data:image/jpeg;base64,AA==",), "indices": (0,)},
    )()
    monkeypatch.setattr(
        "pavg_critic.benchmarking.baselines.sample_video_frames",
        lambda *args, **kwargs: frames,
    )
    sample = sample_factory(index=2, physical=False, generator="g")
    prediction = DirectVLMJudge(
        TimeoutModel(), model_id="timeout", structured=False
    ).evaluate(sample)
    assert prediction.physics_label == "unknown"
    assert prediction.failure["type"] == "TimeoutError"
    assert prediction.coverage == 0.0


def test_decode_failure_becomes_explicit_unknown(sample_factory, monkeypatch):
    def fail_decode(*args, **kwargs):
        raise ValueError("cannot open video")

    monkeypatch.setattr(
        "pavg_critic.benchmarking.baselines.sample_video_frames",
        fail_decode,
    )
    sample = sample_factory(index=3, physical=False, generator="g")
    prediction = DirectVLMJudge(
        ScriptedModel(), model_id="fake", structured=False
    ).evaluate(sample)
    assert prediction.physics_label == "unknown"
    assert prediction.visible_frame_count == 0
    assert prediction.failure["type"] == "ValueError"
