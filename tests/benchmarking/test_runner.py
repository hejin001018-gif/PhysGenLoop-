import json

import pytest

from pavg_critic.benchmarking.runner import BenchmarkRunner, load_predictions


class CountingMethod:
    method_id = "D0_DIRECT_VLM"

    def __init__(self, prediction_factory):
        self.calls = []
        self.prediction_factory = prediction_factory

    def evaluate(self, sample):
        self.calls.append(sample.sample_id)
        return self.prediction_factory(sample.sample_id, "physical", 4.0)


def test_runner_skips_completed_sample_method_pairs(
    tmp_path, sample_factory, prediction_factory
):
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps(prediction_factory("0", "physical", 4.0).to_dict()) + "\n",
        encoding="utf-8",
    )
    method = CountingMethod(prediction_factory)
    samples = (
        sample_factory(index=0, physical=True, generator="g"),
        sample_factory(index=1, physical=True, generator="g"),
    )
    BenchmarkRunner(output).run(samples, (method,))
    assert method.calls == ["1"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_runner_rejects_corrupt_existing_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        BenchmarkRunner(path).run((), ())


def test_runner_rejects_method_key_mismatch(
    tmp_path, sample_factory, prediction_factory
):
    class WrongMethod:
        method_id = "B1_RULE"

        def evaluate(self, sample):
            return prediction_factory(
                sample.sample_id,
                "physical",
                4.0,
                method_id="D0_DIRECT_VLM",
            )

    sample = sample_factory(index=1, physical=True, generator="g")
    with pytest.raises(ValueError, match="mismatched prediction key"):
        BenchmarkRunner(tmp_path / "predictions.jsonl").run(
            (sample,),
            (WrongMethod(),),
        )


def test_load_predictions_round_trips_jsonl(tmp_path, prediction_factory):
    prediction = prediction_factory("1", "physical", 4.0)
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")
    assert load_predictions(path) == (prediction,)


def test_runner_rejects_duplicate_existing_keys(tmp_path, prediction_factory):
    prediction = prediction_factory("1", "physical", 4.0)
    line = json.dumps(prediction.to_dict()) + "\n"
    path = tmp_path / "predictions.jsonl"
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prediction key"):
        BenchmarkRunner(path).run((), ())


def test_runner_rejects_concurrent_lock(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.with_suffix(".jsonl.lock").write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already running"):
        BenchmarkRunner(path).run((), ())
