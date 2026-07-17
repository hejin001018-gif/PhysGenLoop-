"""Crash-recoverable paired prediction and diagnostics JSONL output."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pavg_critic.benchmarking.audited_runner import AuditedBenchmarkRunner


def _keys(path, *, diagnostics=False):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if diagnostics:
        return {(item["key"]["sample_id"], item["key"]["method_id"]) for item in records}
    return {(item["sample_id"], item["method_id"]) for item in records}


class AuditedMethod:
    method_id = "M5_FULL"

    def __init__(self, prediction_factory, *, fail=False):
        self.prediction_factory = prediction_factory
        self.fail = fail
        self.calls = []

    def evaluate_audited(self, sample):
        self.calls.append(sample.sample_id)
        prediction = self.prediction_factory(
            sample.sample_id,
            "physical" if not self.fail else "unknown",
            5.0 if not self.fail else None,
            method_id=self.method_id,
        )
        if self.fail:
            prediction = replace(
                prediction,
                failure={"type": "TimeoutError"},
            )
        return prediction, {
            "schema_version": "1.0",
            "key": {"sample_id": sample.sample_id, "method_id": self.method_id},
            "failure": prediction.failure,
        }


def test_audited_runner_appends_one_pair_and_resumes(
    tmp_path, sample_factory, prediction_factory
):
    predictions = tmp_path / "predictions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    sample = sample_factory(index=1, physical=True, generator="g")
    method = AuditedMethod(prediction_factory)
    runner = AuditedBenchmarkRunner(predictions, diagnostics)

    runner.run((sample,), (method,))
    runner.run((sample,), (method,))

    expected = {(sample.sample_id, "M5_FULL")}
    assert _keys(predictions) == expected
    assert _keys(diagnostics, diagnostics=True) == expected
    assert method.calls == [sample.sample_id]


def test_audited_runner_recovers_crash_after_prediction_append(
    tmp_path, sample_factory, prediction_factory, monkeypatch
):
    predictions = tmp_path / "predictions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    sample = sample_factory(index=1, physical=True, generator="g")
    method = AuditedMethod(prediction_factory)
    runner = AuditedBenchmarkRunner(predictions, diagnostics)
    append = runner._append_fsync

    def crash_on_diagnostics(path, payload):
        if path == diagnostics:
            raise OSError("injected crash")
        append(path, payload)

    monkeypatch.setattr(runner, "_append_fsync", crash_on_diagnostics)
    with pytest.raises(OSError, match="injected crash"):
        runner.run((sample,), (method,))
    assert runner.pending_path.is_file()
    assert _keys(predictions) == {(sample.sample_id, "M5_FULL")}
    assert not diagnostics.exists()

    AuditedBenchmarkRunner(predictions, diagnostics).run((sample,), (method,))

    assert not runner.pending_path.exists()
    assert _keys(predictions) == _keys(diagnostics, diagnostics=True)
    assert method.calls == [sample.sample_id]


def test_audited_runner_recovers_crash_before_pending_clear(
    tmp_path, sample_factory, prediction_factory, monkeypatch
):
    predictions = tmp_path / "predictions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    sample = sample_factory(index=1, physical=True, generator="g")
    method = AuditedMethod(prediction_factory)
    runner = AuditedBenchmarkRunner(predictions, diagnostics)

    monkeypatch.setattr(
        runner,
        "_clear_pending",
        lambda: (_ for _ in ()).throw(OSError("injected clear crash")),
    )
    with pytest.raises(OSError, match="injected clear crash"):
        runner.run((sample,), (method,))
    assert runner.pending_path.is_file()
    assert _keys(predictions) == _keys(diagnostics, diagnostics=True)

    AuditedBenchmarkRunner(predictions, diagnostics).run((sample,), (method,))

    assert not runner.pending_path.exists()
    assert method.calls == [sample.sample_id]


def test_audited_runner_rejects_unrecoverable_asymmetric_files(
    tmp_path, sample_factory, prediction_factory
):
    predictions = tmp_path / "predictions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    sample = sample_factory(index=1, physical=True, generator="g")
    prediction = prediction_factory(
        sample.sample_id, "physical", 5.0, method_id="M5_FULL"
    )
    predictions.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="asymmetric"):
        AuditedBenchmarkRunner(predictions, diagnostics).run(
            (sample,), (AuditedMethod(prediction_factory),)
        )


def test_audited_runner_stops_after_paired_failure(
    tmp_path, sample_factory, prediction_factory
):
    predictions = tmp_path / "predictions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    samples = tuple(
        sample_factory(index=index, physical=True, generator="g")
        for index in range(2)
    )
    method = AuditedMethod(prediction_factory, fail=True)

    with pytest.raises(RuntimeError, match="failure budget"):
        AuditedBenchmarkRunner(
            predictions, diagnostics, max_new_failures=1
        ).run(samples, (method,))

    assert method.calls == ["0"]
    assert _keys(predictions) == _keys(diagnostics, diagnostics=True) == {
        ("0", "M5_FULL")
    }
