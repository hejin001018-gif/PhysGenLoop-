from dataclasses import replace

import pytest

from pavg_critic.benchmarking.metrics import compute_smoke_metrics


def test_metrics_count_unknown_as_missed_violation(sample_factory, prediction_factory):
    samples = (
        sample_factory(index=1, physical=False, generator="g"),
        sample_factory(index=2, physical=True, generator="g"),
        sample_factory(index=3, physical=False, generator="g"),
    )
    predictions = (
        prediction_factory("1", "violation", 0.2),
        prediction_factory("2", "physical", 0.9),
        prediction_factory("3", "unknown", None),
    )
    metrics = compute_smoke_metrics(samples, predictions)
    assert metrics["count"] == 3
    assert metrics["macro_f1"] < 1.0
    assert metrics["violation_recall"] == 0.5
    assert metrics["unknown_rate"] == 1 / 3
    assert metrics["failure_rate"] == 0.0


def test_perfect_predictions_have_perfect_classification_scores(
    sample_factory, prediction_factory
):
    samples = (
        sample_factory(index=1, physical=False, generator="g"),
        sample_factory(index=2, physical=True, generator="g"),
    )
    predictions = (
        prediction_factory("1", "violation", 2.0),
        prediction_factory("2", "physical", 5.0),
    )
    metrics = compute_smoke_metrics(samples, predictions)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["physics_spearman"] == pytest.approx(1.0)


def test_metrics_report_physical_recall(sample_factory, prediction_factory):
    samples = (
        sample_factory(index=1, physical=True, generator="g"),
        sample_factory(index=2, physical=True, generator="g"),
        sample_factory(index=3, physical=False, generator="g"),
    )
    predictions = (
        prediction_factory("1", "physical", 5.0),
        prediction_factory("2", "violation", 2.0),
        prediction_factory("3", "violation", 2.0),
    )

    metrics = compute_smoke_metrics(samples, predictions)

    assert metrics["physical_recall"] == 0.5


def test_metrics_report_linear_latency_percentiles_for_even_length_vector(
    sample_factory, prediction_factory
):
    samples = tuple(
        sample_factory(index=index, physical=index % 2 == 0, generator="g")
        for index in range(1, 5)
    )
    predictions = tuple(
        replace(
            prediction_factory(
                str(index),
                "physical" if index % 2 == 0 else "violation",
                5.0 if index % 2 == 0 else 2.0,
            ),
            latency_sec=float(index),
        )
        for index in range(1, 5)
    )

    metrics = compute_smoke_metrics(samples, predictions)

    assert metrics["p50_latency_sec"] == pytest.approx(2.5)
    assert metrics["p95_latency_sec"] == pytest.approx(3.85)


def test_tied_ordinal_scores_return_no_correlation(sample_factory, prediction_factory):
    samples = (
        sample_factory(index=1, physical=False, generator="g"),
        sample_factory(index=2, physical=True, generator="g"),
    )
    predictions = (
        prediction_factory("1", "violation", 3.0),
        prediction_factory("2", "physical", 3.0),
    )
    assert compute_smoke_metrics(samples, predictions)["physics_spearman"] is None


def test_metrics_reject_duplicate_prediction_ids(sample_factory, prediction_factory):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    predictions = (
        prediction_factory("1", "violation", 2.0),
        prediction_factory("1", "violation", 2.0),
    )
    with pytest.raises(ValueError, match="duplicate"):
        compute_smoke_metrics(samples, predictions)


def test_metrics_reject_empty_inputs():
    with pytest.raises(ValueError, match="at least one"):
        compute_smoke_metrics((), ())
