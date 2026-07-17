"""Strict complete-PAVG metrics, attribution and prompt diagnostics."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pavg_critic.benchmarking.full_pavg_report import (
    FULL_METHODS,
    PAVG_DIAGNOSTIC_METHODS,
    build_full_pavg_report,
)


def _predictions(samples, prediction_factory):
    labels = {
        "D0_DIRECT_VLM": ("physical", "violation", "physical", "violation"),
        "B1_RULE": ("physical", "violation", "physical", "violation"),
        "D1_STRUCTURED_VLM": ("physical", "violation", "physical", "violation"),
        "M1_GRAPH": ("physical", "violation", "physical", "violation"),
        "M2_CHECKLIST": ("violation", "physical", "physical", "violation"),
        "M3_MECHANICS": ("violation", "physical", "physical", "violation"),
        "M4_VLM": ("physical", "physical", "physical", "violation"),
        "M5_FULL": ("physical", "physical", "violation", "violation"),
    }
    return tuple(
        prediction_factory(
            sample.sample_id,
            label,
            5.0 if label == "physical" else 2.0,
            method_id=method,
        )
        for method in FULL_METHODS
        for sample, label in zip(samples, labels[method])
    )


def _diagnostics(samples):
    result = []
    for method in PAVG_DIAGNOSTIC_METHODS:
        for index, sample in enumerate(samples):
            result.append(
                {
                    "schema_version": "1.0",
                    "key": {"sample_id": sample.sample_id, "method_id": method},
                    "planner": {"source": "model" if method == "M5_FULL" else "template", "fallback_used": False},
                    "question_graph": {"source": "graph", "node_count": 2, "physics_coverage": 1.0},
                    "video_science": {"enabled": True, "coverage": 1.0},
                    "mechanics": {"enabled": True, "coverage": 1.0},
                    "rules": {"candidate_count": 1, "retained_violation_count": 1},
                    "vlm_reviews": {"status_counts": {"confirmed": 1, "rejected": 0, "uncertain": 0, "unavailable": 0}},
                    "evidence_families": {
                        family: {"status": "available", "coverage": 1.0}
                        for family in ("rules", "pqsg", "checklist", "mechanics", "vlm")
                    },
                    "fusion": {"pre_evidence_fusion": {"decision": "physical"}, "final": {"decision": "violation"}},
                    "hard_violation_override": method == "M5_FULL" and index == 0,
                    "model_calls": {
                        "planner": {"call_count": 1, "provider_call_count": 1, "cache_hit_count": 0, "error_count": 0, "latency_sec": 0.1, "events": []}
                    },
                    "latency": {"analysis_sec": 0.2, "total_sec": 0.3, "visible_frame_count": 4},
                    "provider_failures": [],
                    "failure": None,
                }
            )
    return tuple(result)


def _prompt_predictions(samples, full_predictions, prediction_factory):
    correct = tuple(
        item for item in full_predictions if item.method_id == "M5_FULL"
    )
    shuffled = tuple(
        prediction_factory(
            sample.sample_id,
            "physical",
            5.0,
            method_id="M5_SHUFFLED_PROMPT_300",
        )
        for sample in samples
    )
    oracle = tuple(
        replace(item, method_id="M5_ORACLE_PLAN_300") for item in correct
    )
    return correct + shuffled + oracle


def test_complete_report_computes_primary_sequential_and_prompt_attribution(
    sample_factory, prediction_factory
):
    gold = (True, True, False, False)
    samples = tuple(
        sample_factory(index=index, physical=physical, generator="g")
        for index, physical in enumerate(gold)
    )
    predictions = _predictions(samples, prediction_factory)

    report = build_full_pavg_report(
        samples=samples,
        predictions=predictions,
        diagnostics=_diagnostics(samples),
        pilot_samples=samples,
        prompt_predictions=_prompt_predictions(samples, predictions, prediction_factory),
        bootstrap_resamples=20,
        bootstrap_seed=20260717,
    )

    assert report["primary"]["candidate"] == "M5_FULL"
    assert report["sequential_attribution"]["M2_CHECKLIST-M1_GRAPH"]["changed"] == 2
    assert report["module_availability"]["video_science"]["available"] == 4
    assert report["hard_override"]["forced_violation"] == 1
    assert report["material_decision"]["gates"]["macro_f1_delta"]["threshold"] == 0.05
    assert report["prompt_diagnostics"]["scope"] == "diagnostic_only"


def test_complete_report_rejects_missing_diagnostic_key(
    sample_factory, prediction_factory
):
    samples = tuple(
        sample_factory(index=index, physical=index < 2, generator="g")
        for index in range(4)
    )
    predictions = _predictions(samples, prediction_factory)
    diagnostics = _diagnostics(samples)[:-1]

    with pytest.raises(ValueError, match="diagnostic keys must match exactly"):
        build_full_pavg_report(
            samples=samples,
            predictions=predictions,
            diagnostics=diagnostics,
            pilot_samples=samples,
            prompt_predictions=_prompt_predictions(samples, predictions, prediction_factory),
            bootstrap_resamples=20,
            bootstrap_seed=20260717,
        )
