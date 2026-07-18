"""Strict complete-PAVG metrics, module attribution and prompt diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample
from .full_report import (
    action_group_bootstrap,
    build_slices,
    evaluate_material_improvement,
    paired_outcomes,
    strict_method_metrics,
)
from .pavg_diagnostics import validate_pavg_diagnostic


FULL_METHODS = (
    "D0_DIRECT_VLM",
    "D1_STRUCTURED_VLM",
    "B1_RULE",
    "M1_GRAPH",
    "M2_CHECKLIST",
    "M3_MECHANICS",
    "M4_VLM",
    "M5_FULL",
)
PAVG_DIAGNOSTIC_METHODS = (
    "M1_GRAPH",
    "M2_CHECKLIST",
    "M3_MECHANICS",
    "M4_VLM",
    "M5_FULL",
)
PROMPT_METHODS = (
    "M5_FULL",
    "M5_SHUFFLED_PROMPT_300",
    "M5_ORACLE_PLAN_300",
)
SEQUENTIAL_TRANSITIONS = (
    ("B1_RULE", "M1_GRAPH"),
    ("M1_GRAPH", "M2_CHECKLIST"),
    ("M2_CHECKLIST", "M3_MECHANICS"),
    ("M3_MECHANICS", "M4_VLM"),
    ("M4_VLM", "M5_FULL"),
)
PHYSICAL_SCORE_THRESHOLD = 0.6


def _expected_hard_override(record: Mapping[str, Any]) -> bool:
    fusion = record["fusion"]
    rules = record["rules"]
    if (
        fusion["final"]["decision"] != "violation"
        or rules["retained_violation_count"] <= 0
    ):
        return False
    weighted_score = 0.0
    effective_total = 0.0
    for family in ("pqsg", "checklist", "mechanics"):
        evidence = record["evidence_families"][family]
        if evidence["status"] != "available" or evidence["score"] is None:
            continue
        weight = float(evidence["effective_weight"])
        weighted_score += float(evidence["score"]) * weight
        effective_total += weight
    return (
        effective_total > 0
        and weighted_score / effective_total >= PHYSICAL_SCORE_THRESHOLD
    )


def _prediction_index(
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[BenchmarkPrediction],
    methods: Sequence[str],
) -> dict[tuple[str, str], BenchmarkPrediction]:
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("samples contain duplicate sample IDs")
    expected = {
        (sample_id, method) for sample_id in sample_ids for method in methods
    }
    result: dict[tuple[str, str], BenchmarkPrediction] = {}
    for prediction in predictions:
        key = (prediction.sample_id, prediction.method_id)
        if key in result:
            raise ValueError(f"duplicate prediction key: {key}")
        result[key] = prediction
    if set(result) != expected:
        raise ValueError(
            "prediction keys must match exactly; "
            f"missing={sorted(expected - set(result))[:5]!r}, "
            f"extra={sorted(set(result) - expected)[:5]!r}"
        )
    return result


def _diagnostic_key(record: Mapping[str, Any]) -> tuple[str, str]:
    try:
        key = record["key"]
        if not isinstance(key, Mapping):
            raise TypeError("diagnostic key must be an object")
        return str(key["sample_id"]), str(key["method_id"])
    except (KeyError, TypeError) as exc:
        raise ValueError("diagnostic record is missing its key") from exc


def _diagnostic_index(
    samples: Sequence[BenchmarkSample],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected = {
        (sample.sample_id, method)
        for sample in samples
        for method in PAVG_DIAGNOSTIC_METHODS
    }
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in diagnostics:
        if not isinstance(record, Mapping):
            raise ValueError("diagnostic record must be an object")
        key = _diagnostic_key(record)
        if key in result:
            raise ValueError(f"duplicate diagnostic key: {key}")
        validated = validate_pavg_diagnostic(record)
        expected_override = _expected_hard_override(validated)
        if validated["hard_violation_override"] is not expected_override:
            raise ValueError(
                "hard_violation_override mismatch for diagnostic key "
                f"{key!r}: expected {expected_override}"
            )
        result[key] = validated
    if set(result) != expected:
        raise ValueError(
            "diagnostic keys must match exactly; "
            f"missing={sorted(expected - set(result))[:5]!r}, "
            f"extra={sorted(set(result) - expected)[:5]!r}"
        )
    return result


def _by_method(
    samples: Sequence[BenchmarkSample],
    indexed: Mapping[tuple[str, str], BenchmarkPrediction],
    method: str,
) -> tuple[BenchmarkPrediction, ...]:
    return tuple(indexed[(sample.sample_id, method)] for sample in samples)


def _prediction_label(prediction: BenchmarkPrediction) -> str:
    return "unknown" if prediction.failure is not None else prediction.physics_label


def _correct(sample: BenchmarkSample, prediction: BenchmarkPrediction) -> bool:
    return _prediction_label(prediction) == sample.physics_label


def _sequential_attribution(
    samples: Sequence[BenchmarkSample],
    indexed: Mapping[tuple[str, str], BenchmarkPrediction],
    diagnostics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    result = {}
    for baseline, candidate in SEQUENTIAL_TRANSITIONS:
        changed = gains = losses = 0
        baseline_failures = candidate_failures = 0
        module_available = 0
        for sample in samples:
            before = indexed[(sample.sample_id, baseline)]
            after = indexed[(sample.sample_id, candidate)]
            changed += _prediction_label(before) != _prediction_label(after)
            before_correct = _correct(sample, before)
            after_correct = _correct(sample, after)
            gains += after_correct and not before_correct
            losses += before_correct and not after_correct
            baseline_failures += before.failure is not None
            candidate_failures += after.failure is not None
            record = diagnostics[(sample.sample_id, candidate)]
            evidence = record["evidence_families"]
            if candidate == "M1_GRAPH":
                available = evidence["pqsg"]["status"] == "available"
            elif candidate == "M2_CHECKLIST":
                available = evidence["checklist"]["status"] == "available"
            elif candidate == "M3_MECHANICS":
                available = evidence["mechanics"]["status"] == "available"
            elif candidate == "M4_VLM":
                available = evidence["vlm"]["status"] == "available"
            else:
                available = (
                    record["planner"]["source"] == "model"
                    and bool(record["question_graph"]["source"])
                )
            module_available += available
        result[f"{candidate}-{baseline}"] = {
            "changed": changed,
            "gains": gains,
            "losses": losses,
            "baseline_failures": baseline_failures,
            "candidate_failures": candidate_failures,
            "failure_change": candidate_failures - baseline_failures,
            "module_available": module_available,
            "module_unavailable": len(samples) - module_available,
        }
    return result


def _module_availability(
    samples: Sequence[BenchmarkSample],
    diagnostics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts = {
        name: {"available": 0, "unavailable": 0}
        for name in (
            "planner",
            "pqsg",
            "video_science",
            "mechanics",
            "rules",
            "vlm",
        )
    }
    for sample in samples:
        record = diagnostics[(sample.sample_id, "M5_FULL")]
        planner = record.get("planner")
        planner_available = isinstance(planner, Mapping) and bool(planner.get("source"))
        evidence = record.get("evidence_families")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        flags = {
            "planner": planner_available,
            "pqsg": isinstance(evidence.get("pqsg"), Mapping)
            and evidence["pqsg"].get("status") == "available",
            "video_science": isinstance(evidence.get("checklist"), Mapping)
            and evidence["checklist"].get("status") == "available",
            "mechanics": isinstance(evidence.get("mechanics"), Mapping)
            and evidence["mechanics"].get("status") == "available",
            "rules": isinstance(evidence.get("rules"), Mapping)
            and evidence["rules"].get("status") == "available",
            "vlm": isinstance(evidence.get("vlm"), Mapping)
            and evidence["vlm"].get("status") == "available",
        }
        for name, available in flags.items():
            counts[name]["available" if available else "unavailable"] += 1
    return counts


def _model_call_summary(
    samples: Sequence[BenchmarkSample],
    diagnostics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    totals: dict[str, Counter] = {}
    for sample in samples:
        calls = diagnostics[(sample.sample_id, "M5_FULL")].get("model_calls", {})
        if not isinstance(calls, Mapping):
            continue
        for stage, values in calls.items():
            if not isinstance(values, Mapping):
                continue
            counter = totals.setdefault(str(stage), Counter())
            for field in (
                "call_count",
                "provider_call_count",
                "cache_hit_count",
                "error_count",
            ):
                counter[field] += int(values.get(field, 0))
            counter["latency_sec"] += float(values.get("latency_sec", 0.0))
    return {stage: dict(values) for stage, values in sorted(totals.items())}


def _metric_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    return {
        field: float(candidate[field]) - float(baseline[field])
        for field in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "physical_recall",
            "violation_recall",
            "failure_rate",
            "mean_latency_sec",
        )
    }


def _prompt_comparison(
    samples: Sequence[BenchmarkSample],
    baseline: Sequence[BenchmarkPrediction],
    candidate: Sequence[BenchmarkPrediction],
    *,
    baseline_method: str,
    candidate_method: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_metrics = strict_method_metrics(
        samples, baseline, expected_method=baseline_method
    )
    candidate_metrics = strict_method_metrics(
        samples, candidate, expected_method=candidate_method
    )
    return {
        "baseline": baseline_method,
        "candidate": candidate_method,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_baseline": _metric_delta(
            baseline_metrics, candidate_metrics
        ),
        "paired_outcomes": paired_outcomes(
            samples,
            baseline,
            candidate,
            baseline_method=baseline_method,
            candidate_method=candidate_method,
        ),
        "bootstrap": action_group_bootstrap(
            samples,
            baseline,
            candidate,
            baseline_method=baseline_method,
            candidate_method=candidate_method,
            resamples=resamples,
            seed=seed,
        ),
    }


def build_full_pavg_report(
    *,
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[BenchmarkPrediction],
    diagnostics: Sequence[Mapping[str, Any]],
    pilot_samples: Sequence[BenchmarkSample],
    prompt_predictions: Sequence[BenchmarkPrediction],
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260717,
) -> dict[str, Any]:
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    full_index = _prediction_index(samples, predictions, FULL_METHODS)
    diagnostic_index = _diagnostic_index(samples, diagnostics)
    metrics = {
        method: strict_method_metrics(
            samples,
            _by_method(samples, full_index, method),
            expected_method=method,
        )
        for method in FULL_METHODS
    }
    d0 = _by_method(samples, full_index, "D0_DIRECT_VLM")
    m5 = _by_method(samples, full_index, "M5_FULL")
    primary_bootstrap = action_group_bootstrap(
        samples,
        d0,
        m5,
        baseline_method="D0_DIRECT_VLM",
        candidate_method="M5_FULL",
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    slices = build_slices(
        samples,
        d0,
        m5,
        baseline_method="D0_DIRECT_VLM",
        candidate_method="M5_FULL",
    )
    material = evaluate_material_improvement(
        baseline_metrics=metrics["D0_DIRECT_VLM"],
        candidate_metrics=metrics["M5_FULL"],
        bootstrap=primary_bootstrap,
        slices=slices,
        baseline_failure_rate=float(metrics["D0_DIRECT_VLM"]["failure_rate"]),
        candidate_failure_rate=float(metrics["M5_FULL"]["failure_rate"]),
    )

    pilot_ids = [sample.sample_id for sample in pilot_samples]
    if len(pilot_ids) != len(set(pilot_ids)) or not set(pilot_ids).issubset(
        sample.sample_id for sample in samples
    ):
        raise ValueError("pilot samples must be unique members of the full population")
    prompt_index = _prediction_index(
        pilot_samples, prompt_predictions, PROMPT_METHODS
    )
    for sample in pilot_samples:
        primary = full_index[(sample.sample_id, "M5_FULL")]
        diagnostic = prompt_index[(sample.sample_id, "M5_FULL")]
        if diagnostic != primary:
            raise ValueError(
                "correct-prompt M5 prediction must equal the primary full-run "
                f"subset for sample {sample.sample_id!r}"
            )
    correct = _by_method(pilot_samples, prompt_index, "M5_FULL")
    shuffled = _by_method(
        pilot_samples, prompt_index, "M5_SHUFFLED_PROMPT_300"
    )
    oracle = _by_method(pilot_samples, prompt_index, "M5_ORACLE_PLAN_300")
    hard_overrides = sum(
        bool(diagnostic_index[(sample.sample_id, "M5_FULL")].get("hard_violation_override"))
        for sample in samples
    )
    provider_failures = sum(
        len(record.get("provider_failures", ()))
        for record in diagnostic_index.values()
        if isinstance(record.get("provider_failures", ()), (list, tuple))
    )
    return {
        "schema_version": "1.0",
        "population": {
            "sample_count": len(samples),
            "prediction_count": len(predictions),
            "diagnostic_count": len(diagnostics),
        },
        "primary": {
            "baseline": "D0_DIRECT_VLM",
            "candidate": "M5_FULL",
            "candidate_minus_baseline": _metric_delta(
                metrics["D0_DIRECT_VLM"], metrics["M5_FULL"]
            ),
            "paired_outcomes": paired_outcomes(
                samples,
                d0,
                m5,
                baseline_method="D0_DIRECT_VLM",
                candidate_method="M5_FULL",
            ),
            "bootstrap": primary_bootstrap,
        },
        "method_metrics": metrics,
        "sequential_attribution": _sequential_attribution(
            samples, full_index, diagnostic_index
        ),
        "module_availability": _module_availability(samples, diagnostic_index),
        "model_calls": _model_call_summary(samples, diagnostic_index),
        "hard_override": {
            "forced_violation": hard_overrides,
            "rate": hard_overrides / len(samples),
        },
        "provider_failure_count": provider_failures,
        "slices": slices,
        "material_decision": material,
        "prompt_diagnostics": {
            "scope": "diagnostic_only",
            "sample_count": len(pilot_samples),
            "correct_minus_shuffled": _prompt_comparison(
                pilot_samples,
                shuffled,
                correct,
                baseline_method="M5_SHUFFLED_PROMPT_300",
                candidate_method="M5_FULL",
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
            "oracle_minus_correct": _prompt_comparison(
                pilot_samples,
                correct,
                oracle,
                baseline_method="M5_FULL",
                candidate_method="M5_ORACLE_PLAN_300",
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
        },
        "ood_evaluation": {
            "benchmark": "VideoPhy-1",
            "status": "deferred",
            "overall_verdict": "not_evaluable_ood_deferred",
        },
    }
