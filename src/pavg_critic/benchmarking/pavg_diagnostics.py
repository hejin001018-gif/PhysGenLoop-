"""Build compact, non-secret module diagnostics for PAVG benchmark records."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Mapping, Sequence

from pavg_critic.config import CriticConfig
from pavg_critic.schemas import CriticArtifacts, EVIDENCE_FAMILIES

from .contracts import BenchmarkSample
from .model_cache import ModelCallEvent


_NODE_STATUSES = ("yes", "no", "blocked", "unknown")
_CHECKLIST_STATUSES = ("pass", "fail", "unknown")
_MECHANICS_STATUSES = ("applicable", "not_applicable", "failed")
_VLM_STATUSES = ("confirmed", "rejected", "uncertain", "unavailable")


def _counts(values, names: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(values)
    return {name: int(counts.get(name, 0)) for name in names}


def _stage_calls(
    stage_events: Mapping[str, Sequence[ModelCallEvent]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for stage in sorted(stage_events):
        events = tuple(stage_events[stage])
        result[stage] = {
            "call_count": len(events),
            "provider_call_count": sum(not event.cache_hit for event in events),
            "cache_hit_count": sum(event.cache_hit for event in events),
            "error_count": sum(event.error_type is not None for event in events),
            "latency_sec": sum(event.latency_sec for event in events),
            "events": [event.to_dict() for event in events],
        }
    return result


def _configured_weights(config: CriticConfig) -> dict[str, float]:
    fusion = config.fusion
    return {
        "rules": fusion.rule_family_weight,
        "pqsg": fusion.pqsg_family_weight,
        "checklist": fusion.checklist_family_weight,
        "mechanics": fusion.mechanics_family_weight,
        "vlm": fusion.vlm_family_weight,
    }


def _evidence_families(
    artifacts: CriticArtifacts,
    config: CriticConfig,
) -> dict[str, dict[str, object]]:
    bundles = {bundle.family: bundle for bundle in artifacts.report.evidence_bundles}
    weights = _configured_weights(config)
    result: dict[str, dict[str, object]] = {}
    for family in EVIDENCE_FAMILIES:
        bundle = bundles.get(family)
        configured_weight = weights[family]
        if bundle is None:
            result[family] = {
                "source": None,
                "status": "unavailable",
                "score": None,
                "confidence": 0.0,
                "coverage": 0.0,
                "configured_weight": configured_weight,
                "effective_weight": 0.0,
            }
            continue
        effective_weight = (
            configured_weight * bundle.coverage * bundle.confidence
            if bundle.status == "available"
            else 0.0
        )
        result[family] = {
            "source": bundle.source,
            "status": bundle.status,
            "score": bundle.score,
            "confidence": bundle.confidence,
            "coverage": bundle.coverage,
            "configured_weight": configured_weight,
            "effective_weight": effective_weight,
        }
    return result


def build_pavg_diagnostics(
    *,
    sample: BenchmarkSample,
    method_id: str,
    artifacts: CriticArtifacts,
    config: CriticConfig,
    stage_events: Mapping[str, Sequence[ModelCallEvent]],
    analysis_latency_sec: float,
    total_latency_sec: float,
    visible_frame_count: int,
) -> dict[str, object]:
    """Return one JSON-safe sidecar without prompts, plans, images or reasons."""

    if artifacts.resolved_request is None:
        raise ValueError("PAVG diagnostics require a resolved request")
    plan = artifacts.resolved_request.physics_plan
    metadata = plan.planner_metadata
    graph = artifacts.question_graph
    summary = artifacts.report.graph_evaluation
    checklist_summary = artifacts.checklist_summary
    mechanics_summary = artifacts.mechanics_summary
    review_statuses = [
        "unavailable" if review is None else review.claim_status
        for _, review in sorted(artifacts.reviews.items())
    ]
    provider_failures = artifacts.report.diagnostics.get("provider_failures", ())
    sanitized_provider_failures = [
        {
            "stage": str(item.get("stage", "unknown")),
            "error_type": str(item.get("error_type", "unknown")),
        }
        for item in provider_failures
        if isinstance(item, Mapping)
    ]
    pre_fusion = artifacts.report.diagnostics.get("pre_evidence_fusion", {})
    record = {
        "schema_version": "1.0",
        "key": {"sample_id": sample.sample_id, "method_id": method_id},
        "planner": {
            "source": metadata.source,
            "fallback_used": metadata.fallback_used,
            "confidence": metadata.confidence,
            "model": metadata.model,
            "object_count": len(plan.objects),
            "expected_event_count": len(plan.expected_events),
            "relation_count": len(plan.relations),
            "constraint_count": len(plan.physics_constraints),
        },
        "question_graph": {
            "source": None if graph is None else graph.source,
            "node_count": 0 if graph is None else len(graph.nodes),
            "status_counts": _counts(
                (result.status for result in artifacts.node_results),
                _NODE_STATUSES,
            ),
            "question_coverage": None if summary is None else summary.question_coverage,
            "physics_coverage": None if summary is None else summary.physics_coverage,
        },
        "video_science": {
            "enabled": config.checklist.enabled,
            "coverage": None if checklist_summary is None else checklist_summary.coverage,
            "status_counts": _counts(
                (result.status for result in artifacts.checklist_results),
                _CHECKLIST_STATUSES,
            ),
        },
        "mechanics": {
            "enabled": config.mechanics.enabled,
            "coverage": None if mechanics_summary is None else mechanics_summary.coverage,
            "applicability_counts": _counts(
                (result.applicability for result in artifacts.mechanics_results),
                _MECHANICS_STATUSES,
            ),
            "evaluator_scores": {
                result.evaluator: result.score for result in artifacts.mechanics_results
            },
        },
        "rules": {
            "candidate_count": len(artifacts.candidates),
            "retained_violation_count": len(artifacts.report.violations),
            "candidate_categories": dict(
                sorted(Counter(item.category for item in artifacts.candidates).items())
            ),
            "retained_categories": dict(
                sorted(Counter(item.category for item in artifacts.report.violations).items())
            ),
        },
        "vlm_reviews": {
            "candidate_count": len(artifacts.candidates),
            "review_slot_count": len(artifacts.reviews),
            "status_counts": _counts(review_statuses, _VLM_STATUSES),
        },
        "evidence_families": _evidence_families(artifacts, config),
        "fusion": {
            "pre_evidence_fusion": dict(pre_fusion),
            "final": {
                "decision": artifacts.report.decision,
                "physics_score": artifacts.report.physics_score,
                "confidence": artifacts.report.confidence,
                "coverage": artifacts.report.coverage,
            },
        },
        "hard_violation_override": bool(
            artifacts.report.diagnostics.get("hard_violation_override", False)
        ),
        "model_calls": _stage_calls(stage_events),
        "latency": {
            "analysis_sec": analysis_latency_sec,
            "total_sec": total_latency_sec,
            "visible_frame_count": visible_frame_count,
        },
        "provider_failures": sanitized_provider_failures,
        "failure": None,
    }
    return validate_pavg_diagnostic(record)


def build_pavg_failure_diagnostics(
    *,
    sample: BenchmarkSample,
    method_id: str,
    stage_events: Mapping[str, Sequence[ModelCallEvent]],
    total_latency_sec: float,
    visible_frame_count: int,
    error: BaseException,
) -> dict[str, object]:
    """Return a minimal keyed sidecar when no complete artifacts are available."""

    record = {
        "schema_version": "1.0",
        "key": {"sample_id": sample.sample_id, "method_id": method_id},
        "planner": None,
        "question_graph": None,
        "video_science": None,
        "mechanics": None,
        "rules": None,
        "vlm_reviews": None,
        "evidence_families": None,
        "fusion": None,
        "hard_violation_override": False,
        "model_calls": _stage_calls(stage_events),
        "latency": {
            "analysis_sec": None,
            "total_sec": total_latency_sec,
            "visible_frame_count": visible_frame_count,
        },
        "provider_failures": [],
        "failure": {"error_type": type(error).__name__},
    }
    return validate_pavg_diagnostic(record)


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "key",
    "planner",
    "question_graph",
    "video_science",
    "mechanics",
    "rules",
    "vlm_reviews",
    "evidence_families",
    "fusion",
    "hard_violation_override",
    "model_calls",
    "latency",
    "provider_failures",
    "failure",
}
_SECTION_FIELDS = {
    "key": {"sample_id", "method_id"},
    "planner": {
        "source",
        "fallback_used",
        "confidence",
        "model",
        "object_count",
        "expected_event_count",
        "relation_count",
        "constraint_count",
    },
    "question_graph": {
        "source",
        "node_count",
        "status_counts",
        "question_coverage",
        "physics_coverage",
    },
    "video_science": {"enabled", "coverage", "status_counts"},
    "mechanics": {
        "enabled",
        "coverage",
        "applicability_counts",
        "evaluator_scores",
    },
    "rules": {
        "candidate_count",
        "retained_violation_count",
        "candidate_categories",
        "retained_categories",
    },
    "vlm_reviews": {"candidate_count", "review_slot_count", "status_counts"},
    "latency": {"analysis_sec", "total_sec", "visible_frame_count"},
}
_EVIDENCE_FIELDS = {
    "source",
    "status",
    "score",
    "confidence",
    "coverage",
    "configured_weight",
    "effective_weight",
}
_MODEL_CALL_FIELDS = {
    "call_count",
    "provider_call_count",
    "cache_hit_count",
    "error_count",
    "latency_sec",
    "events",
}
_MODEL_EVENT_FIELDS = {
    "namespace",
    "model_id",
    "model_revision",
    "sample_id",
    "cache_key",
    "prompt_sha256",
    "schema_sha256",
    "input_evidence_sha256",
    "cache_hit",
    "latency_sec",
    "error_type",
}
_FUSION_FIELDS = {"decision", "physics_score", "confidence", "coverage"}
_FORBIDDEN_CONTENT = (
    "authorization",
    "bearer ",
    "data:image",
    "api_key",
    "bench_api_key",
    "raw provider",
    "chain_of_thought",
    "sk-",
)


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"unexpected diagnostic fields in {context}: "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _validate_safe_values(value: object, *, path: str = "diagnostic") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_safe_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_values(item, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite diagnostic number at {path}")
    if isinstance(value, str):
        lowered = value.lower()
        if any(mark in lowered for mark in _FORBIDDEN_CONTENT):
            raise ValueError(f"forbidden diagnostic content at {path}")


def validate_pavg_diagnostic(record: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the exact public sidecar schema."""

    root = _require_exact_fields(record, _TOP_LEVEL_FIELDS, context="root")
    if root.get("schema_version") != "1.0":
        raise ValueError("diagnostic schema_version must be 1.0")
    _require_exact_fields(root["key"], _SECTION_FIELDS["key"], context="key")
    for name in (
        "planner",
        "question_graph",
        "video_science",
        "mechanics",
        "rules",
        "vlm_reviews",
    ):
        value = root[name]
        if value is not None:
            section = _require_exact_fields(
                value, _SECTION_FIELDS[name], context=name
            )
            if name == "question_graph":
                _require_exact_fields(
                    section["status_counts"],
                    set(_NODE_STATUSES),
                    context="question_graph.status_counts",
                )
            elif name == "video_science":
                _require_exact_fields(
                    section["status_counts"],
                    set(_CHECKLIST_STATUSES),
                    context="video_science.status_counts",
                )
            elif name == "mechanics":
                _require_exact_fields(
                    section["applicability_counts"],
                    set(_MECHANICS_STATUSES),
                    context="mechanics.applicability_counts",
                )
            elif name == "vlm_reviews":
                _require_exact_fields(
                    section["status_counts"],
                    set(_VLM_STATUSES),
                    context="vlm_reviews.status_counts",
                )
    evidence = root["evidence_families"]
    if evidence is not None:
        families = _require_exact_fields(
            evidence, set(EVIDENCE_FAMILIES), context="evidence_families"
        )
        for family, item in families.items():
            _require_exact_fields(
                item, _EVIDENCE_FIELDS, context=f"evidence_families.{family}"
            )
    fusion = root["fusion"]
    if fusion is not None:
        fusion_mapping = _require_exact_fields(
            fusion,
            {"pre_evidence_fusion", "final"},
            context="fusion",
        )
        _require_exact_fields(
            fusion_mapping["pre_evidence_fusion"],
            _FUSION_FIELDS,
            context="fusion.pre_evidence_fusion",
        )
        _require_exact_fields(
            fusion_mapping["final"],
            _FUSION_FIELDS,
            context="fusion.final",
        )
    calls = root["model_calls"]
    if not isinstance(calls, Mapping):
        raise ValueError("model_calls must be an object")
    if set(calls) - {"planner", "pqsg", "verifier"}:
        raise ValueError("unexpected diagnostic fields in model_calls")
    for stage, raw_call in calls.items():
        call = _require_exact_fields(
            raw_call, _MODEL_CALL_FIELDS, context=f"model_calls.{stage}"
        )
        events = call["events"]
        if not isinstance(events, list):
            raise ValueError(f"model_calls.{stage}.events must be an array")
        for index, event in enumerate(events):
            _require_exact_fields(
                event,
                _MODEL_EVENT_FIELDS,
                context=f"model_calls.{stage}.events[{index}]",
            )
    _require_exact_fields(
        root["latency"], _SECTION_FIELDS["latency"], context="latency"
    )
    failures = root["provider_failures"]
    if not isinstance(failures, list):
        raise ValueError("provider_failures must be an array")
    for index, failure in enumerate(failures):
        _require_exact_fields(
            failure,
            {"stage", "error_type"},
            context=f"provider_failures[{index}]",
        )
    failure = root["failure"]
    if failure is not None:
        _require_exact_fields(failure, {"error_type"}, context="failure")
    if not isinstance(root["hard_violation_override"], bool):
        raise ValueError("hard_violation_override must be boolean")
    _validate_safe_values(root)
    return json.loads(
        json.dumps(
            root,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
