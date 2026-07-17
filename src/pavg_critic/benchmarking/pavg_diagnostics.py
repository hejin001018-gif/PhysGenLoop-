"""Build compact, non-secret module diagnostics for PAVG benchmark records."""

from __future__ import annotations

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
    return {
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

    return {
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
