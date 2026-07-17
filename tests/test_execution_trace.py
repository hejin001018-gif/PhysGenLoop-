"""Execution trace contracts for observable, non-secret Critic audits."""

from __future__ import annotations

import pytest

import pavg_critic
from pavg_critic.config import CriticConfig
from pavg_critic.execution_trace import (
    TraceRecorder,
    TraceSafetyError,
    build_fusion_audit,
    validate_trace,
)
from pavg_critic.schemas import CriticReport, EvidenceBundle, Violation


def test_recorder_preserves_order_status_dependencies_and_elapsed_time():
    emitted: list[dict[str, object]] = []
    recorder = TraceRecorder(on_record=emitted.append)
    recorder.record_completed(
        "request",
        label="输入请求",
        source_nodes=(),
        inputs={"prompt": "石头滚下坡"},
        outputs={"accepted": True},
        elapsed_ms=0.0,
    )
    recorder.record_skipped(
        "mechanics",
        label="力学",
        source_nodes=("event_detection",),
        inputs={"event_count": 0},
        reason="not_applicable",
    )

    document = recorder.to_dict()

    assert document["schema_version"] == "pavg-critic-trace/v1"
    assert [node["sequence"] for node in document["nodes"]] == [1, 2]
    assert [node["status"] for node in document["nodes"]] == [
        "completed",
        "skipped",
    ]
    assert document["nodes"][1]["source_nodes"] == ["event_detection"]
    assert emitted == document["nodes"]


def test_node_context_records_sanitized_error_and_reraises():
    recorder = TraceRecorder()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        with recorder.node(
            "physics_planner",
            label="Planner",
            source_nodes=("request",),
            inputs={"prompt": "ball falls"},
        ):
            raise RuntimeError("provider unavailable")

    node = recorder.to_dict()["nodes"][0]
    assert node["status"] == "error"
    assert node["error"] == {
        "type": "RuntimeError",
        "message": "provider unavailable",
    }
    assert node["elapsed_ms"] >= 0.0


def test_node_context_can_mark_degraded_output():
    recorder = TraceRecorder()

    with recorder.node(
        "physics_planner",
        label="Planner",
        source_nodes=("request",),
        inputs={"prompt": "ball falls"},
    ) as node:
        node.degrade(
            outputs={"source": "template"},
            warnings=("SchemaError: model plan rejected",),
        )

    record = recorder.to_dict()["nodes"][0]
    assert record["status"] == "degraded"
    assert record["outputs"] == {"source": "template"}
    assert record["warnings"] == ["SchemaError: model plan rejected"]


@pytest.mark.parametrize(
    "key",
    ["api_key", "authorization", "headers", "raw_response", "image_bytes", "mask"],
)
def test_forbidden_trace_keys_are_rejected(key):
    recorder = TraceRecorder()

    with pytest.raises(TraceSafetyError, match="forbidden trace key"):
        recorder.record_completed(
            "unsafe",
            label="unsafe",
            source_nodes=(),
            inputs={key: "secret"},
            outputs={},
            elapsed_ms=0.0,
        )


def test_trace_rejects_binary_payloads_and_duplicate_node_ids():
    recorder = TraceRecorder()
    with pytest.raises(TraceSafetyError, match="binary payload"):
        recorder.record_completed(
            "bytes",
            label="bytes",
            source_nodes=(),
            inputs={"payload": b"not allowed"},
            outputs={},
            elapsed_ms=0.0,
        )

    recorder.record_completed(
        "request",
        label="request",
        source_nodes=(),
        inputs={},
        outputs={},
        elapsed_ms=0.0,
    )
    with pytest.raises(ValueError, match="duplicate trace node_id"):
        recorder.record_completed(
            "request",
            label="request",
            source_nodes=(),
            inputs={},
            outputs={},
            elapsed_ms=0.0,
        )


def test_large_collection_has_bounded_preview_and_digest():
    recorder = TraceRecorder()
    recorder.record_completed(
        "states",
        label="states",
        source_nodes=(),
        inputs={},
        outputs={"frames": list(range(30))},
        elapsed_ms=0.0,
    )

    frames = recorder.to_dict()["nodes"][0]["outputs"]["frames"]
    assert frames["count"] == 30
    assert frames["preview"] == list(range(20))
    assert len(frames["sha256"]) == 64
    assert frames["truncated"] is True


def test_live_render_callback_failure_does_not_change_traced_operation():
    def broken_renderer(record):
        raise RuntimeError("terminal renderer failed")

    recorder = TraceRecorder(on_record=broken_renderer)

    recorder.record_completed(
        "request",
        label="request",
        source_nodes=(),
        inputs={},
        outputs={"accepted": True},
        elapsed_ms=0.0,
    )

    document = recorder.to_dict()
    assert document["nodes"][0]["status"] == "completed"
    assert document["warnings"] == [
        "trace callback RuntimeError: terminal renderer failed"
    ]


def test_trace_recorder_and_validation_types_are_public_api():
    assert pavg_critic.TraceRecorder is TraceRecorder
    assert pavg_critic.TraceValidationPolicy.__name__ == "TraceValidationPolicy"
    assert pavg_critic.validate_trace is validate_trace


def _missing_bundle(family: str) -> EvidenceBundle:
    return EvidenceBundle(
        family=family,
        source=f"{family}_source",
        status="not_applicable",
        score=None,
        confidence=0.0,
        coverage=0.0,
    )


def _physical_report() -> CriticReport:
    return CriticReport(
        is_physical=True,
        decision="physical",
        physics_score=0.914894,
        confidence=0.282,
        coverage=0.4,
        evidence_bundles=(
            EvidenceBundle(
                family="rules",
                source="deterministic_rules",
                status="available",
                score=1.0,
                coverage=0.8,
                confidence=0.75,
            ),
            _missing_bundle("pqsg"),
            EvidenceBundle(
                family="checklist",
                source="video_science_checklist",
                status="available",
                score=2 / 3,
                coverage=0.6,
                confidence=0.6,
            ),
            _missing_bundle("mechanics"),
            _missing_bundle("vlm"),
        ),
    )


def test_fusion_audit_recomputes_effective_weights_and_score():
    report = _physical_report()

    audit = build_fusion_audit(CriticConfig().fusion, report)

    families = {row["family"]: row for row in audit["families"]}
    assert families["rules"]["effective_weight"] == pytest.approx(0.21)
    assert families["rules"]["weighted_contribution"] == pytest.approx(0.21)
    assert families["checklist"]["effective_weight"] == pytest.approx(0.072)
    assert audit["total_effective_weight"] == pytest.approx(0.282)
    assert audit["score_before_hard_violation"] == pytest.approx(0.914893617)
    assert audit["weighted_coverage"] == pytest.approx(0.4)
    assert audit["final_score"] == pytest.approx(report.physics_score)
    assert audit["final_confidence"] == pytest.approx(report.confidence)
    assert audit["final_decision"] == "physical"
    assert audit["hard_violation"] is False


def test_fusion_audit_caps_score_when_hard_violation_is_retained():
    supporting = tuple(
        EvidenceBundle(
            family=family,
            source=f"{family}_source",
            status="available",
            score=0.9,
            confidence=1.0,
            coverage=1.0,
        )
        for family in ("rules", "pqsg", "checklist", "mechanics", "vlm")
    )
    violation = Violation(
        object="ball",
        category="object_disappearance",
        start_frame=1,
        peak_frame=2,
        end_frame=3,
        critical_frames=(1, 2, 3),
        reason="confirmed",
        repair_instruction="keep visible",
        evidence={},
    )
    report = CriticReport(
        is_physical=False,
        decision="violation",
        physics_score=0.2,
        confidence=0.8,
        coverage=1.0,
        violations=(violation,),
        evidence_bundles=supporting,
    )

    audit = build_fusion_audit(CriticConfig().fusion, report)

    assert audit["score_before_hard_violation"] == pytest.approx(0.9)
    assert audit["hard_violation"] is True
    assert audit["hard_violation_score_cap"] == pytest.approx(0.2)
    assert audit["final_score"] == pytest.approx(0.2)
    assert audit["final_decision"] == "violation"


def _valid_trace_document() -> dict[str, object]:
    recorder = TraceRecorder(
        metadata={
            "detector": {"backend": "sam2", "sam2_used": True},
            "planner": {"source": "model"},
            "provider_fallback_count": 0,
        }
    )
    recorder.record_completed(
        "request",
        label="request",
        source_nodes=(),
        inputs={"prompt": "ball falls"},
        outputs={"accepted": True},
        elapsed_ms=0.0,
    )
    for node_id in (
        "physics_planner",
        "question_graph",
        "video_observation",
        "trajectory",
        "event_detection",
        "mechanics",
        "rule_engine",
        "temporal_localization",
        "visual_evidence",
        "checklist",
        "keyframe_selection",
        "pqsg_execution",
        "vlm_verification",
        "candidate_fusion",
        "question_scoring",
    ):
        recorder.record_skipped(
            node_id,
            label=node_id,
            source_nodes=(),
            inputs={},
            reason="fixture",
        )
    report = _physical_report()
    recorder.record_completed(
        "evidence_fusion",
        label="evidence_fusion",
        source_nodes=("candidate_fusion", "question_scoring"),
        inputs={},
        outputs=build_fusion_audit(CriticConfig().fusion, report),
        elapsed_ms=0.0,
    )
    recorder.record_completed(
        "final_report",
        label="final_report",
        source_nodes=("evidence_fusion",),
        inputs={},
        outputs={
            "decision": report.decision,
            "physics_score": report.physics_score,
            "confidence": report.confidence,
            "coverage": report.coverage,
            "violations": [],
        },
        elapsed_ms=0.0,
    )
    recorder.set_outcome({"status": "completed", "decision": report.decision})
    return recorder.to_dict()


def test_validator_accepts_consistent_fusion_trace():
    validation = validate_trace(_valid_trace_document())

    assert validation.passed
    assert all(check.passed for check in validation.checks if check.level == "error")


def test_validator_fails_when_effective_weight_is_tampered():
    trace = _valid_trace_document()
    fusion = next(
        node for node in trace["nodes"] if node["node_id"] == "evidence_fusion"
    )
    fusion["outputs"]["families"][0]["effective_weight"] += 0.1

    validation = validate_trace(trace)

    assert not validation.passed
    assert any(
        check.code == "fusion.effective_weight" and not check.passed
        for check in validation.checks
    )


def test_validator_rejects_review_filtered_candidate_in_final_violations():
    trace = _valid_trace_document()
    candidate_fusion = next(
        node for node in trace["nodes"] if node["node_id"] == "candidate_fusion"
    )
    candidate_fusion["status"] = "completed"
    candidate_fusion["outputs"] = {
        "candidates": [
            {"index": 0, "review_status": "rejected", "retained": True}
        ]
    }

    validation = validate_trace(trace)

    assert not validation.passed
    assert any(
        check.code == "fusion.review_filter" and not check.passed
        for check in validation.checks
    )
