"""Command-line behavior for independent Critic trace validation."""

from __future__ import annotations

import json

from examples.validate_pipeline_trace import main
from pavg_critic.config import CriticConfig
from pavg_critic.execution_trace import TraceRecorder, build_fusion_audit
from pavg_critic.schemas import CriticReport, EvidenceBundle


def _trace_document() -> dict[str, object]:
    recorder = TraceRecorder(
        metadata={
            "detector": {"backend": "sam2", "sam2_used": True},
            "planner": {"source": "model", "fallback_used": False},
            "provider_fallback_count": 0,
        }
    )
    fixed = (
        "request",
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
    )
    for node_id in fixed:
        recorder.record_skipped(
            node_id,
            label=node_id,
            source_nodes=(),
            inputs={},
            reason="fixture",
        )
    bundles = tuple(
        EvidenceBundle(
            family=family,
            source=f"{family}_source",
            status="available",
            score=0.8,
            confidence=1.0,
            coverage=1.0,
        )
        for family in ("rules", "pqsg", "checklist", "mechanics", "vlm")
    )
    report = CriticReport(
        is_physical=True,
        decision="physical",
        physics_score=0.8,
        confidence=1.0,
        coverage=1.0,
        evidence_bundles=bundles,
    )
    recorder.record_completed(
        "evidence_fusion",
        label="evidence_fusion",
        source_nodes=(),
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
            "decision": "physical",
            "physics_score": 0.8,
            "confidence": 1.0,
            "coverage": 1.0,
            "violations": [],
        },
        elapsed_ms=0.0,
    )
    recorder.set_outcome({"status": "completed", "decision": "physical"})
    return recorder.to_dict()


def test_validator_cli_accepts_strict_valid_trace(tmp_path, capsys):
    path = tmp_path / "valid.trace.json"
    path.write_text(
        json.dumps(_trace_document(), ensure_ascii=False), encoding="utf-8"
    )

    exit_code = main(
        [
            str(path),
            "--require-sam2",
            "--require-model-planner",
            "--fail-on-provider-fallback",
        ]
    )

    assert exit_code == 0
    assert "校验通过" in capsys.readouterr().out


def test_validator_cli_fails_tampered_trace(tmp_path, capsys):
    document = _trace_document()
    fusion = next(
        node
        for node in document["nodes"]
        if node["node_id"] == "evidence_fusion"
    )
    fusion["outputs"]["families"][0]["effective_weight"] = 99.0
    path = tmp_path / "tampered.trace.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    exit_code = main([str(path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "[FAIL] fusion.effective_weight" in output
    assert "校验失败" in output


def test_validator_cli_returns_two_for_invalid_file(tmp_path, capsys):
    exit_code = main([str(tmp_path / "missing.json")])

    assert exit_code == 2
    assert "无法读取 trace" in capsys.readouterr().err

