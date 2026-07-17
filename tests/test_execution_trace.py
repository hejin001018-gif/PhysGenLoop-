"""Execution trace contracts for observable, non-secret Critic audits."""

from __future__ import annotations

import pytest

from pavg_critic.execution_trace import TraceRecorder, TraceSafetyError


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

