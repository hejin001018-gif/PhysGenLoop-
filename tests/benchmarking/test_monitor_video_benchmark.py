"""Read-only benchmark heartbeat and deterministic stall detection."""

from __future__ import annotations

import json

from benchmarks.monitor_video_benchmark import (
    append_heartbeat,
    build_snapshot,
)


def _write_prediction(path, *, sample_id, method, failure=None):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "method_id": method,
                    "failure": failure,
                }
            )
            + "\n"
        )


def test_snapshot_reports_progress_gpu_endpoint_and_eta(tmp_path):
    m4 = tmp_path / "m4.jsonl"
    m5 = tmp_path / "m5.jsonl"
    for index in range(10):
        _write_prediction(m4, sample_id=str(index), method="M4_VLM")
        _write_prediction(m5, sample_id=str(index), method="M5_FULL")
    previous = {
        "timestamp_epoch": 700.0,
        "prediction_count": 10,
        "last_progress_epoch": 700.0,
    }

    snapshot = build_snapshot(
        {"M4_VLM": m4, "M5_FULL": m5},
        expected_per_method=300,
        previous=previous,
        now=1_000.0,
        stall_sec=900,
        gpu_query=lambda: {
            "utilization_percent": 91,
            "memory_used_mib": 23000,
            "memory_total_mib": 40960,
        },
        endpoint_probe=lambda: True,
    )

    assert snapshot["prediction_count"] == 20
    assert snapshot["expected_count"] == 600
    assert snapshot["failure_count"] == 0
    assert snapshot["endpoint_healthy"] is True
    assert snapshot["gpu"]["utilization_percent"] == 91
    assert snapshot["stalled"] is False
    assert snapshot["eta_sec"] == 17400.0
    assert snapshot["secrets_recorded"] is False


def test_snapshot_marks_no_progress_as_stalled(tmp_path):
    predictions = tmp_path / "m5.jsonl"
    _write_prediction(predictions, sample_id="1", method="M5_FULL")
    previous = {
        "timestamp_epoch": 100.0,
        "prediction_count": 1,
        "last_progress_epoch": 100.0,
    }

    snapshot = build_snapshot(
        {"M5_FULL": predictions},
        expected_per_method=2,
        previous=previous,
        now=1_001.0,
        stall_sec=900,
        gpu_query=lambda: {},
        endpoint_probe=lambda: False,
    )

    assert snapshot["stalled"] is True
    assert snapshot["last_progress_epoch"] == 100.0
    assert snapshot["endpoint_healthy"] is False


def test_heartbeat_append_is_one_fsynced_json_line(tmp_path):
    heartbeat = tmp_path / "heartbeat.jsonl"
    snapshot = {
        "timestamp_epoch": 1.0,
        "prediction_count": 0,
        "secrets_recorded": False,
    }

    append_heartbeat(heartbeat, snapshot)

    assert json.loads(heartbeat.read_text(encoding="utf-8")) == snapshot
