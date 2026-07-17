"""Safety behavior for the end-to-end example's sparse VLM fallback."""

import json
from pathlib import Path

import examples.evaluate_video as evaluate_module
from examples.evaluate_video import (
    _configure_sparse_vlm_fallback,
    _resolve_sam2_checkpoint,
    build_parser,
)
from pavg_critic.config import CriticConfig


def test_sparse_vlm_fallback_disables_dense_disappearance_rule():
    configured = _configure_sparse_vlm_fallback(
        CriticConfig(),
        total_frames=61,
        width=640,
        height=360,
        num_keyframes=8,
    )

    assert "object_disappearance" not in configured.rules.enabled
    assert configured.events.min_disappearance_frames >= 16


def test_checkpoint_resolver_prefers_explicit_then_frozen_repo_path(
    tmp_path, monkeypatch
):
    frozen = tmp_path / "evaluation/external/models/sam2.1_hiera_base_plus.pt"
    frozen.parent.mkdir(parents=True)
    frozen.touch()
    explicit = tmp_path / "explicit.pt"
    explicit.touch()
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("SAM2_CHECKPOINT", str(explicit))
    assert _resolve_sam2_checkpoint() == explicit.resolve()
    monkeypatch.delenv("SAM2_CHECKPOINT")
    assert _resolve_sam2_checkpoint() == frozen.resolve()


def test_checkpoint_resolver_returns_actionable_default_when_missing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SAM2_CHECKPOINT", raising=False)

    assert _resolve_sam2_checkpoint() == Path(
        "evaluation/external/models/sam2.1_hiera_base_plus.pt"
    ).resolve()


def test_parser_accepts_live_and_persistent_trace_flags(tmp_path):
    trace_path = tmp_path / "video.trace.json"

    args = build_parser().parse_args(
        [
            "--video",
            "video.mp4",
            "--trace",
            "--trace-output",
            str(trace_path),
        ]
    )

    assert args.trace is True
    assert args.trace_output == trace_path


def test_main_streams_and_persists_trace_on_success(
    tmp_path, monkeypatch, capsys
):
    video = tmp_path / "video.mp4"
    video.touch()
    trace_path = tmp_path / "video.trace.json"

    monkeypatch.setattr(
        evaluate_module,
        "load_config",
        lambda: {
            "api_key": "in-memory-only",
            "base_url": "https://example.invalid/v1",
            "vlm_model": "test-vlm",
            "text_model": "test-text",
        },
    )

    def fake_evaluate(video_path, *, trace, **kwargs):
        trace.record_completed(
            "request",
            label="输入请求",
            source_nodes=(),
            inputs={"video_path": video_path},
            outputs={"accepted": True},
            elapsed_ms=1.5,
        )
        trace.set_outcome({"status": "completed", "decision": "physical"})
        return {"decision": "physical", "physics_score": 0.9}

    monkeypatch.setattr(evaluate_module, "evaluate_video", fake_evaluate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_video.py",
            "--video",
            str(video),
            "--trace",
            "--trace-output",
            str(trace_path),
        ],
    )

    assert evaluate_module.main() == 0
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    assert document["nodes"][0]["node_id"] == "request"
    assert document["outcome"]["decision"] == "physical"
    assert "[TRACE 01]" in capsys.readouterr().err


def test_main_persists_partial_trace_when_pipeline_fails(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()
    trace_path = tmp_path / "failed.trace.json"
    monkeypatch.setattr(
        evaluate_module,
        "load_config",
        lambda: {
            "api_key": "in-memory-only",
            "base_url": "https://example.invalid/v1",
            "vlm_model": "test-vlm",
            "text_model": "test-text",
        },
    )

    def failing_evaluate(video_path, *, trace, **kwargs):
        trace.record_completed(
            "request",
            label="输入请求",
            source_nodes=(),
            inputs={"video_path": video_path},
            outputs={"accepted": True},
            elapsed_ms=0.0,
        )
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(evaluate_module, "evaluate_video", failing_evaluate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_video.py",
            "--video",
            str(video),
            "--trace-output",
            str(trace_path),
        ],
    )

    assert evaluate_module.main() == 2
    document = json.loads(trace_path.read_text(encoding="utf-8"))
    assert document["nodes"][0]["node_id"] == "request"
    assert document["outcome"] == {
        "status": "error",
        "error_type": "RuntimeError",
    }
