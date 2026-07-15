"""Repair Agent 单测——不依赖任何外部 API，用 stub LLM 跑通决策链。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jsonschema

from agents.repairer import RepairConfig, RepairState, decide, repair_once
from agents.prompt_rewriter import make_client, RewriteRequest, rewrite
from agents.video_backend import make_backend


SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _gen_schema():
    return json.loads((SCHEMA_DIR / "generator_request.schema.json").read_text(encoding="utf-8"))


def _critic(score: float, is_physical: bool = False) -> dict:
    return {
        "schema_version": "1.0",
        "is_physical": is_physical,
        "physics_score": score,
        "confidence": 0.9,
        "violations": [{
            "object": "red_ball",
            "category": "premature_rebound",
            "start_frame": 47,
            "end_frame": 53,
            "critical_frames": [44, 47, 49, 53],
            "reason": "The ball reverses direction before contacting the floor.",
            "repair_instruction": "Keep the ball moving downward until visible floor contact.",
        }],
    }


def test_prompt_rewriter_stub_returns_json():
    res = rewrite(
        RewriteRequest(original_prompt="A red ball falls.", violations=_critic(0.4)["violations"]),
        client=make_client("stub"),
    )
    assert "ball" in res.prompt.lower()
    assert res.physics_hint


def test_repair_high_score_uses_prompt_only():
    d = repair_once(_critic(0.7), "A red ball falls.", backend="stub")
    assert d.action == "prompt_only"
    jsonschema.validate(d.generator_request, _gen_schema())


def test_repair_low_score_full_regen():
    d = repair_once(_critic(0.2), "A red ball falls.", backend="stub")
    assert d.action == "full_regen"
    jsonschema.validate(d.generator_request, _gen_schema())


def test_repair_stops_when_passed():
    d = repair_once(_critic(0.9, is_physical=True), "A red ball falls.", backend="stub")
    assert d.action == "stop"


def test_repair_stops_after_max_rounds():
    state = RepairState(original_prompt="A red ball falls.")
    cfg = RepairConfig(max_rounds=1)
    d1 = decide(_critic(0.3), state, cfg)
    d2 = decide(_critic(0.3), state, cfg)
    assert d1.action != "stop"
    assert d2.action == "stop"


def test_video_backend_stub_writes_file(tmp_path):
    backend = make_backend("stub")
    req = {
        "prompt": "test", "seed": 1, "resolution": "480p",
        "num_frames": 8, "num_inference_steps": 5, "image_path": None,
        "output_path": str(tmp_path / "out.mp4"),
    }
    res = backend.generate(req)
    assert res.ok
    assert Path(res.output_path).exists()
