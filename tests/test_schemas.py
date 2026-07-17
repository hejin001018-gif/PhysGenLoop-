"""三份 JSON Schema 自校验：schema 本身合法 + README 示例通过校验。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from jsonschema import Draft202012Validator
import pytest

from pavg_critic.execution_trace import TraceRecorder

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_sample_schema_is_valid():
    Draft202012Validator.check_schema(_load("sample.schema.json"))


def test_critic_schema_is_valid():
    Draft202012Validator.check_schema(_load("critic_output.schema.json"))


def test_readme_sample_example_validates():
    # 对齐 README §标注示例
    payload = {
        "schema_version": "1.0",
        "sample_id": "gravity_001",
        "source": "blender_synthetic",
        "is_physical": False,
        "video_path": "data/samples/gravity_001/video.mp4",
        "violations": [{
            "category": "reverse_gravity",
            "object": "red_ball",
            "start_frame": 38,
            "peak_frame": 45,
            "end_frame": 90,
            "critical_frames": [35, 38, 45, 90],
            "expected_rule": "unsupported objects accelerate downward",
            "repair_instruction": "Ensure continuous downward acceleration under gravity.",
        }],
    }
    jsonschema.validate(payload, _load("sample.schema.json"))


def test_critic_v2_example_validates():
    payload = {
        "schema_version": "2.0",
        "decision": "violation",
        "is_physical": False,
        "physics_score": 0.46,
        "confidence": 0.91,
        "coverage": 0.8,
        "score_breakdown": {"rules": 0.46},
        "violations": [{
            "object": "red_ball",
            "category": "premature_rebound",
            "start_frame": 47,
            "peak_frame": 49,
            "end_frame": 53,
            "critical_frames": [44, 47, 49, 53],
            "reason": "The ball reverses direction before contacting the floor.",
            "repair_instruction": "Keep the ball moving downward until visible floor contact.",
            "evidence": {
                "rules": ["velocity_reversal_before_contact"],
                "detector_score": 0.94,
                "vlm_score": 0.87,
            },
        }],
        "node_results": [],
        "diagnostics": {},
        "model_versions": {},
        "evidence_bundles": [],
    }
    jsonschema.validate(payload, _load("critic_output.schema.json"))


def _trace_payload() -> dict[str, object]:
    recorder = TraceRecorder(metadata={"detector": {"sam2_used": True}})
    recorder.record_completed(
        "request",
        label="输入请求",
        source_nodes=(),
        inputs={"prompt": "石头滚下坡"},
        outputs={"accepted": True},
        elapsed_ms=0.1,
    )
    recorder.set_outcome({"status": "completed", "decision": "physical"})
    return recorder.to_dict()


def test_critic_trace_schema_is_valid_and_accepts_recorder_document():
    schema = _load("critic_trace.schema.json")

    Draft202012Validator.check_schema(schema)
    jsonschema.validate(_trace_payload(), schema)


@pytest.mark.parametrize(
    ("mutator", "expected_path"),
    [
        (
            lambda payload: payload.update(schema_version="wrong"),
            ["schema_version"],
        ),
        (
            lambda payload: payload["nodes"][0].update(status="running"),
            ["nodes", 0, "status"],
        ),
        (
            lambda payload: payload["nodes"][0].update(sequence="one"),
            ["nodes", 0, "sequence"],
        ),
    ],
)
def test_critic_trace_schema_rejects_contract_mutations(mutator, expected_path):
    payload = _trace_payload()
    mutator(payload)

    with pytest.raises(jsonschema.ValidationError) as error:
        jsonschema.validate(payload, _load("critic_trace.schema.json"))

    assert list(error.value.absolute_path) == expected_path
