"""三份 JSON Schema 自校验：schema 本身合法 + README 示例通过校验。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_sample_schema_is_valid():
    Draft202012Validator.check_schema(_load("sample.schema.json"))


def test_critic_schema_is_valid():
    Draft202012Validator.check_schema(_load("critic_output.schema.json"))


def test_generator_schema_is_valid():
    Draft202012Validator.check_schema(_load("generator_request.schema.json"))


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


def test_readme_critic_example_validates():
    payload = {
        "schema_version": "1.0",
        "is_physical": False,
        "physics_score": 0.46,
        "confidence": 0.91,
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
    }
    jsonschema.validate(payload, _load("critic_output.schema.json"))


def test_readme_generator_example_validates():
    payload = {
        "prompt": "A red ball falls from a table.",
        "seed": 42,
        "resolution": "480p",
        "num_frames": 121,
        "num_inference_steps": 50,
        "image_path": None,
        "output_path": "outputs/video.mp4",
    }
    jsonschema.validate(payload, _load("generator_request.schema.json"))
