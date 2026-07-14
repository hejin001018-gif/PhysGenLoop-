"""阶段 1：配置与统一报告契约。"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from pavg_critic import CriticReport, load_config
from pavg_critic.config import CriticConfig, EventConfig, RuleConfig, TemporalConfig


def test_yaml_config_is_supported(tmp_path):
    path = tmp_path / "critic.yaml"
    path.write_text(
        """
detector:
  backend: color_blob
trajectory:
  smoothing_window: 1
rules:
  enabled: [teleportation]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.trajectory.smoothing_window == 1
    assert config.rules.enabled == ("teleportation",)


def test_repository_default_config_is_loadable():
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"

    config = load_config(config_path)

    assert config.detector.backend == "color_blob"
    assert "premature_rebound" in config.rules.enabled


def test_runtime_report_matches_versioned_json_schema():
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "critic_output.schema.json").read_text(
            encoding="utf-8"
        )
    )
    report = CriticReport(
        is_physical=True,
        physics_score=0.8,
        confidence=0.7,
    ).to_dict()

    jsonschema.validate(report, schema)
    assert report["schema_version"] == "2.0"
    assert report["decision"] == "physical"


@pytest.mark.parametrize(
    "config",
    [
        CriticConfig(events=EventConfig(velocity_epsilon_px_s=0)),
        CriticConfig(events=EventConfig(contact_tolerance_px=-1)),
        CriticConfig(events=EventConfig(penetration_tolerance_px=-1)),
        CriticConfig(events=EventConfig(teleport_speed_px_s=0)),
        CriticConfig(rules=RuleConfig(contact_lookback_frames=-1)),
        CriticConfig(rules=RuleConfig(gravity_contact_lookback_frames=-1)),
        CriticConfig(temporal=TemporalConfig(pre_context_frames=-1)),
        CriticConfig(temporal=TemporalConfig(post_context_frames=-1)),
    ],
)
def test_invalid_event_and_window_thresholds_are_rejected(config):
    with pytest.raises(ValueError):
        config.validate()
