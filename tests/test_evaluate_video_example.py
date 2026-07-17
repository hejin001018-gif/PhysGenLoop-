"""Safety behavior for the end-to-end example's sparse VLM fallback."""

from pathlib import Path

from examples.evaluate_video import (
    _configure_sparse_vlm_fallback,
    _resolve_sam2_checkpoint,
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
