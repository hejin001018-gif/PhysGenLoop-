"""Safety behavior for the end-to-end example's sparse VLM fallback."""

from examples.evaluate_video import _configure_sparse_vlm_fallback
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
