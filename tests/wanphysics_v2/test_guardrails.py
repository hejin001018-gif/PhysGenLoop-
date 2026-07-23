import pytest

from generators.wanphysics.v2.guardrails import (
    GateThresholds,
    MODE_ENFORCE,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNAVAILABLE,
    evaluate_gate,
)


class _Report:
    def __init__(self, decision="physical", score=0.9, violations=(), degraded=False):
        self.decision = decision
        self.physics_score = score
        self.confidence = 0.8
        self.coverage = 0.8
        self.violations = violations
        self.evidence_bundles = ()
        self.diagnostics = {"degraded": degraded}


def _gate(report=None, **overrides):
    values = {
        "semantic_score": 0.9,
        "original_prompt_semantic_score": 0.9,
        "quality_score": 0.9,
    }
    values.update(overrides)
    return evaluate_gate(
        report=report or _Report(),
        mode=MODE_ENFORCE,
        thresholds=GateThresholds(),
        fail_on_degraded=True,
        **values,
    )


def test_strict_gate_accepts_only_complete_evidence():
    result = _gate()
    assert result.status == STATUS_ACCEPTED and result.accepted


def test_strict_gate_rejects_physical_violation_or_low_score():
    assert _gate(_Report(violations=(object(),))).status == STATUS_REJECTED
    assert _gate(_Report(score=0.2)).status == STATUS_REJECTED


def test_strict_gate_unavailable_is_not_rejected():
    result = _gate(semantic_score=None)
    assert result.status == STATUS_UNAVAILABLE and not result.accepted
    assert "semantic_score" in result.unavailable


def test_original_prompt_semantic_is_required():
    assert _gate(original_prompt_semantic_score=None).status == STATUS_UNAVAILABLE
    assert _gate(original_prompt_semantic_score=0.2).status == STATUS_REJECTED


def test_shadow_is_rejected_by_active_gate():
    with pytest.raises(ValueError, match="requires acceptance.mode=enforce"):
        evaluate_gate(
            report=_Report(),
            mode="shadow",
            thresholds=GateThresholds(),
        )
