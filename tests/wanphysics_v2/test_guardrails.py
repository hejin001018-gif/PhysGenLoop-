"""接受门 shadow/enforce + scorer unavailable。"""
from generators.wanphysics.v2.guardrails import evaluate_gate, GateThresholds, NoOpSemanticScorer, MODE_SHADOW, MODE_ENFORCE


class _Report:
    def __init__(self, decision, physics, conf=0.7, cov=0.8):
        self.decision = decision; self.physics_score = physics
        self.confidence = conf; self.coverage = cov; self.diagnostics = {}


def test_shadow_uses_legacy_only():
    r = _Report("physical", 0.85)
    g = evaluate_gate(report=r, mode=MODE_SHADOW, thresholds=GateThresholds())
    assert g.accepted
    # semantic/quality 缺失只记录，不改判
    assert "semantic_score" in g.unavailable


def test_shadow_rejects_below():
    g = evaluate_gate(report=_Report("violation", 0.2), mode=MODE_SHADOW, thresholds=GateThresholds())
    assert not g.accepted


def test_enforce_needs_all_gates():
    r = _Report("physical", 0.85)
    g = evaluate_gate(report=r, mode=MODE_ENFORCE, thresholds=GateThresholds(), semantic_score=None, quality_score=0.9)
    assert not g.accepted
    assert "semantic_score" in g.unavailable  # 不伪装 accepted


def test_enforce_all_pass():
    r = _Report("physical", 0.85)
    g = evaluate_gate(report=r, mode=MODE_ENFORCE, thresholds=GateThresholds(), semantic_score=0.9, quality_score=0.9)
    assert g.accepted


def test_semantic_scorer_noop_returns_none():
    assert NoOpSemanticScorer().score(prompt="p", video_path="v") is None
