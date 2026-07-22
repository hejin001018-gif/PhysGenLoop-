"""P0-1/P0-2 核心：CriticReport 跨进程 round-trip 无损。"""
from pavg_critic.schemas import CriticReport, Violation, EvidenceBundle
from generators.wanphysics.v2.critic_codec import decode_report, encode_report


def _report_with_violation():
    v = Violation(
        object="baseball", category="penetration",
        start_frame=12, peak_frame=15, end_frame=18,
        critical_frames=(12, 13, 14, 15),
        reason="ball passes through wall",
        repair_instruction="keep the ball outside the wall",
        evidence={"mask_uri": "sam2_masks/baseball_00015.png", "mask_uris": ["sam2_masks/baseball_00012.png"]},
    )
    eb = EvidenceBundle(family="rules", source="rule-engine", status="available", score=0.1, confidence=0.9, coverage=0.8, critical_frames=(12, 15))
    return CriticReport(
        is_physical=False, physics_score=0.07, confidence=0.9,
        violations=(v,), decision="violation", coverage=0.83,
        diagnostics={"detector_backend": "sam2"}, evidence_bundles=(eb,),
    )


def test_roundtrip_preserves_violations_and_masks():
    report = _report_with_violation()
    result = decode_report(encode_report(report))
    assert result.ok
    assert len(result.report.violations) == 1
    v = result.report.violations[0]
    assert v.critical_frames == (12, 13, 14, 15)
    assert v.evidence["mask_uri"] == "sam2_masks/baseball_00015.png"
    assert len(result.report.evidence_bundles) == 1
    assert result.recovered_fields["violations"] == 1
    assert result.recovered_fields["mask_uris"] == 1


def test_roundtrip_failure_preserves_raw_and_flags():
    bad = {"violations": [{"object": "x"}], "physics_score": 0.1, "decision": "violation", "is_physical": False}
    result = decode_report(bad)
    assert not result.ok
    assert result.status == "roundtrip_failed"
    assert result.error is not None
    assert result.raw_payload == bad  # raw 保留，未静默丢


def test_physical_report_roundtrip():
    report = CriticReport(is_physical=True, physics_score=0.9, confidence=0.8, decision="physical")
    result = decode_report(encode_report(report))
    assert result.ok and result.report.decision == "physical"
