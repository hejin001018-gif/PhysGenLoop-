"""审计产物 + append-only + resume 扫描。"""
from generators.wanphysics.v2.artifacts import RunArtifacts, SampleStatus, pending_samples


def test_status_and_resume(tmp_path):
    art = RunArtifacts(tmp_path / "run")
    art.set_status(SampleStatus(sample_id="s1", state="CREATED"))
    art.set_status(SampleStatus(sample_id="s1", state="GENERATED"))
    assert not art.is_complete("s1")
    art.set_status(SampleStatus(sample_id="s1", state="ACCEPTED"))
    assert art.is_complete("s1")
    art.set_status(SampleStatus(sample_id="s2", state="CREATED"))
    assert pending_samples(tmp_path / "run", ["s1", "s2"]) == ["s2"]


def test_history_is_append_only(tmp_path):
    art = RunArtifacts(tmp_path / "run")
    for st in ("CREATED", "GENERATING", "GENERATED"):
        art.set_status(SampleStatus(sample_id="s1", state=st))
    hist = (art.run_dir / "s1" / "sample_status_history.jsonl").read_text().strip().splitlines()
    assert len(hist) == 3


def test_invalid_state_rejected():
    try:
        SampleStatus(sample_id="s1", state="NONSENSE")
        assert False
    except ValueError:
        pass


def test_critic_report_and_raw(tmp_path):
    art = RunArtifacts(tmp_path / "run")
    art.write_critic_report("s1", "c1", {"a": 1})
    art.write_raw_payload("s1", "c1", {"raw": True}, "boom")
    assert (art.run_dir / "s1" / "c1" / "critic_report.json").exists()
    assert (art.run_dir / "s1" / "c1" / "critic_roundtrip_error.json").exists()
