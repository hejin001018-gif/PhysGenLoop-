import json

import pytest

from generators.wanphysics.v2.artifacts import (
    RunArtifacts,
    SampleStatus,
    pending_samples,
    rebuild_summary,
)


def test_manifest_is_create_only(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.create_run_manifest({"run_id": "r1"})
    with pytest.raises(FileExistsError):
        artifacts.create_run_manifest({"run_id": "r2"})


def test_attempt_isolation_and_authoritative_pointer(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    attempt_id, attempt_dir = artifacts.start_attempt("s1", reason="initial")
    artifacts.write_loop_result("s1", {"final_state": "ACCEPTED"})
    artifacts.set_status(SampleStatus("s1", "ACCEPTED"))
    raw = json.loads((tmp_path / "run" / "s1" / "sample_status.json").read_text())
    assert raw["authoritative_attempt"] == attempt_id
    assert (attempt_dir / "loop_result.json").exists()


def test_retry_failed_only_retries_failures(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    for sample_id, state in (("ok", "ACCEPTED"), ("bad", "EVALUATION_FAILED")):
        artifacts.start_attempt(sample_id, reason="initial")
        artifacts.set_status(SampleStatus(sample_id, state))
    assert pending_samples(artifacts.run_dir, ["ok", "bad"]) == []
    assert pending_samples(
        artifacts.run_dir, ["ok", "bad"], retry_failed=True
    ) == ["bad"]


def test_summary_reads_only_authoritative_attempt(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.start_attempt("s1", reason="initial")
    artifacts.write_loop_result("s1", {"final_state": "REJECTED"})
    artifacts.set_status(SampleStatus("s1", "REJECTED"))
    summary = rebuild_summary(artifacts.run_dir, ["s1"])
    assert summary["rejected"] == 1
    assert len(summary["results"]) == 1
