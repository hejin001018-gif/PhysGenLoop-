from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import benchmarks.report_full_video_benchmark as report_cli
from benchmarks.report_full_video_benchmark import build_parser, main
from pavg_critic.benchmarking.datasets import write_manifest


CORE_REPORT_FILES = (
    "artifact_audit.json",
    "merged_predictions.jsonl",
    "paired_outcomes.json",
    "slices.json",
    "summary.json",
    "summary.md",
)


def _write_jsonl(path: Path, records: tuple[object, ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    sample_factory,
    prediction_factory,
) -> tuple[Path, tuple[Path, Path], tuple[Path, Path]]:
    samples = (
        sample_factory(index=1, physical=True, generator="生成器甲"),
        sample_factory(index=2, physical=False, generator="生成器甲"),
        sample_factory(index=3, physical=True, generator="生成器乙"),
        sample_factory(index=4, physical=False, generator="生成器乙"),
    )
    manifest = tmp_path / "manifest.json"
    write_manifest(samples, manifest)
    shard_a = tmp_path / "shard-a.jsonl"
    shard_b = tmp_path / "shard-b.jsonl"
    _write_jsonl(
        shard_a,
        (
            prediction_factory("1", "physical", 5.0, method_id="D0_DIRECT_VLM"),
            prediction_factory("1", "physical", 5.0, method_id="B1_RULE"),
            prediction_factory("2", "physical", 5.0, method_id="D0_DIRECT_VLM"),
            prediction_factory("2", "violation", 2.0, method_id="B1_RULE"),
        ),
    )
    candidate_failure = replace(
        prediction_factory("3", "unknown", None, method_id="B1_RULE"),
        failure={"type": "timeout"},
    )
    _write_jsonl(
        shard_b,
        (
            prediction_factory("3", "physical", 5.0, method_id="D0_DIRECT_VLM"),
            candidate_failure,
            prediction_factory("4", "physical", 5.0, method_id="D0_DIRECT_VLM"),
            prediction_factory("4", "violation", 2.0, method_id="B1_RULE"),
        ),
    )
    observation_a = tmp_path / "observation-a"
    observation_b = tmp_path / "observation-b"
    observation_a.mkdir()
    observation_b.mkdir()
    for sample_id, latency, directory in (
        ("1", 1.0, observation_a),
        ("2", 2.0, observation_a),
        ("3", 3.0, observation_b),
        ("4", 4.0, observation_b),
    ):
        (directory / f"{sample_id}.meta.json").write_text(
            json.dumps(
                {"sample_id": sample_id, "production_latency_sec": latency}
            ),
            encoding="utf-8",
        )
    return manifest, (shard_a, shard_b), (observation_a, observation_b)


def _arguments(
    manifest: Path,
    shards: tuple[Path, Path],
    observation_dirs: tuple[Path, Path],
    output_dir: Path,
) -> list[str]:
    return [
        "--manifest",
        str(manifest),
        "--predictions",
        str(shards[0]),
        "--predictions",
        str(shards[1]),
        "--observation-meta-dir",
        str(observation_dirs[0]),
        "--observation-meta-dir",
        str(observation_dirs[1]),
        "--output-dir",
        str(output_dir),
        "--bootstrap-resamples",
        "25",
    ]


def test_full_report_cli_parser_freezes_methods_and_statistical_defaults(tmp_path):
    parsed = build_parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "a.jsonl"),
            "--predictions",
            str(tmp_path / "b.jsonl"),
            "--observation-meta-dir",
            str(tmp_path / "meta-a"),
            "--observation-meta-dir",
            str(tmp_path / "meta-b"),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )

    assert parsed.bootstrap_resamples == 2000
    assert parsed.bootstrap_seed == 20260717
    assert parsed.predictions == [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    assert parsed.observation_meta_dir == [tmp_path / "meta-a", tmp_path / "meta-b"]
    assert report_cli.BASELINE_METHOD == "D0_DIRECT_VLM"
    assert report_cli.CANDIDATE_METHOD == "B1_RULE"

    with pytest.raises(SystemExit) as missing_required:
        build_parser().parse_args([])
    assert missing_required.value.code == 2


def test_full_report_cli_generates_deterministic_auditable_chinese_report(
    tmp_path,
    sample_factory,
    prediction_factory,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = tmp_path / "report"
    arguments = _arguments(manifest, shards, observation_dirs, output_dir)

    assert main(arguments) == 0

    assert tuple(sorted(path.name for path in output_dir.iterdir())) == CORE_REPORT_FILES
    first_hashes = {
        name: _sha256(output_dir / name) for name in CORE_REPORT_FILES
    }
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["population"] == {
        "prediction_count": 8,
        "sample_count": 4,
    }
    assert summary["method_ids"] == {
        "baseline": "D0_DIRECT_VLM",
        "candidate": "B1_RULE",
    }
    assert set(summary["method_metrics"]) == {"D0_DIRECT_VLM", "B1_RULE"}
    assert summary["bootstrap"]["resamples"] == 25
    assert summary["bootstrap"]["seed"] == 20260717
    assert summary["paired_outcomes"] == {
        "baseline_only_correct": 1,
        "both_correct": 1,
        "both_wrong": 0,
        "candidate_only_correct": 2,
    }
    assert summary["prediction_failures"]["count"] == 1
    assert summary["prediction_failures"]["records"] == [
        {"method_id": "B1_RULE", "reason": "timeout", "sample_id": "3"}
    ]
    assert summary["prediction_latency"]["scope"] == "model_or_rule_prediction"
    assert summary["sam2_production_latency"]["scope"] == "sam2_production"
    assert summary["sam2_production_latency"]["valid_count"] == 4
    assert summary["material_decision"]["overall_verdict"] == (
        "not_evaluable_ood_deferred"
    )
    assert summary["ood_evaluation"] == {
        "benchmark": "VideoPhy-1",
        "overall_verdict": "not_evaluable_ood_deferred",
        "status": "deferred",
    }
    assert summary["artifacts"]["core_file_count"] == 6

    paired = json.loads(
        (output_dir / "paired_outcomes.json").read_text(encoding="utf-8")
    )
    slices = json.loads((output_dir / "slices.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (output_dir / "artifact_audit.json").read_text(encoding="utf-8")
    )
    assert paired == summary["paired_outcomes"]
    assert set(slices) == {"action", "generator", "rule_family"}
    assert audit["expected_count"] == 8
    assert audit["merged_count"] == 8
    assert audit["missing_count"] == audit["duplicate_count"] == 0
    assert audit["manifest"] == {
        "name": manifest.name,
        "path": str(manifest),
        "sha256": _sha256(manifest),
    }
    assert {
        entry["path"]: entry["sha256"] for entry in audit["inputs"]
    } == {str(path): _sha256(path) for path in shards}
    assert audit["observation_metadata"]["directory_count"] == 2
    assert audit["observation_metadata"]["file_count"] == 4
    assert {
        entry["sha256"] for entry in audit["observation_metadata"]["files"]
    } == {
        _sha256(path)
        for directory in observation_dirs
        for path in directory.glob("*.meta.json")
    }
    assert set(audit["report_output_sha256"]) == {
        "merged_predictions.jsonl",
        "paired_outcomes.json",
        "slices.json",
        "summary.json",
        "summary.md",
    }
    assert audit["report_output_sha256_scope"] == {
        "excluded": ["artifact_audit.json"],
        "hashed_count": 5,
        "reason": "artifact_audit.json is self-referential",
    }

    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "预测延迟（模型/规则）" in markdown
    assert "SAM2 production latency" in markdown
    assert "VideoPhy-1 OOD：deferred" in markdown
    assert "overall: not_evaluable_ood_deferred" in markdown
    assert "不能据此声称架构已被证明" in markdown
    assert "timeout" in markdown
    assert "20260717" in markdown and "25" in markdown

    assert main(arguments) == 0
    assert {
        name: _sha256(output_dir / name) for name in CORE_REPORT_FILES
    } == first_hashes


def _staging_directories(output_dir: Path) -> list[Path]:
    return list(output_dir.parent.glob(f".{output_dir.name}.full-report-*"))


def test_single_atomic_directory_publish_failure_leaves_no_output(
    tmp_path,
    sample_factory,
    prediction_factory,
    monkeypatch,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = tmp_path / "report"
    arguments = _arguments(manifest, shards, observation_dirs, output_dir)

    def fail_publish(staged: Path, destination: Path) -> None:
        raise OSError("injected atomic directory publish failure")

    monkeypatch.setattr(report_cli, "_publish_staged_directory", fail_publish)

    with pytest.raises(OSError, match="injected atomic directory publish failure"):
        main(arguments)

    assert not output_dir.exists()
    assert not _staging_directories(output_dir)


def test_identical_existing_bundle_is_deterministic_noop(
    tmp_path,
    sample_factory,
    prediction_factory,
    monkeypatch,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = tmp_path / "report"
    arguments = _arguments(manifest, shards, observation_dirs, output_dir)
    assert main(arguments) == 0
    originals = {
        name: (output_dir / name).read_bytes() for name in CORE_REPORT_FILES
    }

    def unexpected_publish(staged: Path, destination: Path) -> None:
        raise AssertionError("identical rerun must not publish another directory")

    monkeypatch.setattr(
        report_cli,
        "_publish_staged_directory",
        unexpected_publish,
    )

    assert main(arguments) == 0
    assert {
        name: (output_dir / name).read_bytes() for name in CORE_REPORT_FILES
    } == originals
    assert not _staging_directories(output_dir)


@pytest.mark.parametrize("existing_state", ("different", "partial"))
def test_different_or_partial_existing_bundle_fails_closed_unchanged(
    tmp_path,
    sample_factory,
    prediction_factory,
    existing_state,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = tmp_path / "report"
    arguments = _arguments(manifest, shards, observation_dirs, output_dir)
    assert main(arguments) == 0
    if existing_state == "different":
        (output_dir / "summary.md").write_bytes(b"different accepted report\n")
    else:
        (output_dir / "summary.md").unlink()
    existing = {
        path.name: path.read_bytes() for path in output_dir.iterdir()
    }

    with pytest.raises(ValueError, match="already exists.*new output path"):
        main(arguments)

    assert {
        path.name: path.read_bytes() for path in output_dir.iterdir()
    } == existing
    assert not _staging_directories(output_dir)


def test_staging_write_failure_leaves_no_output_or_temporary_directory(
    tmp_path,
    sample_factory,
    prediction_factory,
    monkeypatch,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = tmp_path / "report"
    arguments = _arguments(manifest, shards, observation_dirs, output_dir)
    calls = 0
    real_write = report_cli._write_staged_artifact

    def fail_third_write(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected stage write failure")
        real_write(path, content)

    monkeypatch.setattr(report_cli, "_write_staged_artifact", fail_third_write)

    with pytest.raises(OSError, match="injected stage write failure"):
        main(arguments)

    assert not output_dir.exists()
    assert not _staging_directories(output_dir)


def test_invalid_cli_input_exits_nonzero_without_publishing_partial_report(
    tmp_path,
    sample_factory,
    prediction_factory,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = tmp_path / "report"
    arguments = _arguments(manifest, shards, observation_dirs, output_dir)
    assert main(arguments) == 0
    originals = {
        name: (output_dir / name).read_bytes() for name in CORE_REPORT_FILES
    }
    shards[1].write_text(
        "\n".join(shards[1].read_text(encoding="utf-8").splitlines()[:-1])
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.report_full_video_benchmark",
            *arguments,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "missing 1 expected prediction key" in completed.stderr
    assert {
        name: (output_dir / name).read_bytes() for name in CORE_REPORT_FILES
    } == originals
    assert not list(output_dir.glob(".*.tmp"))


def test_cli_rejects_output_aliasing_manifest_before_overwrite(
    tmp_path,
    sample_factory,
    prediction_factory,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = manifest.parent
    aliased_manifest = output_dir / "summary.json"
    manifest.replace(aliased_manifest)
    original = aliased_manifest.read_bytes()

    with pytest.raises(ValueError, match="destination aliases an input"):
        main(_arguments(aliased_manifest, shards, observation_dirs, output_dir))

    assert aliased_manifest.read_bytes() == original
    assert not (output_dir / "artifact_audit.json").exists()
    assert not (output_dir / "merged_predictions.jsonl").exists()


@pytest.mark.parametrize(
    "mutation",
    ("manifest", "prediction", "meta_content", "meta_added", "meta_deleted"),
)
def test_cli_rejects_any_input_snapshot_change_before_publication(
    tmp_path,
    sample_factory,
    prediction_factory,
    monkeypatch,
    mutation,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = tmp_path / "report"
    arguments = _arguments(manifest, shards, observation_dirs, output_dir)
    real_build = report_cli._build_report_artifacts

    def build_then_mutate(*args, **kwargs):
        artifacts = real_build(*args, **kwargs)
        if mutation == "manifest":
            manifest.write_bytes(manifest.read_bytes() + b" \n")
        elif mutation == "prediction":
            shards[0].write_bytes(shards[0].read_bytes() + b"\n")
        elif mutation == "meta_content":
            target = observation_dirs[0] / "1.meta.json"
            target.write_text(
                json.dumps({"sample_id": "1", "production_latency_sec": 99.0}),
                encoding="utf-8",
            )
        elif mutation == "meta_added":
            (observation_dirs[0] / "extra.meta.json").write_text(
                json.dumps(
                    {"sample_id": "1", "production_latency_sec": 1.0}
                ),
                encoding="utf-8",
            )
        else:
            (observation_dirs[0] / "1.meta.json").unlink()
        return artifacts

    monkeypatch.setattr(report_cli, "_build_report_artifacts", build_then_mutate)

    with pytest.raises(ValueError, match="input snapshot changed"):
        main(arguments)

    assert not output_dir.exists()
    assert not _staging_directories(output_dir)


@pytest.mark.parametrize(
    "invalid_failure",
    ("provider failure", ["provider failure"], {"reason": float("nan")}),
)
def test_cli_rejects_non_object_or_non_json_failure_with_source_line(
    tmp_path,
    sample_factory,
    prediction_factory,
    invalid_failure,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    output_dir = tmp_path / "report"
    records = [
        json.loads(line)
        for line in shards[1].read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        if record["sample_id"] == "3" and record["method_id"] == "B1_RULE":
            record["failure"] = invalid_failure
    shards[1].write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"(?s)(?=.*failure)(?=.*shard-b\.jsonl)(?=.*line)",
    ):
        main(_arguments(manifest, shards, observation_dirs, output_dir))

    assert not output_dir.exists()


def test_non_string_failure_reason_is_deterministic_json(
    tmp_path,
    sample_factory,
    prediction_factory,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    records = [
        json.loads(line)
        for line in shards[1].read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        if record["sample_id"] == "3" and record["method_id"] == "B1_RULE":
            record["failure"] = {"reason": {"code": 504, "retry": False}}
    shards[1].write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"

    assert main(_arguments(manifest, shards, observation_dirs, output_dir)) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["prediction_failures"]["records"][0]["reason"] == (
        '{"code":504,"retry":false}'
    )


def test_markdown_text_escapes_control_markdown_and_html_characters():
    raw = "<script>|`code`\n- injected\t&"

    escaped = report_cli._markdown_text(raw)

    assert escaped == (
        "&lt;script&gt;&#124;&#96;code&#96;\\n&#45; injected\\t&amp;"
    )
    assert "<script>" not in escaped
    assert "\n- injected" not in escaped


def test_failure_reason_is_escaped_in_markdown_but_preserved_in_json(
    tmp_path,
    sample_factory,
    prediction_factory,
):
    manifest, shards, observation_dirs = _fixture(
        tmp_path, sample_factory, prediction_factory
    )
    malicious = "<script>|`code`\n- injected"
    records = [
        json.loads(line)
        for line in shards[1].read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        if record["sample_id"] == "3" and record["method_id"] == "B1_RULE":
            record["failure"] = {"reason": malicious}
    shards[1].write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"

    assert main(_arguments(manifest, shards, observation_dirs, output_dir)) == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert summary["prediction_failures"]["records"][0]["reason"] == malicious
    failure_section = markdown.split("## 失败记录", 1)[1].split("## 冻结门槛", 1)[0]
    assert "<script>" not in failure_section
    assert "\n- injected" not in failure_section
    assert "&lt;script&gt;&#124;&#96;code&#96;\\n&#45; injected" in failure_section
