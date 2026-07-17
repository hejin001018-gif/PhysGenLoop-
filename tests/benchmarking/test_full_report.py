from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import pavg_critic.benchmarking.full_report as full_report
from pavg_critic.benchmarking.full_report import (
    action_group_bootstrap,
    build_slices,
    merge_prediction_shards,
    paired_outcomes,
    parse_rule_families,
)


METHODS = ("D0_DIRECT_VLM", "B1_RULE")


def _write_jsonl(path: Path, records: tuple[object, ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                record.to_dict() if hasattr(record, "to_dict") else record,
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_disjoint_shards_merge_in_stable_sample_and_method_order_with_audit(
    tmp_path, sample_factory, prediction_factory
):
    samples = (
        sample_factory(index=2, physical=True, generator="g"),
        sample_factory(index=1, physical=False, generator="g"),
    )
    shard_a = tmp_path / "shard-a.jsonl"
    shard_b = tmp_path / "shard-b.jsonl"
    _write_jsonl(
        shard_a,
        (
            prediction_factory("2", "physical", 5.0, method_id="B1_RULE"),
            prediction_factory("2", "physical", 5.0, method_id="D0_DIRECT_VLM"),
        ),
    )
    _write_jsonl(
        shard_b,
        (
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
            replace(
                prediction_factory(
                    "1", "unknown", None, method_id="D0_DIRECT_VLM"
                ),
                failure={"type": "provider_error"},
            ),
        ),
    )
    merged_path = tmp_path / "merged.jsonl"

    result = merge_prediction_shards(
        samples,
        (shard_b, shard_a),
        METHODS,
        merged_path,
    )

    assert [
        (prediction.sample_id, prediction.method_id)
        for prediction in result.predictions
    ] == [
        ("1", "D0_DIRECT_VLM"),
        ("1", "B1_RULE"),
        ("2", "D0_DIRECT_VLM"),
        ("2", "B1_RULE"),
    ]
    written_keys = [
        (record["sample_id"], record["method_id"])
        for record in map(
            json.loads, merged_path.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert written_keys == [
        ("1", "D0_DIRECT_VLM"),
        ("1", "B1_RULE"),
        ("2", "D0_DIRECT_VLM"),
        ("2", "B1_RULE"),
    ]

    audit = result.artifact_audit
    assert audit["methods"] == list(METHODS)
    assert audit["expected_count"] == 4
    assert audit["merged_count"] == 4
    assert audit["duplicate_count"] == 0
    assert audit["extra_count"] == 0
    assert audit["missing_count"] == 0
    assert audit["merged_output_sha256"] == _sha256(merged_path)
    assert audit["merged_output_line_count"] == 4
    inputs = {entry["name"]: entry for entry in audit["inputs"]}
    assert inputs["shard-a.jsonl"] == {
        "path": str(shard_a),
        "name": "shard-a.jsonl",
        "sha256": _sha256(shard_a),
        "line_count": 2,
        "method_counts": {"D0_DIRECT_VLM": 1, "B1_RULE": 1},
        "terminal_count": 2,
        "failure_count": 0,
    }
    assert inputs["shard-b.jsonl"]["failure_count"] == 1
    assert inputs["shard-b.jsonl"]["terminal_count"] == 2

    audit_path = tmp_path / "artifact_audit.json"
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit
    first_merged_bytes = merged_path.read_bytes()
    first_audit_bytes = audit_path.read_bytes()
    second_result = merge_prediction_shards(
        tuple(reversed(samples)),
        (shard_a, shard_b),
        METHODS,
        merged_path,
    )
    assert merged_path.read_bytes() == first_merged_bytes
    assert audit_path.read_bytes() == first_audit_bytes
    assert second_result.artifact_audit == audit


@pytest.mark.parametrize("destination", ("output", "audit"))
def test_merge_rejects_destination_aliasing_an_input(
    tmp_path, sample_factory, prediction_factory, destination
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(
        shard,
        (
            prediction_factory(
                "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
            ),
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        ),
    )
    original = shard.read_bytes()
    output_path = tmp_path / "merged.jsonl"
    audit_path = tmp_path / "audit.json"
    if destination == "output":
        output_path = shard
    else:
        os.link(shard, audit_path)

    with pytest.raises(ValueError, match="destination.*prediction input"):
        merge_prediction_shards(
            samples,
            (shard,),
            METHODS,
            output_path,
            audit_path=audit_path,
        )

    assert shard.read_bytes() == original


def test_merge_rejects_audit_destination_aliasing_merged_output(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(
        shard,
        (
            prediction_factory(
                "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
            ),
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        ),
    )
    destination = tmp_path / "same.json"

    with pytest.raises(ValueError, match="must not alias"):
        merge_prediction_shards(
            samples,
            (shard,),
            METHODS,
            destination,
            audit_path=destination,
        )


def test_merge_rejects_duplicate_key_across_shards(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    duplicate = prediction_factory(
        "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
    )
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    _write_jsonl(shard_a, (duplicate,))
    _write_jsonl(
        shard_b,
        (
            duplicate,
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        ),
    )

    with pytest.raises(ValueError, match="duplicate prediction key"):
        merge_prediction_shards(
            samples, (shard_a, shard_b), METHODS, tmp_path / "merged.jsonl"
        )


def test_merge_rejects_missing_expected_key(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(
        shard,
        (
            prediction_factory(
                "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
            ),
        ),
    )

    with pytest.raises(ValueError, match="missing 1 expected prediction key"):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


def test_merge_rejects_unknown_sample_id(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(
        shard,
        (
            prediction_factory(
                "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
            ),
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
            prediction_factory(
                "999", "physical", 5.0, method_id="D0_DIRECT_VLM"
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown sample_id.*999"):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


def test_merge_rejects_unknown_method_id(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(
        shard,
        (
            prediction_factory("1", "violation", 2.0, method_id="OTHER"),
        ),
    )

    with pytest.raises(ValueError, match="unknown method_id.*OTHER"):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


@pytest.mark.parametrize(
    "record",
    (
        "{not-json}",
        json.dumps({"sample_id": "1", "method_id": "D0_DIRECT_VLM"}),
    ),
)
def test_merge_rejects_malformed_prediction_record(
    tmp_path, sample_factory, record
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    shard.write_text(record + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed prediction record.*line 1"):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("semantic_label", "invalid"),
        ("physics_label", "invalid"),
    ),
)
def test_merge_rejects_invalid_prediction_label_vocabulary(
    tmp_path, sample_factory, prediction_factory, field, value
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    record = prediction_factory(
        "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
    ).to_dict()
    record[field] = value
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(shard, (record,))

    with pytest.raises(ValueError, match=f"invalid {field}"):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("semantic_score", float("nan")),
        ("physics_score", float("inf")),
        ("latency_sec", float("-inf")),
    ),
)
def test_merge_rejects_non_finite_prediction_numbers(
    tmp_path, sample_factory, prediction_factory, field, value
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    record = prediction_factory(
        "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
    ).to_dict()
    record[field] = value
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(shard, (record,))

    with pytest.raises(ValueError, match=f"non-finite {field}"):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


def test_deterministic_writer_disallows_nan_outside_scored_fields(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    failed = replace(
        prediction_factory(
            "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
        ),
        failure={"provider_value": float("nan")},
    )
    _write_jsonl(
        shard,
        (
            failed,
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        ),
    )
    merged_path = tmp_path / "merged.jsonl"

    with pytest.raises(ValueError, match="non-finite JSON number"):
        merge_prediction_shards(samples, (shard,), METHODS, merged_path)

    assert not merged_path.exists()
    assert not (tmp_path / "artifact_audit.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("semantic_score", True),
        ("physics_score", False),
        ("confidence", True),
        ("coverage", False),
        ("latency_sec", True),
    ),
)
def test_merge_rejects_boolean_numeric_fields(
    tmp_path, sample_factory, prediction_factory, field, value
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    record = prediction_factory(
        "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
    ).to_dict()
    record[field] = value
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(shard, (record,))

    with pytest.raises(ValueError, match=f"invalid numeric {field}"):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


def test_merge_rejects_fractional_visible_frame_count(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    record = prediction_factory(
        "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
    ).to_dict()
    record["visible_frame_count"] = 1.5
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(shard, (record,))

    with pytest.raises(ValueError, match="invalid integer visible_frame_count"):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


def test_merge_wraps_integer_overflow_with_source_line_context(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    record = prediction_factory(
        "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
    ).to_dict()
    record["latency_sec"] = 10**1000
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(shard, (record,))

    with pytest.raises(
        ValueError,
        match="malformed prediction record.*shard.jsonl.*line 1",
    ):
        merge_prediction_shards(
            samples, (shard,), METHODS, tmp_path / "merged.jsonl"
        )


def test_unwritable_audit_parent_preserves_existing_merged_output(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(
        shard,
        (
            prediction_factory(
                "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
            ),
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        ),
    )
    merged_path = tmp_path / "merged.jsonl"
    original = b"previous-verified-output\n"
    merged_path.write_bytes(original)
    parent_blocker = tmp_path / "not-a-directory"
    parent_blocker.write_text("block directory creation", encoding="utf-8")

    with pytest.raises(OSError):
        merge_prediction_shards(
            samples,
            (shard,),
            METHODS,
            merged_path,
            audit_path=parent_blocker / "artifact_audit.json",
        )

    assert merged_path.read_bytes() == original
    assert not (parent_blocker / "artifact_audit.json").exists()


def test_second_stage_failure_cleans_temp_and_preserves_existing_artifacts(
    tmp_path, sample_factory, prediction_factory, monkeypatch
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(
        shard,
        (
            prediction_factory(
                "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
            ),
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        ),
    )
    merged_path = tmp_path / "merged.jsonl"
    audit_path = tmp_path / "artifact_audit.json"
    original_merged = b"previous-verified-output\n"
    original_audit = b'{"previous":"verified"}\n'
    merged_path.write_bytes(original_merged)
    audit_path.write_bytes(original_audit)
    original_stage = full_report._stage_bytes
    stage_count = 0

    def fail_second_stage(path, content):
        nonlocal stage_count
        stage_count += 1
        if stage_count == 2:
            raise OSError("injected second-stage failure")
        return original_stage(path, content)

    monkeypatch.setattr(full_report, "_stage_bytes", fail_second_stage)

    with pytest.raises(OSError, match="second-stage failure"):
        merge_prediction_shards(
            samples,
            (shard,),
            METHODS,
            merged_path,
            audit_path=audit_path,
        )

    assert merged_path.read_bytes() == original_merged
    assert audit_path.read_bytes() == original_audit
    assert not [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]


def test_second_replace_failure_rolls_back_existing_artifact_pair(
    tmp_path, sample_factory, prediction_factory, monkeypatch
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(
        shard,
        (
            prediction_factory(
                "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
            ),
            prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        ),
    )
    merged_path = tmp_path / "merged.jsonl"
    audit_path = tmp_path / "artifact_audit.json"
    original_merged = b"previous-verified-output\n"
    original_audit = b'{"previous":"verified"}\n'
    merged_path.write_bytes(original_merged)
    audit_path.write_bytes(original_audit)
    original_replace = full_report.os.replace
    replace_count = 0

    def fail_second_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("injected second-replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(full_report.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="second-replace failure"):
        merge_prediction_shards(
            samples,
            (shard,),
            METHODS,
            merged_path,
            audit_path=audit_path,
        )

    assert merged_path.read_bytes() == original_merged
    assert audit_path.read_bytes() == original_audit
    assert not [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]


def test_paired_outcomes_treats_unknown_and_failure_as_incorrect(
    tmp_path, sample_factory, prediction_factory
):
    samples = tuple(
        sample_factory(index=index, physical=index < 2, generator="g")
        for index in range(4)
    )
    baseline = [
        prediction_factory("0", "physical", 5.0),
        prediction_factory("1", "violation", 2.0),
        prediction_factory("2", "unknown", None),
        replace(
            prediction_factory("3", "violation", 2.0),
            failure={"type": "provider_error"},
        ),
    ]
    candidate = [
        prediction_factory("0", "physical", 5.0, method_id="B1_RULE"),
        prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        prediction_factory("2", "violation", 2.0, method_id="B1_RULE"),
        prediction_factory("3", "physical", 5.0, method_id="B1_RULE"),
    ]

    assert paired_outcomes(samples, baseline, candidate) == {
        "both_correct": 1,
        "baseline_only_correct": 0,
        "candidate_only_correct": 1,
        "both_wrong": 2,
    }


def test_action_group_bootstrap_is_deterministic_and_samples_groups(
    sample_factory, prediction_factory
):
    samples = (
        replace(
            sample_factory(index=0, physical=True, generator="g"),
            prompt_group_id="group-a",
        ),
        replace(
            sample_factory(index=1, physical=False, generator="g"),
            prompt_group_id="group-a",
        ),
        replace(
            sample_factory(index=2, physical=True, generator="g"),
            prompt_group_id="group-b",
        ),
        replace(
            sample_factory(index=3, physical=False, generator="g"),
            prompt_group_id="group-b",
        ),
    )
    baseline = [
        prediction_factory("0", "physical", 5.0),
        prediction_factory("1", "physical", 5.0),
        prediction_factory("2", "violation", 2.0),
        prediction_factory("3", "violation", 2.0),
    ]
    candidate = [
        prediction_factory("0", "physical", 5.0, method_id="B1_RULE"),
        prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
        prediction_factory("2", "physical", 5.0, method_id="B1_RULE"),
        prediction_factory("3", "violation", 2.0, method_id="B1_RULE"),
    ]

    first = action_group_bootstrap(
        samples, baseline, candidate, resamples=19, seed=123
    )
    second = action_group_bootstrap(
        tuple(reversed(samples)),
        list(reversed(baseline)),
        list(reversed(candidate)),
        resamples=19,
        seed=123,
    )
    assert first == second
    assert first["resamples"] == 19
    assert first["seed"] == 123
    assert first["group_count"] == 2
    assert first["point_estimate"] == pytest.approx(0.5)
    assert first["lower"] <= first["point_estimate"] <= first["upper"]


@pytest.mark.parametrize(
    ("raw_metadata", "expected"),
    (
        ("{'a': ' gravity ', 'b': ['contact', 'gravity']}", ("contact", "gravity")),
        ({"a": ("gravity", "contact"), "b": {"contact"}}, ("contact", "gravity")),
        ("not a mapping", ("__unmapped__",)),
        ("{'a': 7}", ("__unmapped__",)),
        (None, ("__unmapped__",)),
    ),
)
def test_parse_rule_families_is_strict_and_multilabel(
    sample_factory, raw_metadata, expected
):
    sample = replace(
        sample_factory(index=1, physical=False, generator="g"),
        raw_labels={"metadata_rules": raw_metadata},
    )
    assert parse_rule_families(sample) == expected


def test_build_slices_contains_generator_action_and_rule_family_metrics(
    sample_factory, prediction_factory
):
    samples = (
        replace(
            sample_factory(index=0, physical=True, generator="g1"),
            prompt_group_id="action-a",
            raw_labels={"metadata_rules": "{'r': 'gravity'}"},
        ),
        replace(
            sample_factory(index=1, physical=False, generator="g2"),
            prompt_group_id="action-b",
            raw_labels={"metadata_rules": "{'r': ['contact', 'gravity']}"},
        ),
    )
    baseline = [
        prediction_factory("0", "physical", 5.0),
        prediction_factory("1", "physical", 5.0),
    ]
    candidate = [
        prediction_factory("0", "physical", 5.0, method_id="B1_RULE"),
        prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
    ]

    slices = build_slices(samples, baseline, candidate)
    assert list(slices) == ["action", "generator", "rule_family"]
    assert set(slices["generator"]) == {"g1", "g2"}
    assert set(slices["action"]) == {"action-a", "action-b"}
    assert set(slices["rule_family"]) == {"contact", "gravity"}
    gravity = slices["rule_family"]["gravity"]
    assert gravity["count"] == 2
    assert gravity["candidate_minus_baseline"]["accuracy"] == pytest.approx(0.5)
    assert "macro_f1" in gravity["baseline_metrics"]
    assert "macro_f1" in gravity["candidate_metrics"]
