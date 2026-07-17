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
    evaluate_material_improvement,
    merge_prediction_shards,
    paired_outcomes,
    parse_rule_families,
    summarize_observation_latencies,
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


def test_action_group_bootstrap_freezes_asymmetric_cluster_ci(
    sample_factory, prediction_factory
):
    """The frozen seed exposes group (not row) resampling and multiplicity."""

    samples = []
    group_sizes = (("group-a", 1), ("group-b", 2), ("group-c", 4))
    index = 0
    for group_id, size in group_sizes:
        for position in range(size):
            sample = sample_factory(
                index=index,
                physical=position % 2 == 0,
                generator="g",
            )
            samples.append(replace(sample, prompt_group_id=group_id))
            index += 1

    baseline = []
    candidate = []
    for sample in samples:
        # Group A is baseline-correct/candidate-wrong; group B swaps its
        # two labels; group C is baseline-correct/candidate-all-physical.
        baseline_label = (
            "physical" if sample.prompt_group_id == "group-b" else sample.physics_label
        )
        candidate_label = (
            "violation"
            if sample.prompt_group_id in {"group-a", "group-b"}
            else "physical"
        )
        baseline.append(
            prediction_factory(
                sample.sample_id,
                baseline_label,
                5.0 if baseline_label == "physical" else 2.0,
            )
        )
        candidate.append(
            prediction_factory(
                sample.sample_id,
                candidate_label,
                5.0 if candidate_label == "physical" else 2.0,
                method_id="B1_RULE",
            )
        )

    result = action_group_bootstrap(samples, baseline, candidate)

    # Hand calculation uses 3 sorted clusters of sizes 1/2/4; each draw
    # selects exactly 3 clusters with replacement and appends every row of
    # each selected cluster. The frozen 2,000-draw linear percentiles are
    # intentionally unlike the row-level bootstrap distribution.
    assert result["resamples"] == 2000
    assert result["seed"] == 20260717
    assert result["group_count"] == 3
    assert result["point_estimate"] == pytest.approx(-0.4277777777777779)
    assert result["lower"] == -0.75
    assert result["upper"] == 0.0


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


def _write_observation_metadata(path: Path, **overrides: object) -> None:
    payload = {
        "schema_version": "1.0",
        "sample_id": "0",
        "video_sha256": "fixture",
        "production_latency_sec": 1.0,
        "propagation_failure": None,
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_observation_latency_summary_recurses_and_reports_linear_percentiles(
    tmp_path, sample_factory
):
    samples = tuple(
        sample_factory(index=index, physical=True, generator="g")
        for index in range(5)
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_observation_metadata(
        first / "nested" / "0.meta.json",
        sample_id="0",
        production_latency_sec=1.0,
    )
    _write_observation_metadata(
        first / "1.meta.json",
        sample_id="1",
        production_latency_sec=2.0,
    )
    _write_observation_metadata(
        second / "deeper" / "2.meta.json",
        sample_id="2",
        production_latency_sec=4.0,
    )
    _write_observation_metadata(
        second / "3.meta.json",
        sample_id="3",
        production_latency_sec=8.0,
    )

    summary = summarize_observation_latencies(samples, (second, first))

    assert summary == {
        "expected_count": 5,
        "valid_count": 4,
        "missing_count": 1,
        "missing_sample_ids": ["4"],
        "mean_production_latency_sec": pytest.approx(3.75),
        "p50_production_latency_sec": pytest.approx(3.0),
        "p95_production_latency_sec": pytest.approx(7.4),
    }


def test_observation_latency_accepts_identical_duplicate_metadata(
    tmp_path, sample_factory
):
    samples = (sample_factory(index=0, physical=True, generator="g"),)
    payload = {
        "sample_id": "0",
        "schema_version": "1.0",
        "production_latency_sec": 2.5,
        "represented_frames": [0, 3],
    }
    for directory in (tmp_path / "a", tmp_path / "b"):
        _write_observation_metadata(directory / "0.meta.json", **payload)

    summary = summarize_observation_latencies(
        samples, (tmp_path / "b", tmp_path / "a")
    )

    assert summary["valid_count"] == 1
    assert summary["mean_production_latency_sec"] == pytest.approx(2.5)


def test_observation_latency_rejects_conflicting_duplicate_metadata(
    tmp_path, sample_factory
):
    samples = (sample_factory(index=0, physical=True, generator="g"),)
    _write_observation_metadata(
        tmp_path / "a" / "0.meta.json",
        sample_id="0",
        production_latency_sec=1.0,
    )
    _write_observation_metadata(
        tmp_path / "b" / "0.meta.json",
        sample_id="0",
        production_latency_sec=2.0,
    )

    with pytest.raises(ValueError, match="conflicting observation metadata.*0"):
        summarize_observation_latencies(
            samples, (tmp_path / "a", tmp_path / "b")
        )


def test_observation_latency_rejects_unknown_sample_id(tmp_path, sample_factory):
    samples = (sample_factory(index=0, physical=True, generator="g"),)
    _write_observation_metadata(
        tmp_path / "999.meta.json",
        sample_id="999",
        production_latency_sec=1.0,
    )

    with pytest.raises(ValueError, match="unknown observation sample_id.*999"):
        summarize_observation_latencies(samples, (tmp_path,))


@pytest.mark.parametrize(
    "latency",
    (-1.0, float("nan"), float("inf"), True),
)
def test_observation_latency_rejects_invalid_latency(
    tmp_path, sample_factory, latency
):
    samples = (sample_factory(index=0, physical=True, generator="g"),)
    _write_observation_metadata(
        tmp_path / "0.meta.json",
        sample_id="0",
        production_latency_sec=latency,
    )

    with pytest.raises(ValueError, match="invalid production_latency_sec"):
        summarize_observation_latencies(samples, (tmp_path,))


def test_observation_latency_empty_valid_set_reports_all_missing(
    tmp_path, sample_factory
):
    samples = tuple(
        sample_factory(index=index, physical=True, generator="g")
        for index in range(2)
    )
    _write_observation_metadata(
        tmp_path / "0.meta.json",
        sample_id="0",
        production_latency_sec=None,
    )

    assert summarize_observation_latencies(samples, (tmp_path,)) == {
        "expected_count": 2,
        "valid_count": 0,
        "missing_count": 2,
        "missing_sample_ids": ["0", "1"],
        "mean_production_latency_sec": None,
        "p50_production_latency_sec": None,
        "p95_production_latency_sec": None,
    }


def _passing_material_inputs() -> dict[str, object]:
    return {
        "baseline_metrics": {
            "macro_f1": 0.64,
            "physical_recall": 0.75,
            "violation_recall": 0.70,
        },
        "candidate_metrics": {
            "macro_f1": 0.70,
            "physical_recall": 0.80,
            "violation_recall": 0.82,
        },
        "bootstrap": {"lower": 0.01, "upper": 0.11},
        "slices": {
            "generator": {
                "g1": {"candidate_minus_baseline": {"macro_f1": 0.08}},
                "g2": {"candidate_minus_baseline": {"macro_f1": 0.02}},
                "g3": {"candidate_minus_baseline": {"macro_f1": 0.0}},
            }
        },
        "baseline_failure_rate": 0.02,
        "candidate_failure_rate": 0.025,
    }


def test_material_improvement_reports_explicit_passing_gates_and_defers_ood():
    result = evaluate_material_improvement(**_passing_material_inputs())

    assert result["gates"] == {
        "macro_f1_delta": {
            "value": pytest.approx(0.06),
            "threshold": 0.05,
            "operator": ">=",
            "pass": True,
        },
        "bootstrap_lower": {
            "value": 0.01,
            "threshold": 0.0,
            "operator": ">",
            "pass": True,
        },
        "candidate_nonzero_recalls": {
            "value": {
                "physical_recall": 0.80,
                "violation_recall": 0.82,
            },
            "threshold": {
                "physical_recall": 0.0,
                "violation_recall": 0.0,
            },
            "operator": ">",
            "pass": True,
        },
        "failure_rate_increase": {
            "value": pytest.approx(0.005),
            "threshold": 0.01,
            "operator": "<=",
            "pass": True,
        },
        "positive_generator_count": {
            "value": 2,
            "threshold": 2,
            "operator": ">=",
            "pass": True,
        },
    }
    assert result["videophy2_support"] is True
    assert result["ood_status"] == "deferred"
    assert result["overall_verdict"] == "not_evaluable_ood_deferred"


def test_material_improvement_uses_exact_decimal_threshold_boundaries():
    inputs = _passing_material_inputs()
    inputs["baseline_metrics"]["macro_f1"] = 0.64
    inputs["candidate_metrics"]["macro_f1"] = 0.69
    inputs["baseline_failure_rate"] = 0.03
    inputs["candidate_failure_rate"] = 0.04

    result = evaluate_material_improvement(**inputs)

    assert result["gates"]["macro_f1_delta"]["value"] == 0.05
    assert result["gates"]["macro_f1_delta"]["pass"] is True
    assert result["gates"]["failure_rate_increase"]["value"] == 0.01
    assert result["gates"]["failure_rate_increase"]["pass"] is True
    assert result["videophy2_support"] is True


@pytest.mark.parametrize(
    ("gate", "updates"),
    (
        ("macro_f1_delta", {"candidate_macro_f1": 0.689999999}),
        ("bootstrap_lower", {"bootstrap_lower": 0.0}),
        ("candidate_nonzero_recalls", {"physical_recall": 0.0}),
        ("candidate_nonzero_recalls", {"violation_recall": 0.0}),
        ("failure_rate_increase", {"candidate_failure_rate": 0.030000001}),
        ("positive_generator_count", {"g2_delta": 0.0}),
    ),
)
def test_material_improvement_each_gate_can_fail_independently(gate, updates):
    inputs = _passing_material_inputs()
    candidate_metrics = inputs["candidate_metrics"]
    bootstrap = inputs["bootstrap"]
    generator_slices = inputs["slices"]["generator"]
    if "candidate_macro_f1" in updates:
        candidate_metrics["macro_f1"] = updates["candidate_macro_f1"]
    if "bootstrap_lower" in updates:
        bootstrap["lower"] = updates["bootstrap_lower"]
    if "physical_recall" in updates:
        candidate_metrics["physical_recall"] = updates["physical_recall"]
    if "violation_recall" in updates:
        candidate_metrics["violation_recall"] = updates["violation_recall"]
    if "candidate_failure_rate" in updates:
        inputs["candidate_failure_rate"] = updates["candidate_failure_rate"]
    if "g2_delta" in updates:
        generator_slices["g2"]["candidate_minus_baseline"]["macro_f1"] = updates[
            "g2_delta"
        ]

    result = evaluate_material_improvement(**inputs)

    assert result["gates"][gate]["pass"] is False
    assert sum(not item["pass"] for item in result["gates"].values()) == 1
    assert result["videophy2_support"] is False
    assert result["overall_verdict"] == "not_evaluable_ood_deferred"
