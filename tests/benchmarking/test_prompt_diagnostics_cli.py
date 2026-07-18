"""Freeze shuffled-prompt manifests without reading benchmark labels."""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmarks.build_prompt_diagnostics import build_prompt_diagnostics, main


def _manifest(path, *, labels):
    samples = []
    for index, (prompt, action) in enumerate(
        (
            ("A ball falls.", "fall"),
            ("A cup slides.", "slide"),
            ("A car stops.", "stop"),
            ("A block rotates.", "rotate"),
        )
    ):
        samples.append(
            {
                "sample_id": f"sample-{index}",
                "benchmark": "videophy2",
                "split": "pilot300",
                "prompt": prompt,
                "video_path": f"../../data/{index}.mp4",
                "prompt_group_id": action,
                "generator": "g",
                "semantic_label": "unknown",
                "physics_label": labels[index],
                "physical_rules": [f"rule-{index}"],
                "raw_labels": {"private": index},
            }
        )
    path.write_text(
        json.dumps({"schema_version": "1.0", "samples": samples}) + "\n",
        encoding="utf-8",
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prompt_derangement_is_deterministic_cross_action_and_label_blind(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _manifest(first, labels=("physical", "violation", "physical", "violation"))
    _manifest(second, labels=("violation", "physical", "violation", "physical"))
    out_a, map_a = tmp_path / "a.json", tmp_path / "a-map.json"
    out_b, map_b = tmp_path / "b.json", tmp_path / "b-map.json"

    build_prompt_diagnostics(first, out_a, map_a, seed=20260717)
    build_prompt_diagnostics(second, out_b, map_b, seed=20260717)

    donors_a = json.loads(map_a.read_text(encoding="utf-8"))
    donors_b = json.loads(map_b.read_text(encoding="utf-8"))
    assert donors_a == donors_b
    mappings = donors_a["mappings"]
    assert len(mappings) == 4
    assert len({item["donor_sample_id"] for item in mappings}) == 4
    assert all(item["recipient_sample_id"] != item["donor_sample_id"] for item in mappings)
    assert all(item["recipient_prompt_sha256"] != item["donor_prompt_sha256"] for item in mappings)
    assert all(item["recipient_action"] != item["donor_action"] for item in mappings)

    source = json.loads(first.read_text(encoding="utf-8"))["samples"]
    shuffled = json.loads(out_a.read_text(encoding="utf-8"))["samples"]
    source_by_id = {item["sample_id"]: item for item in source}
    for item in shuffled:
        original = source_by_id[item["sample_id"]]
        assert item["video_path"] == original["video_path"]
        assert item["physics_label"] == original["physics_label"]
        assert item["physical_rules"] == original["physical_rules"]
        assert item["prompt"] != original["prompt"]


def test_builder_is_byte_stable_and_refuses_different_overwrite(tmp_path):
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, labels=("physical",) * 4)
    output, donor_map = tmp_path / "output.json", tmp_path / "map.json"

    assert main(
        [
            "--manifest",
            str(manifest),
            "--output-manifest",
            str(output),
            "--donor-map",
            str(donor_map),
            "--seed",
            "20260717",
        ]
    ) == 0
    hashes = (_sha(output), _sha(donor_map))
    assert main(
        [
            "--manifest",
            str(manifest),
            "--output-manifest",
            str(output),
            "--donor-map",
            str(donor_map),
            "--seed",
            "20260717",
        ]
    ) == 0
    assert (_sha(output), _sha(donor_map)) == hashes

    donor_map.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different existing file"):
        build_prompt_diagnostics(manifest, output, donor_map, seed=20260717)
