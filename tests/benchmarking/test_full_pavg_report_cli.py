"""Atomic, deterministic complete-PAVG report publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from benchmarks.report_full_pavg_critic import CORE_REPORT_FILES, main
from pavg_critic.benchmarking.datasets import write_manifest
from pavg_critic.benchmarking.full_pavg_report import (
    FULL_METHODS,
    PAVG_DIAGNOSTIC_METHODS,
)


def _write_jsonl(path, records):
    path.write_text(
        "".join(
            json.dumps(
                item.to_dict() if hasattr(item, "to_dict") else item,
                ensure_ascii=False,
            )
            + "\n"
            for item in records
        ),
        encoding="utf-8",
    )


def _hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
    }


def test_full_pavg_report_bundle_is_complete_and_byte_stable(
    tmp_path, sample_factory, prediction_factory
):
    samples = tuple(
        sample_factory(index=index, physical=index < 2, generator="g")
        for index in range(4)
    )
    manifest = tmp_path / "manifest.json"
    pilot = tmp_path / "pilot.json"
    write_manifest(samples, manifest)
    write_manifest(samples, pilot)
    labels = ("physical", "physical", "violation", "violation")
    predictions = tuple(
        prediction_factory(
            sample.sample_id,
            labels[index],
            5.0 if labels[index] == "physical" else 2.0,
            method_id=method,
        )
        for method in FULL_METHODS
        for index, sample in enumerate(samples)
    )
    prediction_path = tmp_path / "predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    diagnostics = tuple(
        {
            "schema_version": "1.0",
            "key": {"sample_id": sample.sample_id, "method_id": method},
            "planner": {"source": "model", "fallback_used": False},
            "question_graph": {"source": "graph", "node_count": 1, "physics_coverage": 1.0},
            "video_science": {"enabled": True, "coverage": 1.0},
            "mechanics": {"enabled": True, "coverage": 1.0},
            "rules": {"candidate_count": 0, "retained_violation_count": 0},
            "vlm_reviews": {"status_counts": {"confirmed": 0, "rejected": 0, "uncertain": 0, "unavailable": 0}},
            "evidence_families": {
                family: {"status": "available", "coverage": 1.0}
                for family in ("rules", "pqsg", "checklist", "mechanics", "vlm")
            },
            "fusion": {"pre_evidence_fusion": {"decision": labels[index]}, "final": {"decision": labels[index]}},
            "hard_violation_override": False,
            "model_calls": {},
            "latency": {"analysis_sec": 0.1, "total_sec": 0.1, "visible_frame_count": 4},
            "provider_failures": [],
            "failure": None,
        }
        for method in PAVG_DIAGNOSTIC_METHODS
        for index, sample in enumerate(samples)
    )
    diagnostics_path = tmp_path / "diagnostics.jsonl"
    _write_jsonl(diagnostics_path, diagnostics)
    correct = tuple(item for item in predictions if item.method_id == "M5_FULL")
    prompt_predictions = correct + tuple(
        replace(item, method_id="M5_SHUFFLED_PROMPT_300") for item in correct
    ) + tuple(replace(item, method_id="M5_ORACLE_PLAN_300") for item in correct)
    prompt_path = tmp_path / "prompt-predictions.jsonl"
    _write_jsonl(prompt_path, prompt_predictions)
    observation_dir = tmp_path / "observations"
    observation_dir.mkdir()
    for sample in samples:
        (observation_dir / f"{sample.sample_id}.meta.json").write_text(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "production_latency_sec": 1.0,
                }
            ),
            encoding="utf-8",
        )

    def arguments(output):
        return [
            "--manifest", str(manifest),
            "--predictions", str(prediction_path),
            "--diagnostics", str(diagnostics_path),
            "--pilot-manifest", str(pilot),
            "--prompt-predictions", str(prompt_path),
            "--observation-meta-dir", str(observation_dir),
            "--output-dir", str(output),
            "--bootstrap-resamples", "20",
            "--bootstrap-seed", "20260717",
        ]

    first = tmp_path / "report-a"
    second = tmp_path / "report-b"
    assert main(arguments(first)) == 0
    assert set(path.name for path in first.iterdir()) == set(CORE_REPORT_FILES)
    assert main(arguments(first)) == 0
    assert main(arguments(second)) == 0
    assert _hashes(first) == _hashes(second)
