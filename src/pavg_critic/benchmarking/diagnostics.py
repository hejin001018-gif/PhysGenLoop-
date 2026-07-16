"""Auditable rule-level diagnostics for frozen benchmark observations."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from pavg_critic.evaluation import build_ablation_config
from pavg_critic.pipeline import PhysicsCritic
from pavg_critic.schemas import CriticRequest, FrameState, load_frame_states

from .contracts import BenchmarkSample


def build_sample_diagnostic(
    sample: BenchmarkSample,
    states: Sequence[FrameState],
    *,
    mode: str,
) -> dict:
    artifacts = PhysicsCritic(build_ablation_config(mode)).analyze_detailed(
        CriticRequest(video_path=sample.video_path, prompt=sample.prompt),
        observations=tuple(states),
    )
    represented_frames = sorted({state.frame for state in states})
    report = artifacts.report
    return {
        "schema_version": "1.0",
        "sample_id": sample.sample_id,
        "mode": mode,
        "prompt": sample.prompt,
        "prompt_group_id": sample.prompt_group_id,
        "generator": sample.generator,
        "gold_label": sample.physics_label,
        "gold_score": sample.physics_score,
        "prediction": report.decision,
        "physics_score": report.physics_score,
        "confidence": report.confidence,
        "coverage": report.coverage,
        "represented_frames": represented_frames,
        "observation_count": len(states),
        "track_count": len(artifacts.tracks),
        "tracks": [
            {
                "track_id": track.track_id,
                "object": track.object,
                "state_count": len(track.states),
                "start_frame": track.states[0].frame,
                "end_frame": track.states[-1].frame,
                "visible_count": sum(state.visible for state in track.states),
            }
            for track in artifacts.tracks
        ],
        "events": [asdict(event) for event in artifacts.events],
        "raw_candidates": [asdict(candidate) for candidate in artifacts.candidates],
        "fused_violations": [asdict(item) for item in report.violations],
        "score_breakdown": dict(report.score_breakdown),
        "physics_plan": (
            None
            if artifacts.resolved_request is None
            or artifacts.resolved_request.physics_plan is None
            else artifacts.resolved_request.physics_plan.to_dict()
        ),
    }


def write_diagnostics(
    samples: Sequence[BenchmarkSample],
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    mode: str,
) -> dict:
    cache = Path(cache_dir)
    destination = Path(output_dir)
    sample_dir = destination / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    failures = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        try:
            states = load_frame_states(cache / f"{sample.sample_id}.json")
            diagnostic = build_sample_diagnostic(sample, states, mode=mode)
            diagnostics.append(diagnostic)
            (sample_dir / f"{sample.sample_id}.json").write_text(
                json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": sample.sample_id,
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                }
            )

    raw_by_gold: dict[str, Counter[str]] = {}
    fused_by_gold: dict[str, Counter[str]] = {}
    for item in diagnostics:
        gold = item["gold_label"]
        raw_by_gold.setdefault(gold, Counter()).update(
            candidate["category"] for candidate in item["raw_candidates"]
        )
        fused_by_gold.setdefault(gold, Counter()).update(
            violation["category"] for violation in item["fused_violations"]
        )
    false_positives = [
        item
        for item in diagnostics
        if item["gold_label"] == "physical" and item["prediction"] == "violation"
    ]
    summary = {
        "schema_version": "1.0",
        "mode": mode,
        "sample_count": len(samples),
        "diagnosed_count": len(diagnostics),
        "failure_count": len(failures),
        "false_positive_count": len(false_positives),
        "raw_candidate_categories_by_gold": {
            gold: dict(sorted(counter.items()))
            for gold, counter in sorted(raw_by_gold.items())
        },
        "fused_violation_categories_by_gold": {
            gold: dict(sorted(counter.items()))
            for gold, counter in sorted(fused_by_gold.items())
        },
        "failures": failures,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "category_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# PAVG dev false positives", ""]
    if false_positives:
        for item in false_positives:
            categories = sorted(
                {violation["category"] for violation in item["fused_violations"]}
            )
            lines.append(
                f"- `{item['sample_id']}`: categories={categories}; "
                f"tracks={item['track_count']}; frames={len(item['represented_frames'])}"
            )
    else:
        lines.append("- None")
    (destination / "false_positives.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary
