import json

from pavg_critic.benchmarking.diagnostics import (
    build_sample_diagnostic,
    write_diagnostics,
)


def test_diagnostic_contains_tracks_events_candidates_and_report(
    sample_factory, frame_state_factory
):
    sample = sample_factory(index=1, physical=False, generator="g")
    diagnostic = build_sample_diagnostic(
        sample,
        (frame_state_factory(),),
        mode="B1_RULE",
    )
    assert diagnostic["sample_id"] == sample.sample_id
    assert diagnostic["gold_label"] == "violation"
    assert diagnostic["track_count"] == 1
    assert diagnostic["represented_frames"] == [0]
    assert isinstance(diagnostic["events"], list)
    assert isinstance(diagnostic["raw_candidates"], list)
    assert diagnostic["prediction"] in {"physical", "violation", "unknown"}


def test_write_diagnostics_is_stable_and_summarizes_false_positives(
    tmp_path, sample_factory, frame_state_factory
):
    samples = (
        sample_factory(index=1, physical=True, generator="g"),
        sample_factory(index=2, physical=False, generator="g"),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    for sample in samples:
        (cache / f"{sample.sample_id}.json").write_text(
            json.dumps([frame_state_factory().to_dict()]),
            encoding="utf-8",
        )
    output = tmp_path / "diagnostics"
    write_diagnostics(samples, cache_dir=cache, output_dir=output, mode="B1_RULE")
    first = (output / "category_summary.json").read_bytes()
    write_diagnostics(samples, cache_dir=cache, output_dir=output, mode="B1_RULE")
    assert first == (output / "category_summary.json").read_bytes()
    summary = json.loads(first)
    assert summary["sample_count"] == 2
    assert (output / "false_positives.md").is_file()
