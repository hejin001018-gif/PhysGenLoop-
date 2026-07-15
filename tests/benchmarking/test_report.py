from pavg_critic.benchmarking.report import build_smoke_report, write_smoke_report


def test_report_separates_methods_and_contains_claims_warning(
    sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    predictions = (
        prediction_factory(
            "1", "violation", 2.0, method_id="D0_DIRECT_VLM"
        ),
        prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
    )
    report = build_smoke_report(samples, predictions)
    assert set(report["methods"]) == {"D0_DIRECT_VLM", "B1_RULE"}
    assert report["claims_allowed"] is False
    assert "smoke" in report["warning"].lower()


def test_report_output_is_stable_and_warning_precedes_table(
    tmp_path, sample_factory, prediction_factory
):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    predictions = (
        prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
    )
    write_smoke_report(samples, predictions, tmp_path)
    first = (tmp_path / "summary.md").read_bytes()
    write_smoke_report(samples, predictions, tmp_path)
    second = (tmp_path / "summary.md").read_bytes()
    assert first == second
    text = first.decode("utf-8")
    assert text.index("Warning") < text.index("| Method")
