import pytest

from benchmarks.evaluate_video_benchmark import (
    build_benchmark_model,
    build_parser,
    parse_methods,
)


def test_method_parser_preserves_declared_order():
    assert parse_methods(
        "D0_DIRECT_VLM,D1_STRUCTURED_VLM,B1_RULE,M3_MECHANICS"
    ) == (
        "D0_DIRECT_VLM",
        "D1_STRUCTURED_VLM",
        "B1_RULE",
        "M3_MECHANICS",
    )


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown benchmark method"):
        parse_methods("D0_DIRECT_VLM,NOT_A_METHOD")


def test_duplicate_method_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_methods("B1_RULE,B1_RULE")


def test_missing_model_credentials_do_not_echo_secrets(monkeypatch):
    monkeypatch.delenv("BENCH_API_KEY", raising=False)
    monkeypatch.setenv("BENCH_MODEL", "gpt-test")
    with pytest.raises(ValueError) as error:
        build_benchmark_model("responses")
    assert "BENCH_API_KEY" in str(error.value)
    assert "gpt-test" not in str(error.value)


def test_max_samples_defaults_to_none():
    args = build_parser().parse_args(
        [
            "--manifest",
            "m.json",
            "--run-dir",
            "run",
            "--methods",
            "B1_RULE",
        ]
    )
    assert args.max_samples is None
