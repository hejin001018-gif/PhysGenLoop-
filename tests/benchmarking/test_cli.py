import pytest

from benchmarks.evaluate_video_benchmark import (
    build_benchmark_model,
    build_parser,
    load_benchmark_environment,
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


def test_method_parser_accepts_grouped_m4():
    assert parse_methods("B1_RULE,M4_VLM") == ("B1_RULE", "M4_VLM")


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


def test_env_file_maps_project_names_without_returning_secrets(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        'API_KEY="test-secret"\nBASE_URL="https://example.test/v1"\n'
        'VLM_MODEL="test-model"\n',
        encoding="utf-8",
    )
    for name in ("BENCH_API_KEY", "BENCH_BASE_URL", "BENCH_MODEL"):
        monkeypatch.delenv(name, raising=False)
    snapshot = load_benchmark_environment(path)
    assert snapshot == {
        "api_key_configured": True,
        "base_url_configured": True,
        "model": "test-model",
    }
    assert "test-secret" not in repr(snapshot)
