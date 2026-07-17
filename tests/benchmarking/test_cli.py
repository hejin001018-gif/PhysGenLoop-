from pathlib import Path

import pytest

from benchmarks.evaluate_video_benchmark import (
    build_benchmark_model,
    build_parser,
    load_benchmark_environment,
    parse_methods,
    sample_selection_sha256,
    validate_cache_only_observations,
    validate_manifest_binding,
    write_resolved_config,
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


def test_method_parser_accepts_full_and_oracle_m5():
    assert parse_methods("M5_FULL,M5_ORACLE_PLAN_300") == (
        "M5_FULL",
        "M5_ORACLE_PLAN_300",
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


def test_m4_detector_weight_defaults_to_detector_dominant_value():
    args = build_parser().parse_args(
        [
            "--manifest",
            "m.json",
            "--run-dir",
            "run",
            "--methods",
            "M4_VLM",
        ]
    )
    assert args.m4_detector_weight == 0.7


def test_model_cache_and_failure_budget_are_explicit_cli_inputs():
    args = build_parser().parse_args(
        [
            "--manifest",
            "m.json",
            "--run-dir",
            "run",
            "--methods",
            "M5_FULL",
            "--model-cache-dir",
            "cache",
            "--max-new-failures",
            "1",
        ]
    )
    assert args.model_cache_dir == Path("cache")
    assert args.max_new_failures == 1


def test_full_pavg_methods_prohibit_sam2_cache_misses():
    with pytest.raises(ValueError, match="cache-only"):
        validate_cache_only_observations(("M1_GRAPH", "M5_FULL"), "sam2")

    validate_cache_only_observations(("B1_RULE",), "sam2")


def test_chat_json_schema_mode_is_explicit_and_forwarded(monkeypatch):
    args = build_parser().parse_args(
        [
            "--manifest",
            "m.json",
            "--run-dir",
            "run",
            "--methods",
            "D0_DIRECT_VLM",
            "--provider",
            "chat",
            "--chat-response-format",
            "json_schema",
        ]
    )
    monkeypatch.setenv("BENCH_API_KEY", "local")
    monkeypatch.setenv("BENCH_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

    model = build_benchmark_model(
        args.provider,
        chat_response_format=args.chat_response_format,
    )

    assert model.strict_json_schema is True


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


def test_resolved_config_is_write_once_and_exactly_compared(tmp_path):
    path = tmp_path / "resolved_config.json"
    config = {"manifest_sha256": "a" * 64, "frame_count": 16}

    write_resolved_config(path, config)
    original = path.read_bytes()
    write_resolved_config(path, dict(config))

    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="different resolved config"):
        write_resolved_config(path, {**config, "frame_count": 8})
    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob("*.tmp"))


def test_sample_selection_hash_is_order_independent_and_membership_sensitive(
    sample_factory,
):
    first = sample_factory(index=1, physical=True, generator="g")
    second = sample_factory(index=2, physical=False, generator="g")

    assert sample_selection_sha256((first, second)) == sample_selection_sha256(
        (second, first)
    )
    assert sample_selection_sha256((first,)) != sample_selection_sha256(
        (first, second)
    )


def test_manifest_hash_and_diagnostic_method_binding_are_enforced():
    pilot = "a97762fe4033789eb14a82717c72c14e89bc75a7a67200d5890ff1647f72a670"
    shuffled = "5250aea3077f9360e42e20008ee8873a9d9a5f3284e7b52270cba33b098e5848"

    validate_manifest_binding(("M5_ORACLE_PLAN_300",), pilot, pilot)
    validate_manifest_binding(("M5_SHUFFLED_PROMPT_300",), shuffled, shuffled)
    with pytest.raises(ValueError, match="expected manifest SHA-256"):
        validate_manifest_binding(("M5_FULL",), pilot, "0" * 64)
    with pytest.raises(ValueError, match="oracle method requires"):
        validate_manifest_binding(("M5_ORACLE_PLAN_300",), shuffled, shuffled)
    with pytest.raises(ValueError, match="shuffled method requires"):
        validate_manifest_binding(("M5_SHUFFLED_PROMPT_300",), pilot, pilot)
