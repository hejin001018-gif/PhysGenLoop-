"""Run the resumable PAVG Stage A video benchmark smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from pavg_critic.api_models import OpenAIChatModel, OpenAIResponsesModel
from pavg_critic.benchmarking.baselines import DirectVLMJudge
from pavg_critic.benchmarking.audited_runner import AuditedBenchmarkRunner
from pavg_critic.benchmarking.datasets import load_manifest
from pavg_critic.benchmarking.model_cache import AuditedCachedModel
from pavg_critic.benchmarking.pavg_methods import (
    CachedObservationProvider,
    PAVGMethod,
    make_sam2_observation_producer,
)
from pavg_critic.benchmarking.report import write_smoke_report
from pavg_critic.benchmarking.runner import BenchmarkRunner, load_predictions


ALLOWED_METHODS = (
    "D0_DIRECT_VLM",
    "D1_STRUCTURED_VLM",
    "B1_RULE",
    "M1_GRAPH",
    "M2_CHECKLIST",
    "M3_MECHANICS",
    "M4_VLM",
    "M5_FULL",
    "M5_SHUFFLED_PROMPT_300",
    "M5_ORACLE_PLAN_300",
)
PILOT_MANIFEST_SHA256 = (
    "a97762fe4033789eb14a82717c72c14e89bc75a7a67200d5890ff1647f72a670"
)
SHUFFLED_MANIFEST_SHA256 = (
    "5250aea3077f9360e42e20008ee8873a9d9a5f3284e7b52270cba33b098e5848"
)
CACHE_ONLY_PAVG_METHODS = frozenset(
    {
        "M1_GRAPH",
        "M2_CHECKLIST",
        "M3_MECHANICS",
        "M4_VLM",
        "M5_FULL",
        "M5_SHUFFLED_PROMPT_300",
        "M5_ORACLE_PLAN_300",
    }
)


def parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in methods if item not in ALLOWED_METHODS]
    if not methods or unknown:
        raise ValueError(f"unknown benchmark method(s): {unknown}")
    if len(methods) != len(set(methods)):
        raise ValueError("duplicate benchmark methods are not allowed")
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PAVG Stage A video benchmark smoke"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--provider",
        choices=("responses", "chat"),
        default="responses",
    )
    parser.add_argument(
        "--chat-response-format",
        choices=("json_object", "json_schema"),
        default="json_object",
    )
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--observations-dir", type=Path)
    parser.add_argument(
        "--observation-provider",
        choices=("none", "sam2"),
        default="none",
    )
    parser.add_argument("--sam2-config")
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--m4-detector-weight", type=float, default=0.7)
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--model-snapshot-sha256")
    parser.add_argument("--max-new-failures", type=int, default=1)
    parser.add_argument("--expected-manifest-sha256")
    return parser


def load_benchmark_environment(path: str | Path) -> dict[str, object]:
    """Map project env names to BENCH_* without exposing credential values."""

    try:
        from dotenv import dotenv_values
    except ImportError as exc:
        raise RuntimeError("--env-file requires pavg-critic[env]") from exc
    values = dotenv_values(path)
    api_key = str(values.get("BENCH_API_KEY") or values.get("API_KEY") or "")
    base_url = str(values.get("BENCH_BASE_URL") or values.get("BASE_URL") or "")
    model = str(values.get("BENCH_MODEL") or values.get("VLM_MODEL") or "")
    if api_key:
        os.environ.setdefault("BENCH_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("BENCH_BASE_URL", base_url)
    if model:
        os.environ.setdefault("BENCH_MODEL", model)
    return {
        "api_key_configured": bool(api_key),
        "base_url_configured": bool(base_url),
        "model": model or None,
    }


def build_benchmark_model(
    provider: str,
    *,
    chat_response_format: str = "json_object",
):
    api_key = os.environ.get("BENCH_API_KEY", "")
    model = os.environ.get("BENCH_MODEL", "")
    if not api_key or not model:
        raise ValueError(
            "Set BENCH_API_KEY and BENCH_MODEL before running model baselines"
        )
    if provider == "responses":
        return OpenAIResponsesModel(
            api_key=api_key,
            model=model,
            base_url=os.environ.get(
                "BENCH_BASE_URL",
                "https://api.openai.com/v1",
            ),
        )
    if provider == "chat":
        return OpenAIChatModel(
            api_key=api_key,
            model=model,
            base_url=os.environ.get(
                "BENCH_BASE_URL",
                "http://127.0.0.1:8000/v1",
            ),
            strict_json_schema=chat_response_format == "json_schema",
        )
    raise ValueError(f"unsupported provider: {provider}")


def _unavailable_observations(sample):
    raise ValueError(f"ObservationUnavailable: {sample.sample_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _redact_config(value):
    if isinstance(value, dict):
        return {
            key: (
                "REDACTED"
                if any(mark in key.lower() for mark in ("key", "token", "secret"))
                else _redact_config(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_config(item) for item in value]
    return value


def sample_selection_sha256(samples) -> str:
    sample_ids = sorted(str(sample.sample_id) for sample in samples)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample selection contains duplicate sample IDs")
    content = json.dumps(
        sample_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_resolved_config(path: str | Path, config: dict[str, object]) -> None:
    destination = Path(path)
    content = (
        json.dumps(
            config,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise ValueError(
                f"run directory contains a different resolved config: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            raw_temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw_temporary, destination)
        raw_temporary = None
        _fsync_directory(destination.parent)
    finally:
        if raw_temporary is not None:
            Path(raw_temporary).unlink(missing_ok=True)


def validate_manifest_binding(
    method_ids: tuple[str, ...],
    actual_sha256: str,
    expected_sha256: str | None,
) -> None:
    actual = actual_sha256.lower()
    expected = "" if expected_sha256 is None else expected_sha256.lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("--expected-manifest-sha256 must be a 64-character hex digest")
    if actual != expected:
        raise ValueError(
            f"expected manifest SHA-256 {expected}, observed {actual}"
        )
    if "M5_ORACLE_PLAN_300" in method_ids and actual != PILOT_MANIFEST_SHA256:
        raise ValueError(
            "oracle method requires the frozen correct-prompt pilot manifest"
        )
    if (
        "M5_SHUFFLED_PROMPT_300" in method_ids
        and actual != SHUFFLED_MANIFEST_SHA256
    ):
        raise ValueError(
            "shuffled method requires the frozen shuffled-prompt manifest"
        )


def validate_cache_only_observations(
    method_ids: tuple[str, ...], observation_provider: str
) -> None:
    protected = CACHE_ONLY_PAVG_METHODS.intersection(method_ids)
    if protected and observation_provider == "sam2":
        raise ValueError(
            "full PAVG evaluation is cache-only and must not propagate SAM2; "
            f"cache misses are terminal for {sorted(protected)!r}"
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.env_file is not None:
        load_benchmark_environment(args.env_file)
    method_ids = parse_methods(args.methods)
    validate_cache_only_observations(method_ids, args.observation_provider)
    manifest_sha256 = _sha256(args.manifest)
    validate_manifest_binding(
        method_ids,
        manifest_sha256,
        args.expected_manifest_sha256,
    )
    samples = tuple(
        sorted(load_manifest(args.manifest), key=lambda item: item.sample_id)
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    if args.frame_count <= 0:
        raise ValueError("--frame-count must be positive")

    model_pavg_ids = {
        "M4_VLM",
        "M5_FULL",
        "M5_SHUFFLED_PROMPT_300",
        "M5_ORACLE_PLAN_300",
    }
    if model_pavg_ids.intersection(method_ids) and args.model_cache_dir is None:
        raise ValueError("--model-cache-dir is required for M4/M5 methods")
    if model_pavg_ids.intersection(method_ids) and (
        args.model_snapshot_sha256 is None
        or len(args.model_snapshot_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.model_snapshot_sha256.lower()
        )
    ):
        raise ValueError(
            "--model-snapshot-sha256 is required for M4/M5 methods and must "
            "be a 64-character hex digest"
        )
    if args.max_new_failures <= 0:
        raise ValueError("--max-new-failures must be positive")

    requires_model = (
        any(item.startswith("D") for item in method_ids)
        or bool(model_pavg_ids.intersection(method_ids))
        or args.observation_provider == "sam2"
    )
    model = (
        build_benchmark_model(
            args.provider,
            chat_response_format=args.chat_response_format,
        )
        if requires_model
        else None
    )
    model_id = os.environ.get("BENCH_MODEL") if model is not None else None
    methods = []
    if "D0_DIRECT_VLM" in method_ids:
        methods.append(
            DirectVLMJudge(
                model,
                model_id=model_id,
                structured=False,
                frame_count=args.frame_count,
            )
        )
    if "D1_STRUCTURED_VLM" in method_ids:
        methods.append(
            DirectVLMJudge(
                model,
                model_id=model_id,
                structured=True,
                frame_count=args.frame_count,
            )
        )
    pavg_ids = [
        item
        for item in method_ids
        if item
        in {
            "B1_RULE",
            "M1_GRAPH",
            "M2_CHECKLIST",
            "M3_MECHANICS",
            "M4_VLM",
            "M5_FULL",
            "M5_SHUFFLED_PROMPT_300",
            "M5_ORACLE_PLAN_300",
        }
    ]
    if pavg_ids:
        if args.observations_dir is None:
            raise ValueError("--observations-dir is required for PAVG methods")
        if args.observation_provider == "sam2":
            if not args.sam2_config or args.sam2_checkpoint is None:
                raise ValueError(
                    "--sam2-config and --sam2-checkpoint are required for "
                    "SAM2 observations"
                )
            producer = make_sam2_observation_producer(
                model,
                model_config=args.sam2_config,
                checkpoint=str(args.sam2_checkpoint),
            )
        else:
            producer = _unavailable_observations
        observations = CachedObservationProvider(args.observations_dir, producer)
        stage_models = {}
        if model_pavg_ids.intersection(pavg_ids):
            assert args.model_cache_dir is not None
            stage_models["verifier"] = AuditedCachedModel(
                model,
                cache_dir=args.model_cache_dir,
                namespace="verifier",
                model_id=model_id or "benchmark-model",
                model_revision=args.model_snapshot_sha256,
            )
        if any(item.startswith("M5_") or item == "M5_FULL" for item in pavg_ids):
            assert args.model_cache_dir is not None
            stage_models["planner"] = AuditedCachedModel(
                model,
                cache_dir=args.model_cache_dir,
                namespace="planner",
                model_id=model_id or "benchmark-model",
                model_revision=args.model_snapshot_sha256,
            )
            stage_models["pqsg"] = AuditedCachedModel(
                model,
                cache_dir=args.model_cache_dir,
                namespace="pqsg",
                model_id=model_id or "benchmark-model",
                model_revision=args.model_snapshot_sha256,
            )
        for item in pavg_ids:
            is_m5 = item.startswith("M5_") or item == "M5_FULL"
            mode = "M5_FULL" if is_m5 else item
            used_stages = {
                name: stage_models[name]
                for name in (
                    ("planner", "pqsg", "verifier")
                    if is_m5
                    else (("verifier",) if item == "M4_VLM" else ())
                )
            }
            methods.append(
                PAVGMethod(
                    mode,
                    observations,
                    model_id=model_id,
                    planner_model=used_stages.get("planner"),
                    question_model=used_stages.get("pqsg"),
                    verifier_model=used_stages.get("verifier"),
                    verifier_detector_weight=args.m4_detector_weight,
                    model_stages=used_stages,
                    output_method_id=item,
                    oracle_plan=item == "M5_ORACLE_PLAN_300",
                )
            )
    by_id = {method.method_id: method for method in methods}
    ordered_methods = tuple(by_id[item] for item in method_ids)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = _redact_config(
        {
            "provider": args.provider,
            "chat_response_format": args.chat_response_format,
            "model": model_id,
            "frame_count": args.frame_count,
            "methods": list(method_ids),
            "observation_provider": args.observation_provider,
            "m4_detector_weight": args.m4_detector_weight,
            "model_cache_dir": (
                None if args.model_cache_dir is None else str(args.model_cache_dir)
            ),
            "model_snapshot_sha256": args.model_snapshot_sha256,
            "model_cache_namespaces": sorted(
                {
                    name
                    for method in methods
                    for name in getattr(method, "model_stages", {})
                }
            ),
            "max_new_failures": args.max_new_failures,
            "sam2_config": args.sam2_config,
            "sam2_checkpoint_sha256": (
                _sha256(args.sam2_checkpoint)
                if args.sam2_checkpoint is not None
                and args.sam2_checkpoint.is_file()
                else None
            ),
            "manifest_sha256": manifest_sha256,
            "expected_manifest_sha256": args.expected_manifest_sha256,
            "sample_count": len(samples),
            "sample_selection_sha256": sample_selection_sha256(samples),
            "max_samples": args.max_samples,
            "git_revision": _git_revision(),
        }
    )
    write_resolved_config(args.run_dir / "resolved_config.json", config)
    prediction_path = args.run_dir / "predictions.jsonl"
    direct_methods = tuple(
        method for method in ordered_methods if method.method_id.startswith("D")
    )
    pavg_methods = tuple(
        method for method in ordered_methods if not method.method_id.startswith("D")
    )
    if direct_methods:
        BenchmarkRunner(prediction_path).run(
            samples,
            direct_methods,
            max_new_failures=args.max_new_failures,
        )
    if pavg_methods:
        AuditedBenchmarkRunner(
            prediction_path,
            args.run_dir / "diagnostics.jsonl",
            max_new_failures=args.max_new_failures,
        ).run(samples, pavg_methods)
    predictions = load_predictions(prediction_path)
    write_smoke_report(samples, predictions, args.run_dir)
    for method_id in method_ids:
        records = [item for item in predictions if item.method_id == method_id]
        if not records or all(item.failure is not None for item in records):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
