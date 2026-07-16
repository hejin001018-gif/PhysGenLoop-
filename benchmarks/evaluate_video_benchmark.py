"""Run the resumable PAVG Stage A video benchmark smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from pavg_critic.api_models import OpenAIChatModel, OpenAIResponsesModel
from pavg_critic.benchmarking.baselines import DirectVLMJudge
from pavg_critic.benchmarking.datasets import load_manifest
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
    parser.add_argument("--m4-detector-weight", type=float, default=0.4)
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.env_file is not None:
        load_benchmark_environment(args.env_file)
    method_ids = parse_methods(args.methods)
    samples = tuple(
        sorted(load_manifest(args.manifest), key=lambda item: item.sample_id)
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    if args.frame_count <= 0:
        raise ValueError("--frame-count must be positive")

    requires_model = (
        any(item.startswith("D") for item in method_ids)
        or "M4_VLM" in method_ids
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
        in {"B1_RULE", "M1_GRAPH", "M2_CHECKLIST", "M3_MECHANICS", "M4_VLM"}
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
        methods.extend(
            PAVGMethod(
                item,
                observations,
                model_id=model_id,
                verifier_model=model if item == "M4_VLM" else None,
                verifier_detector_weight=args.m4_detector_weight,
            )
            for item in pavg_ids
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
            "sam2_config": args.sam2_config,
            "sam2_checkpoint_sha256": (
                _sha256(args.sam2_checkpoint)
                if args.sam2_checkpoint is not None
                and args.sam2_checkpoint.is_file()
                else None
            ),
            "manifest_sha256": _sha256(args.manifest),
            "git_revision": _git_revision(),
        }
    )
    (args.run_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction_path = args.run_dir / "predictions.jsonl"
    BenchmarkRunner(prediction_path).run(samples, ordered_methods)
    predictions = load_predictions(prediction_path)
    write_smoke_report(samples, predictions, args.run_dir)
    for method_id in method_ids:
        records = [item for item in predictions if item.method_id == method_id]
        if not records or all(item.failure is not None for item in records):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
