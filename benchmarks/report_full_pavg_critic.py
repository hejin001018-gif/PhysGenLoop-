"""Publish an immutable, audit-ready complete PAVG Critic report bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from pavg_critic.benchmarking.contracts import BenchmarkPrediction
from pavg_critic.benchmarking.datasets import load_manifest
from pavg_critic.benchmarking.full_pavg_report import (
    FULL_METHODS,
    PAVG_DIAGNOSTIC_METHODS,
    build_full_pavg_report,
)
from pavg_critic.benchmarking.full_report import summarize_observation_latencies


CORE_REPORT_FILES = (
    "artifact_audit.json",
    "merged_diagnostics.jsonl",
    "merged_predictions.jsonl",
    "module_attribution.json",
    "paired_outcomes.json",
    "prompt_diagnostics.json",
    "slices.json",
    "summary.json",
    "summary.md",
)
FULL_MANIFEST_SHA256 = (
    "d8be5fe97ddf6902515c09ccbb53f394b25230213db7c3058d61f84748624906"
)
PILOT_MANIFEST_SHA256 = (
    "a97762fe4033789eb14a82717c72c14e89bc75a7a67200d5890ff1647f72a670"
)
SHUFFLED_MANIFEST_SHA256 = (
    "5250aea3077f9360e42e20008ee8873a9d9a5f3284e7b52270cba33b098e5848"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report the frozen complete PAVG Critic evaluation"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, action="append", type=Path)
    parser.add_argument("--diagnostics", required=True, action="append", type=Path)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument(
        "--prompt-predictions", required=True, action="append", type=Path
    )
    parser.add_argument(
        "--prompt-resolved-config", required=True, action="append", type=Path
    )
    parser.add_argument(
        "--observation-meta-dir", required=True, action="append", type=Path
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    return parser


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    ).encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_frozen_manifests(manifest: Path, pilot_manifest: Path) -> None:
    observed_full = _sha(manifest.read_bytes())
    observed_pilot = _sha(pilot_manifest.read_bytes())
    if observed_full != FULL_MANIFEST_SHA256:
        raise ValueError(
            f"full manifest SHA-256 mismatch: {observed_full}"
        )
    if observed_pilot != PILOT_MANIFEST_SHA256:
        raise ValueError(
            f"pilot manifest SHA-256 mismatch: {observed_pilot}"
        )


def _capture(paths: Sequence[Path], *, kind: str) -> dict[Path, bytes]:
    result = {}
    for path in sorted(paths, key=lambda item: os.path.normcase(str(item.resolve()))):
        if not path.is_file():
            raise ValueError(f"{kind} input does not exist: {path}")
        result[path] = path.read_bytes()
    return result


def _load_predictions(
    captured: Mapping[Path, bytes],
) -> tuple[BenchmarkPrediction, ...]:
    result = []
    for path, content in captured.items():
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"prediction input is not UTF-8: {path}") from exc
        for line_number, line in enumerate(lines, start=1):
            try:
                raw = json.loads(line)
                result.append(BenchmarkPrediction.from_dict(raw))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"invalid prediction JSONL line {line_number}: {path}"
                ) from exc
    return tuple(result)


def _load_diagnostics(
    captured: Mapping[Path, bytes],
) -> tuple[dict[str, Any], ...]:
    result = []
    for path, content in captured.items():
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"diagnostics input is not UTF-8: {path}") from exc
        for line_number, line in enumerate(lines, start=1):
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("diagnostic record must be an object")
                result.append(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"invalid diagnostics JSONL line {line_number}: {path}"
                ) from exc
    return tuple(result)


def validate_prompt_run_bindings(
    captured: Mapping[Path, bytes],
    *,
    expected_sample_count: int = 300,
) -> None:
    expected = {
        "M5_SHUFFLED_PROMPT_300": SHUFFLED_MANIFEST_SHA256,
        "M5_ORACLE_PLAN_300": PILOT_MANIFEST_SHA256,
    }
    observed: dict[str, str] = {}
    for path, content in captured.items():
        try:
            config = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid prompt resolved config: {path}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"prompt resolved config must be an object: {path}")
        methods = config.get("methods")
        if not isinstance(methods, list) or len(methods) != 1:
            raise ValueError("prompt resolved config bindings require one method")
        method = str(methods[0])
        if method not in expected or method in observed:
            raise ValueError(
                "prompt resolved config bindings must contain each frozen "
                "diagnostic method exactly once"
            )
        manifest_hash = str(config.get("manifest_sha256", "")).lower()
        expected_hash = str(config.get("expected_manifest_sha256", "")).lower()
        if (
            manifest_hash != expected[method]
            or expected_hash != expected[method]
            or config.get("sample_count") != expected_sample_count
        ):
            raise ValueError(
                f"prompt resolved config bindings mismatch for {method}"
            )
        observed[method] = manifest_hash
    if set(observed) != set(expected):
        raise ValueError(
            "prompt resolved config bindings must cover shuffled and oracle runs"
        )


def _markdown(summary: Mapping[str, Any]) -> bytes:
    metrics = summary["method_metrics"]
    primary = summary["primary"]
    decision = summary["material_decision"]
    lines = [
        "# VideoPhy-2 完整 PAVG Critic 评测",
        "",
        f"样本数：{summary['population']['sample_count']}；"
        f"预测数：{summary['population']['prediction_count']}；"
        f"模块诊断数：{summary['population']['diagnostic_count']}。",
        "",
        "## 完整方法矩阵",
        "",
        "| 方法 | Accuracy | Balanced Accuracy | Macro-F1 | Physical recall | Violation recall | Violation precision | Unknown rate | Failure rate | Physics Spearman | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in FULL_METHODS:
        item = metrics[method]
        spearman = item["physics_spearman"]
        spearman_text = "N/A" if spearman is None else f"{spearman:.6f}"
        lines.append(
            f"| {method} | {item['accuracy']:.6f} | "
            f"{item['balanced_accuracy']:.6f} | {item['macro_f1']:.6f} | "
            f"{item['physical_recall']:.6f} | {item['violation_recall']:.6f} | "
            f"{item['violation_precision']:.6f} | {item['unknown_rate']:.6f} | "
            f"{item['failure_rate']:.6f} | {spearman_text} | "
            f"{item['mean_latency_sec']:.6f} |"
        )
    sam2 = summary["sam2_production_latency"]
    lines.extend(
        [
            "",
            "## SAM2 production latency (cached provenance)",
            "",
            f"Valid: {sam2['valid_count']}/{sam2['expected_count']}; "
            f"missing: {sam2['missing_count']}; "
            f"mean: {sam2['mean_production_latency_sec']}; "
            f"p50: {sam2['p50_production_latency_sec']}; "
            f"p95: {sam2['p95_production_latency_sec']}.",
        ]
    )
    delta = primary["candidate_minus_baseline"]
    bootstrap = primary["bootstrap"]
    lines.extend(
        [
            "",
            "## 主比较：M5_FULL − D0_DIRECT_VLM",
            "",
            f"Macro-F1 差值：{delta['macro_f1']:+.6f}；"
            f"Accuracy 差值：{delta['accuracy']:+.6f}。",
            f"action-group bootstrap 95% CI："
            f"[{bootstrap['lower']:+.6f}, {bootstrap['upper']:+.6f}]。",
            "",
            "## 冻结门槛",
            "",
        ]
    )
    for name, gate in decision["gates"].items():
        lines.append(
            f"- {name}: value={gate['value']}, operator={gate['operator']}, "
            f"threshold={gate['threshold']}, pass={str(gate['pass']).lower()}"
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "> 本报告只评测 prompt-conditioned Critic。Generator/Repairer/Selector loop "
            "未评测，VideoPhy-1 OOD 仍为 deferred。",
            "",
            f"VideoPhy-2 support：{str(decision['videophy2_support']).lower()}；"
            "overall：not_evaluable_ood_deferred。",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _publish(output_dir: Path, artifacts: Mapping[str, bytes]) -> None:
    if tuple(sorted(artifacts)) != tuple(sorted(CORE_REPORT_FILES)):
        raise ValueError("report artifact set does not match CORE_REPORT_FILES")
    if output_dir.exists():
        if (
            output_dir.is_dir()
            and set(path.name for path in output_dir.iterdir()) == set(CORE_REPORT_FILES)
            and all((output_dir / name).read_bytes() == content for name, content in artifacts.items())
        ):
            return
        raise ValueError(f"output directory already contains a different bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.full-pavg-report-",
        )
    )
    try:
        for name in CORE_REPORT_FILES:
            _write_file(staged / name, artifacts[name])
        os.replace(staged, output_dir)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def _build_artifacts(
    *,
    manifest: Path,
    prediction_paths: Sequence[Path],
    diagnostic_paths: Sequence[Path],
    pilot_manifest: Path,
    prompt_prediction_paths: Sequence[Path],
    prompt_resolved_config_paths: Sequence[Path],
    observation_dirs: Sequence[Path],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, bytes], dict[Path, bytes]]:
    captured = _capture(
        (
            manifest,
            *prediction_paths,
            *diagnostic_paths,
            pilot_manifest,
            *prompt_prediction_paths,
            *prompt_resolved_config_paths,
        ),
        kind="report",
    )
    samples = tuple(sorted(load_manifest(manifest), key=lambda item: item.sample_id))
    pilot_samples = tuple(
        sorted(load_manifest(pilot_manifest), key=lambda item: item.sample_id)
    )
    validate_prompt_run_bindings(
        {path: captured[path] for path in prompt_resolved_config_paths},
        expected_sample_count=len(pilot_samples),
    )
    predictions = _load_predictions(
        {path: captured[path] for path in prediction_paths}
    )
    diagnostics = _load_diagnostics(
        {path: captured[path] for path in diagnostic_paths}
    )
    prompt_predictions = _load_predictions(
        {path: captured[path] for path in prompt_prediction_paths}
    )
    summary = build_full_pavg_report(
        samples=samples,
        predictions=predictions,
        diagnostics=diagnostics,
        pilot_samples=pilot_samples,
        prompt_predictions=prompt_predictions,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    summary["sam2_production_latency"] = summarize_observation_latencies(
        samples, observation_dirs
    )
    method_order = {method: index for index, method in enumerate(FULL_METHODS)}
    merged_predictions = tuple(
        item.to_dict()
        for item in sorted(
            predictions,
            key=lambda item: (item.sample_id, method_order[item.method_id]),
        )
    )
    merged_diagnostics = tuple(
        sorted(
            diagnostics,
            key=lambda item: (
                str(item["key"]["sample_id"]),
                PAVG_DIAGNOSTIC_METHODS.index(str(item["key"]["method_id"])),
            ),
        )
    )
    attribution = {
        "sequential_attribution": summary["sequential_attribution"],
        "module_availability": summary["module_availability"],
        "model_calls": summary["model_calls"],
        "hard_override": summary["hard_override"],
        "provider_failure_count": summary["provider_failure_count"],
    }
    artifacts = {
        "merged_predictions.jsonl": _jsonl_bytes(merged_predictions),
        "merged_diagnostics.jsonl": _jsonl_bytes(merged_diagnostics),
        "module_attribution.json": _json_bytes(attribution),
        "paired_outcomes.json": _json_bytes(summary["primary"]["paired_outcomes"]),
        "prompt_diagnostics.json": _json_bytes(summary["prompt_diagnostics"]),
        "slices.json": _json_bytes(summary["slices"]),
        "summary.json": _json_bytes(summary),
        "summary.md": _markdown(summary),
    }
    observation_files = tuple(
        sorted(
            (path for directory in observation_dirs for path in directory.rglob("*.meta.json")),
            key=lambda item: os.path.normcase(str(item.resolve())),
        )
    )
    observation_capture = _capture(observation_files, kind="observation metadata")
    audit = {
        "schema_version": "1.0",
        "methods": list(FULL_METHODS),
        "diagnostic_methods": list(PAVG_DIAGNOSTIC_METHODS),
        "inputs": [
            {"path": str(path), "sha256": _sha(content), "size": len(content)}
            for path, content in captured.items()
        ],
        "observation_metadata": {
            "file_count": len(observation_capture),
            "files": [
                {"path": str(path), "sha256": _sha(content)}
                for path, content in observation_capture.items()
            ],
        },
        "outputs": {
            name: {"sha256": _sha(content), "size": len(content)}
            for name, content in sorted(artifacts.items())
        },
    }
    artifacts = {"artifact_audit.json": _json_bytes(audit), **artifacts}
    return artifacts, {**captured, **observation_capture}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_frozen_manifests(args.manifest, args.pilot_manifest)
    artifacts, captured = _build_artifacts(
        manifest=args.manifest,
        prediction_paths=tuple(args.predictions),
        diagnostic_paths=tuple(args.diagnostics),
        pilot_manifest=args.pilot_manifest,
        prompt_prediction_paths=tuple(args.prompt_predictions),
        prompt_resolved_config_paths=tuple(args.prompt_resolved_config),
        observation_dirs=tuple(args.observation_meta_dir),
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for path, content in captured.items():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"report input changed during generation: {path}")
    _publish(args.output_dir, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
