"""Generate the deterministic, audit-ready full VideoPhy-2 report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pavg_critic.benchmarking.datasets import load_manifest
from pavg_critic.benchmarking.full_report import (
    action_group_bootstrap,
    build_slices,
    evaluate_material_improvement,
    merge_prediction_shards,
    paired_outcomes,
    strict_method_metrics,
    summarize_observation_latencies,
)


BASELINE_METHOD = "D0_DIRECT_VLM"
CANDIDATE_METHOD = "B1_RULE"
CORE_REPORT_FILES = (
    "artifact_audit.json",
    "merged_predictions.jsonl",
    "paired_outcomes.json",
    "slices.json",
    "summary.json",
    "summary.md",
)


@dataclass(frozen=True)
class _FrozenFile:
    path: Path
    content: bytes
    sha256: str


@dataclass(frozen=True)
class _FrozenObservationFile:
    path: Path
    directory_index: int
    relative_path: str
    content: bytes
    sha256: str


@dataclass(frozen=True)
class _InputSnapshot:
    manifest: _FrozenFile
    predictions: tuple[_FrozenFile, ...]
    observation_dirs: tuple[Path, ...]
    observation_files: tuple[_FrozenObservationFile, ...]

    @property
    def paths(self) -> tuple[Path, ...]:
        return (
            self.manifest.path,
            *(item.path for item in self.predictions),
            *(item.path for item in self.observation_files),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge and report the frozen full VideoPhy-2 evaluation"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--predictions",
        required=True,
        action="append",
        type=Path,
        help="Prediction JSONL shard; repeat for every canonical shard",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260717,
    )
    parser.add_argument(
        "--observation-meta-dir",
        required=True,
        action="append",
        type=Path,
        help="SAM2 observation metadata root; repeat for every shard",
    )
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


def _deterministic_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _failure_reason(failure: Mapping[str, Any] | object) -> str:
    if not isinstance(failure, Mapping):
        return _deterministic_json(failure)
    for key in ("reason", "message", "type", "error"):
        value = failure.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None:
            return _deterministic_json(value)
    return _deterministic_json(failure)


def _markdown_text(value: object) -> str:
    """Render untrusted text without Markdown, HTML, or line injection."""

    markdown_punctuation = "\\`*_{}[]()#+-.!|"
    escaped: list[str] = []
    for character in str(value):
        codepoint = ord(character)
        if character == "\n":
            escaped.append("\\n")
        elif character == "\r":
            escaped.append("\\r")
        elif character == "\t":
            escaped.append("\\t")
        elif codepoint < 32 or codepoint == 127:
            escaped.append(f"\\u{codepoint:04x}")
        elif character == "&":
            escaped.append("&amp;")
        elif character == "<":
            escaped.append("&lt;")
        elif character == ">":
            escaped.append("&gt;")
        elif character in markdown_punctuation:
            escaped.append(f"&#{codepoint};")
        else:
            escaped.append(character)
    return "".join(escaped)


def _canonical_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _freeze_file(path: Path, *, kind: str) -> _FrozenFile:
    if not path.is_file():
        raise ValueError(f"{kind} input does not exist: {path}")
    content = path.read_bytes()
    return _FrozenFile(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _capture_input_snapshot(
    manifest: Path,
    prediction_paths: Sequence[Path],
    observation_meta_dirs: Sequence[Path],
) -> _InputSnapshot:
    if not prediction_paths:
        raise ValueError("at least one prediction input is required")
    if not observation_meta_dirs:
        raise ValueError("at least one observation metadata directory is required")
    frozen_manifest = _freeze_file(manifest, kind="manifest")
    frozen_predictions = tuple(
        _freeze_file(path, kind="prediction")
        for path in sorted(prediction_paths, key=_canonical_path)
    )
    directories = tuple(sorted(observation_meta_dirs, key=_canonical_path))
    observations: list[_FrozenObservationFile] = []
    for directory_index, directory in enumerate(directories):
        if not directory.is_dir():
            raise ValueError(
                f"observation metadata directory does not exist: {directory}"
            )
        for path in sorted(directory.rglob("*.meta.json"), key=_canonical_path):
            frozen = _freeze_file(path, kind="observation metadata")
            observations.append(
                _FrozenObservationFile(
                    path=path,
                    directory_index=directory_index,
                    relative_path=path.relative_to(directory).as_posix(),
                    content=frozen.content,
                    sha256=frozen.sha256,
                )
            )
    return _InputSnapshot(
        manifest=frozen_manifest,
        predictions=frozen_predictions,
        observation_dirs=directories,
        observation_files=tuple(observations),
    )


def _verify_input_snapshot(snapshot: _InputSnapshot) -> None:
    try:
        current = _capture_input_snapshot(
            snapshot.manifest.path,
            tuple(item.path for item in snapshot.predictions),
            snapshot.observation_dirs,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "input snapshot changed during report generation"
        ) from exc
    if current != snapshot:
        raise ValueError("input snapshot changed during report generation")


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        return os.path.samefile(first, second)
    return False


def _write_staged_artifact(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_staged_directory(staged: Path, destination: Path) -> None:
    os.replace(staged, destination)


def _existing_bundle_matches(
    output_dir: Path,
    artifacts: Mapping[str, bytes],
) -> bool:
    if output_dir.is_symlink() or not output_dir.is_dir():
        return False
    children = tuple(sorted(output_dir.iterdir(), key=lambda path: path.name))
    if tuple(path.name for path in children) != CORE_REPORT_FILES:
        return False
    return all(
        path.is_file()
        and not path.is_symlink()
        and path.read_bytes() == artifacts[path.name]
        for path in children
    )


def _publish_report_bundle(
    output_dir: Path,
    artifacts: Mapping[str, bytes],
    *,
    input_paths: Sequence[Path],
) -> None:
    """Publish one immutable bundle with a single atomic directory replace."""

    if tuple(sorted(artifacts)) != CORE_REPORT_FILES:
        raise ValueError(
            f"report bundle must contain exactly {CORE_REPORT_FILES!r}"
        )
    destinations = {name: output_dir / name for name in CORE_REPORT_FILES}
    for destination in destinations.values():
        for input_path in input_paths:
            if _paths_alias(destination, input_path):
                raise ValueError(
                    f"report destination aliases an input: {input_path}"
                )

    if os.path.lexists(output_dir):
        if _existing_bundle_matches(output_dir, artifacts):
            return
        raise ValueError(
            f"output directory already exists with a different or partial "
            f"bundle: {output_dir}; choose a new output path"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.full-report-",
        )
    )
    try:
        for name in CORE_REPORT_FILES:
            _write_staged_artifact(staged / name, artifacts[name])
        _publish_staged_directory(staged, output_dir)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def _candidate_minus_baseline(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    fields = (
        "accuracy",
        "macro_f1",
        "physical_recall",
        "violation_recall",
        "failure_rate",
        "mean_latency_sec",
        "p50_latency_sec",
        "p95_latency_sec",
    )
    return {
        field: float(candidate[field]) - float(baseline[field])
        for field in fields
    }


def _prediction_latency(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean_latency_sec": metrics["mean_latency_sec"],
        "p50_latency_sec": metrics["p50_latency_sec"],
        "p95_latency_sec": metrics["p95_latency_sec"],
    }


def _render_markdown(summary: Mapping[str, Any]) -> bytes:
    baseline_id = summary["method_ids"]["baseline"]
    candidate_id = summary["method_ids"]["candidate"]
    metrics = summary["method_metrics"]
    bootstrap = summary["bootstrap"]
    paired = summary["paired_outcomes"]
    failures = summary["prediction_failures"]
    sam2 = summary["sam2_production_latency"]
    decision = summary["material_decision"]
    lines = [
        "# VideoPhy-2 全量评测报告",
        "",
        f"样本数：{summary['population']['sample_count']}；终止预测数："
        f"{summary['population']['prediction_count']}。",
        "",
        "## 完整指标",
        "",
        "| 方法 | Accuracy | Macro-F1 | Physical recall | Violation recall | Failure rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_id in (baseline_id, candidate_id):
        item = metrics[method_id]
        lines.append(
            f"| {method_id} | {item['accuracy']:.6f} | "
            f"{item['macro_f1']:.6f} | {item['physical_recall']:.6f} | "
            f"{item['violation_recall']:.6f} | {item['failure_rate']:.6f} |"
        )
    delta = summary["candidate_minus_baseline"]
    lines.extend(
        [
            "",
            f"候选减基线 Macro-F1：{delta['macro_f1']:+.6f}；"
            f"Accuracy：{delta['accuracy']:+.6f}。",
            "",
            "## 配对 action-group bootstrap",
            "",
            f"重采样次数：{bootstrap['resamples']}；seed：{bootstrap['seed']}；"
            f"cluster 数：{bootstrap['group_count']}。",
            f"Macro-F1 差值点估计：{bootstrap['point_estimate']:+.6f}；"
            f"95% CI：[{bootstrap['lower']:+.6f}, {bootstrap['upper']:+.6f}]。",
            "",
            "## 配对结果",
            "",
            f"- 两者均正确：{paired['both_correct']}",
            f"- 仅基线正确：{paired['baseline_only_correct']}",
            f"- 仅候选正确：{paired['candidate_only_correct']}",
            f"- 两者均错误：{paired['both_wrong']}",
            "",
            "## 延迟",
            "",
            "预测延迟（模型/规则）与 SAM2 轨迹生产耗时分别统计，不可相加或混用。",
            "",
            "### 预测延迟（模型/规则）",
            "",
        ]
    )
    prediction_latency = summary["prediction_latency"]["methods"]
    for method_id in (baseline_id, candidate_id):
        item = prediction_latency[method_id]
        lines.append(
            f"- {method_id}: mean={item['mean_latency_sec']:.6f}s, "
            f"p50={item['p50_latency_sec']:.6f}s, "
            f"p95={item['p95_latency_sec']:.6f}s"
        )
    lines.extend(
        [
            "",
            "### SAM2 production latency",
            "",
            f"valid={sam2['valid_count']}, missing={sam2['missing_count']}, "
            f"mean={sam2['mean_production_latency_sec']}, "
            f"p50={sam2['p50_production_latency_sec']}, "
            f"p95={sam2['p95_production_latency_sec']} 秒。",
            "",
            "## 失败记录",
            "",
            f"预测失败 {failures['count']} / "
            f"{summary['population']['prediction_count']} "
            f"({failures['rate']:.6f})。失败保留在分母内。",
            "",
        ]
    )
    if failures["records"]:
        lines.extend(
            f"- {_markdown_text(item['sample_id'])} / "
            f"{_markdown_text(item['method_id'])}: "
            f"{_markdown_text(item['reason'])}"
            for item in failures["records"]
        )
    else:
        lines.append("- 无")
    lines.extend(["", "## 冻结门槛", ""])
    for name, gate in decision["gates"].items():
        lines.append(
            f"- {name}: value={gate['value']}, operator={gate['operator']}, "
            f"threshold={gate['threshold']}, pass={str(gate['pass']).lower()}"
        )
    lines.extend(
        [
            "",
            f"VideoPhy-2-only support：{str(decision['videophy2_support']).lower()}。",
            "",
            "## OOD 限制（醒目）",
            "",
            "> **VideoPhy-1 OOD：deferred。**",
            "> **overall: not_evaluable_ood_deferred。**",
            "",
            "本报告只覆盖冻结的 VideoPhy-2 全量比较；在 VideoPhy-1 OOD 完成前，"
            "不能据此声称架构已被证明。",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _load_frozen_manifest(snapshot: _FrozenFile):
    descriptor, raw_path = tempfile.mkstemp(
        dir=snapshot.path.parent,
        prefix=f".{snapshot.path.name}.full-report-snapshot-",
        suffix=snapshot.path.suffix,
    )
    frozen_path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(snapshot.content)
            handle.flush()
            os.fsync(handle.fileno())
        return load_manifest(frozen_path)
    finally:
        frozen_path.unlink(missing_ok=True)


def _build_report_artifacts(
    *,
    snapshot: _InputSnapshot,
    output_dir: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, bytes]:
    if bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    samples = tuple(
        sorted(
            _load_frozen_manifest(snapshot.manifest),
            key=lambda item: item.sample_id,
        )
    )

    with tempfile.TemporaryDirectory(prefix="pavg-full-report-") as raw_temp:
        temporary = Path(raw_temp)
        frozen_prediction_paths = []
        prediction_snapshot_dir = temporary / "predictions"
        prediction_snapshot_dir.mkdir()
        for index, item in enumerate(snapshot.predictions):
            frozen_path = prediction_snapshot_dir / f"{index:06d}.jsonl"
            frozen_path.write_bytes(item.content)
            frozen_prediction_paths.append(frozen_path)
        frozen_observation_dirs = [
            temporary / "observations" / f"{index:06d}"
            for index in range(len(snapshot.observation_dirs))
        ]
        for directory in frozen_observation_dirs:
            directory.mkdir(parents=True)
        for item in snapshot.observation_files:
            frozen_path = (
                frozen_observation_dirs[item.directory_index]
                / Path(item.relative_path)
            )
            frozen_path.parent.mkdir(parents=True, exist_ok=True)
            frozen_path.write_bytes(item.content)
        merge_result = merge_prediction_shards(
            samples,
            frozen_prediction_paths,
            (BASELINE_METHOD, CANDIDATE_METHOD),
            temporary / "merged_predictions.jsonl",
            audit_path=temporary / "artifact_audit.json",
            source_display_paths=tuple(
                item.path for item in snapshot.predictions
            ),
        )
        merged_bytes = (temporary / "merged_predictions.jsonl").read_bytes()
        sam2_latency = summarize_observation_latencies(
            samples,
            frozen_observation_dirs,
        )
        input_audits = merge_result.artifact_audit["inputs"]
        if len(input_audits) != len(snapshot.predictions):
            raise ValueError("frozen prediction snapshot audit count mismatch")
        for input_audit, frozen in zip(input_audits, snapshot.predictions):
            if input_audit["sha256"] != frozen.sha256:
                raise ValueError("frozen prediction snapshot hash mismatch")
            input_audit["path"] = str(frozen.path)
            input_audit["name"] = frozen.path.name
    predictions = merge_result.predictions
    baseline_predictions = tuple(
        item for item in predictions if item.method_id == BASELINE_METHOD
    )
    candidate_predictions = tuple(
        item for item in predictions if item.method_id == CANDIDATE_METHOD
    )
    baseline_metrics = strict_method_metrics(
        samples,
        baseline_predictions,
        expected_method=BASELINE_METHOD,
    )
    candidate_metrics = strict_method_metrics(
        samples,
        candidate_predictions,
        expected_method=CANDIDATE_METHOD,
    )
    paired = paired_outcomes(samples, baseline_predictions, candidate_predictions)
    bootstrap = action_group_bootstrap(
        samples,
        baseline_predictions,
        candidate_predictions,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    slices = build_slices(samples, baseline_predictions, candidate_predictions)
    sam2_latency = {"scope": "sam2_production", **sam2_latency}

    failures = []
    for prediction in predictions:
        if prediction.failure is not None:
            failures.append(
                {
                    "sample_id": prediction.sample_id,
                    "method_id": prediction.method_id,
                    "reason": _failure_reason(prediction.failure),
                }
            )
    failures.sort(key=lambda item: (item["sample_id"], item["method_id"], item["reason"]))
    failure_by_method = {
        method_id: {
            "count": sum(item["method_id"] == method_id for item in failures),
            "rate": metrics["failure_rate"],
        }
        for method_id, metrics in (
            (BASELINE_METHOD, baseline_metrics),
            (CANDIDATE_METHOD, candidate_metrics),
        )
    }
    decision = evaluate_material_improvement(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        bootstrap=bootstrap,
        slices=slices,
        baseline_failure_rate=float(baseline_metrics["failure_rate"]),
        candidate_failure_rate=float(candidate_metrics["failure_rate"]),
    )

    audit = dict(merge_result.artifact_audit)
    audit["merged_output_path"] = str(output_dir / "merged_predictions.jsonl")
    audit["artifact_audit_path"] = str(output_dir / "artifact_audit.json")
    audit["manifest"] = {
        "path": str(snapshot.manifest.path),
        "name": snapshot.manifest.path.name,
        "sha256": snapshot.manifest.sha256,
    }
    audit["observation_metadata"] = {
        "directory_count": len(snapshot.observation_dirs),
        "file_count": len(snapshot.observation_files),
        "files": [
            {
                "path": str(item.path),
                "name": item.path.name,
                "sha256": item.sha256,
            }
            for item in snapshot.observation_files
        ],
    }
    audit["report_artifacts"] = {
        "core_file_count": len(CORE_REPORT_FILES),
        "core_files": list(CORE_REPORT_FILES),
    }
    summary = {
        "population": {
            "sample_count": len(samples),
            "prediction_count": len(predictions),
        },
        "method_ids": {
            "baseline": BASELINE_METHOD,
            "candidate": CANDIDATE_METHOD,
        },
        "method_metrics": {
            BASELINE_METHOD: baseline_metrics,
            CANDIDATE_METHOD: candidate_metrics,
        },
        "candidate_minus_baseline": _candidate_minus_baseline(
            baseline_metrics, candidate_metrics
        ),
        "bootstrap": bootstrap,
        "paired_outcomes": paired,
        "slices": {
            "artifact": "slices.json",
            "counts": {
                dimension: len(items) for dimension, items in slices.items()
            },
        },
        "prediction_failures": {
            "count": len(failures),
            "rate": len(failures) / len(predictions),
            "by_method": failure_by_method,
            "records": failures,
        },
        "prediction_latency": {
            "scope": "model_or_rule_prediction",
            "methods": {
                BASELINE_METHOD: _prediction_latency(baseline_metrics),
                CANDIDATE_METHOD: _prediction_latency(candidate_metrics),
            },
        },
        "sam2_production_latency": sam2_latency,
        "material_decision": decision,
        "ood_evaluation": {
            "benchmark": "VideoPhy-1",
            "status": "deferred",
            "overall_verdict": "not_evaluable_ood_deferred",
        },
        "artifacts": {
            "core_file_count": len(CORE_REPORT_FILES),
            "core_files": list(CORE_REPORT_FILES),
            "merged_prediction_count": len(predictions),
        },
    }
    artifacts = {
        "merged_predictions.jsonl": merged_bytes,
        "paired_outcomes.json": _json_bytes(paired),
        "slices.json": _json_bytes(slices),
        "summary.json": _json_bytes(summary),
        "summary.md": _render_markdown(summary),
    }
    audit["report_output_sha256"] = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(artifacts.items())
    }
    audit["report_output_sha256_scope"] = {
        "hashed_count": len(artifacts),
        "excluded": ["artifact_audit.json"],
        "reason": "artifact_audit.json is self-referential",
    }
    return {"artifact_audit.json": _json_bytes(audit), **artifacts}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = _capture_input_snapshot(
        args.manifest,
        tuple(args.predictions),
        tuple(args.observation_meta_dir),
    )
    artifacts = _build_report_artifacts(
        snapshot=snapshot,
        output_dir=args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _verify_input_snapshot(snapshot)
    _publish_report_bundle(
        args.output_dir,
        artifacts,
        input_paths=snapshot.paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
