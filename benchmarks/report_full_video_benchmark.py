"""Generate the deterministic, audit-ready full VideoPhy-2 report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
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


def _failure_reason(failure: Mapping[str, Any]) -> str:
    for key in ("reason", "message", "type", "error"):
        value = failure.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return json.dumps(
        failure,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        return os.path.samefile(first, second)
    return False


def _stage_bytes(path: Path, content: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    staged = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _replace_staged_file(staged: Path, destination: Path) -> None:
    """Single publication seam used by the bundle transaction."""

    os.replace(staged, destination)


def _restore_file(path: Path, original: bytes | None) -> None:
    if original is None:
        path.unlink(missing_ok=True)
        return
    staged = _stage_bytes(path, original)
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _publish_report_bundle(
    output_dir: Path,
    artifacts: Mapping[str, bytes],
    *,
    input_paths: Sequence[Path],
) -> None:
    """Publish the six report files as one rollback-protected transaction."""

    if tuple(sorted(artifacts)) != CORE_REPORT_FILES:
        raise ValueError(
            f"report bundle must contain exactly {CORE_REPORT_FILES!r}"
        )
    destinations = {
        name: output_dir / name for name in CORE_REPORT_FILES
    }
    for destination in destinations.values():
        for input_path in input_paths:
            if _paths_alias(destination, input_path):
                raise ValueError(
                    f"report destination aliases an input: {input_path}"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    originals = {
        name: destination.read_bytes() if destination.exists() else None
        for name, destination in destinations.items()
    }
    staged: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for name in CORE_REPORT_FILES:
            staged[name] = _stage_bytes(destinations[name], artifacts[name])
        for name in CORE_REPORT_FILES:
            _replace_staged_file(staged[name], destinations[name])
            replaced.append(name)
    except BaseException:
        for name in reversed(replaced):
            _restore_file(destinations[name], originals[name])
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


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
            f"- {item['sample_id']} / {item['method_id']}: {item['reason']}"
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


def _build_report_artifacts(
    *,
    manifest: Path,
    prediction_paths: Sequence[Path],
    observation_meta_dirs: Sequence[Path],
    output_dir: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, bytes]:
    if bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    samples = tuple(sorted(load_manifest(manifest), key=lambda item: item.sample_id))

    with tempfile.TemporaryDirectory(prefix="pavg-full-report-") as raw_temp:
        temporary = Path(raw_temp)
        merge_result = merge_prediction_shards(
            samples,
            prediction_paths,
            (BASELINE_METHOD, CANDIDATE_METHOD),
            temporary / "merged_predictions.jsonl",
            audit_path=temporary / "artifact_audit.json",
        )
        merged_bytes = (temporary / "merged_predictions.jsonl").read_bytes()
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
    sam2_latency = summarize_observation_latencies(samples, observation_meta_dirs)
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
        "path": str(manifest),
        "name": manifest.name,
        "sha256": _file_sha256(manifest),
    }
    observation_files = sorted(
        (
            path
            for directory in observation_meta_dirs
            for path in directory.rglob("*.meta.json")
        ),
        key=lambda path: os.path.normcase(str(path.resolve(strict=False))),
    )
    audit["observation_metadata"] = {
        "directory_count": len(observation_meta_dirs),
        "file_count": len(observation_files),
        "files": [
            {
                "path": str(path),
                "name": path.name,
                "sha256": _file_sha256(path),
            }
            for path in observation_files
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
    return {"artifact_audit.json": _json_bytes(audit), **artifacts}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = _build_report_artifacts(
        manifest=args.manifest,
        prediction_paths=tuple(args.predictions),
        observation_meta_dirs=tuple(args.observation_meta_dir),
        output_dir=args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _publish_report_bundle(
        args.output_dir,
        artifacts,
        input_paths=(args.manifest, *args.predictions),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
