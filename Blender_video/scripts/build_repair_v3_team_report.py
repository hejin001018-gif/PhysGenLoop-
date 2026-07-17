"""Build a readable Markdown/JSON handoff report for Repair Agent v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True, type=Path)
    parser.add_argument("--cleanup-receipt", action="append", default=[], type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.campaign_root.resolve()
    source = load(root / "sources" / "source_index.json")
    adaptation = load(root / "proxy_adaptation.json")
    audit = load(root / "target_audit.json")
    compatibility = load(root / "compatibility.json")
    training = load(root / "training" / "training_report.json")
    evaluation = load(root / "evaluation_test.json")
    smoke = load(root / "release_smoke.json")
    release_root = root / "repair_agent"
    release_manifest_path = release_root / "release_manifest.json"
    release = load(release_manifest_path)
    cleanup = [load(path.resolve()) for path in args.cleanup_receipt]

    release_failures = []
    for relative, expected in release["files"].items():
        path = release_root / relative
        if not path.is_file():
            release_failures.append(f"missing:{relative}")
        elif path.stat().st_size != expected["bytes"]:
            release_failures.append(f"size:{relative}")
        elif sha256(path) != expected["sha256"]:
            release_failures.append(f"sha256:{relative}")
    archive = args.archive.resolve()
    expected_archive = args.archive_sha256.read_text(encoding="utf-8").split()[0]
    actual_archive = sha256(archive)
    requirements = {
        "source_groups_1200": source.get("group_count") == 1200,
        "source_samples_22200": source.get("sample_count") == 22200,
        "target_audit_valid": bool(audit.get("valid")),
        "no_group_leakage": not audit.get("group_leakage"),
        "proxy_labels_explicit": audit.get("proxy_label_count") == 22200,
        "actual_trials_not_fabricated": audit.get("actual_trial_label_count") == 0,
        "compatibility_valid": bool(compatibility.get("valid")),
        "held_out_test_present": bool(training.get("held_out_test")),
        "release_smoke_four_actions": bool(smoke.get("valid")) and smoke.get("checked") == 4,
        "release_files_valid": not release_failures,
        "archive_sha256_valid": actual_archive == expected_archive,
        "cleanup_receipts_present": len(cleanup) == 2,
        "cleanup_complete": len(cleanup) == 2 and all(item.get("status") == "cleaned" for item in cleanup),
    }
    status = "complete" if all(requirements.values()) else "incomplete"
    payload = {
        "schema_version": "repair-v3-team-report/1.0",
        "status": status,
        "requirements": requirements,
        "data": {
            "groups": source.get("group_count"),
            "samples": source.get("sample_count"),
            "source_count": source.get("source_count"),
            "action_counts": adaptation.get("action_counts"),
            "label_type": adaptation.get("label_type"),
            "proxy_label_count": audit.get("proxy_label_count"),
            "actual_trial_label_count": audit.get("actual_trial_label_count"),
            "group_leakage": audit.get("group_leakage"),
            "source_index_sha256": sha256(root / "sources" / "source_index.json"),
            "targets_sha256": sha256(root / "proxy_targets.jsonl"),
        },
        "architecture": {
            "policy_format": training.get("format_version"),
            "model_id": training.get("model_id"),
            "selection_mode": training.get("label_provenance", {}).get("selection_mode"),
            "compatibility_id": compatibility.get("compatibility_id"),
            "executor_contract": "ExecutorRegistry with capability masking",
            "memory": "curated proxy target memory; actual Trial memory pending",
        },
        "training": training,
        "evaluation": evaluation,
        "release": {
            "manifest": release,
            "manifest_sha256": sha256(release_manifest_path),
            "verification_failures": release_failures,
            "smoke": smoke,
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": actual_archive,
        },
        "cleanup": cleanup,
        "recovery": {
            "incident": "hard-v1 multi_corrupt could leave the target out of frame and expose only disappearance",
            "rejected_samples": ["group_000957--multi_corrupt", "group_001155--multi_corrupt"],
            "fix": "hard-v1.1 keeps multi_corrupt in frame and preserves penetration plus disappearance evidence",
            "gate_was_not_weakened": True,
        },
        "limitations": adaptation.get("limitations", []),
        "team_next_tasks": [
            "Connect real Prompt/Global/Local Executor backends and collect RepairTrialV1 records.",
            "Keep proxy and actual-Trial metrics separated; do not claim Hunyuan success from Blender tests.",
            "Replace source_revision=unknown with a clean reviewed revision before deployment promotion.",
            "Run Hunyuan calibration/test as frozen, disjoint campaigns before enabling value-led selection.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    winner = training.get("winner", {})
    test = training.get("held_out_test", {})
    seed_lines = "\n".join(
        f"| {item.get('seed')} | {item.get('validation', {}).get('macro_f1', 0):.6f} | "
        f"{item.get('validation', {}).get('balanced_accuracy', 0):.6f} | "
        f"{item.get('validation', {}).get('value_mae', 0):.6f} | {item.get('best_epoch')} |"
        for item in training.get("seeds", [])
    )
    action_lines = "\n".join(
        f"| {name} | {metrics.get('precision', 0):.6f} | {metrics.get('recall', 0):.6f} | "
        f"{metrics.get('f1', 0):.6f} | {metrics.get('support', 0)} |"
        for name, metrics in test.get("per_action", {}).items()
    )
    method_lines = "\n".join(
        f"| {name} | {metrics.get('accuracy', 0):.6f} | {metrics.get('macro_f1', 0):.6f} | "
        f"{metrics.get('balanced_accuracy', 0):.6f} | {metrics.get('mean_regret', 0):.6f} |"
        for name, metrics in evaluation.get("overall", {}).items()
    )
    requirement_lines = "\n".join(
        f"- {'✅' if value else '❌'} `{name}`" for name, value in requirements.items()
    )
    cleanup_lines = "\n".join(
        f"- {row.get('campaign')}: {row.get('deleted_bytes', 0)} bytes"
        for receipt in cleanup
        for row in receipt.get("campaigns", [])
    ) or "- 尚无清理回执"
    markdown = f"""# Repair Agent v3 Action-Value 训练与交付报告

## 结论

- 状态：**{status}**
- 数据：{source.get('group_count')} 个场景组，{source.get('sample_count')} 条监督样本
- 模型：`{training.get('model_id')}`
- 最佳随机种子：{winner.get('seed')}
- Held-out test macro-F1：{test.get('macro_f1')}
- Held-out test balanced accuracy：{test.get('balanced_accuracy')}
- 选择模式：`{training.get('label_provenance', {}).get('selection_mode')}`
- Actual Executor Trial：0；当前结果严格标记为 Blender proxy 训练

## 完成门禁

{requirement_lines}

## 数据与来源

- 普通数据：900组、11,700条
- 困难数据：300组、10,500条
- 总计：1,200组、22,200条
- Group leakage：`{json.dumps(audit.get('group_leakage', {}), ensure_ascii=False)}`
- 动作分布：`{json.dumps(adaptation.get('action_counts', {}), ensure_ascii=False)}`
- Targets SHA256：`{payload['data']['targets_sha256']}`

`hard-v1` 曾有两个 multi-corrupt 样本因目标离开画面而只暴露消失错误。门禁没有放宽；使用
`hard-v1.1` 重新生成后，两者均稳定暴露穿透、消失和轨迹异常，再纳入训练。

## 架构合并

```text
Frozen CriticReport + RepairContext
            ↓
Action-Value Policy（四动作分类头 + 四动作 value 头）
            ↓
Capability Mask / provenance-aware selection
            ↓
ExecutorRegistry
            ↓
Prompt | Global | Local | Reject
```

当前只有 selected-action proxy reward；未执行动作保持 null，不伪造成失败 Trial。因此纯 proxy
checkpoint 使用分类概率选择；收集真实多动作 `RepairTrialV1` 后才启用 value 主导。

## 五种子验证

| Seed | Macro-F1 | Balanced accuracy | Value MAE | Best epoch |
|---:|---:|---:|---:|---:|
{seed_lines}

## Held-out test 分动作结果

| Action | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{action_lines}

## R0–R4 独立评估

| Method | Accuracy | Macro-F1 | Balanced accuracy | Mean regret |
|---|---:|---:|---:|---:|
{method_lines}

## Release

- Release manifest SHA256：`{payload['release']['manifest_sha256']}`
- 部署包：`{archive}`
- 部署包 SHA256：`{actual_archive}`
- 四动作 inference smoke：`{smoke.get('valid')}`
- 文件数：{len(release.get('files', {}))}

## Blender shards 清理

{cleanup_lines}

## 已知边界

- 当前数据为 Blender proxy 标签，不是实际 Executor 闭环回报。
- 当前结果不能解释为 HunyuanVideo 修复成功率。
- `source_revision=unknown`，在干净且经过团队评审的 revision 前不得提升为正式部署版本。
- Release 已包含 Executor-facing Action-Value Policy，但真实 Prompt/Global/Local 后端仍需部署侧注入。

## 团队下一步

1. 接入真实 Prompt、Global、Local Executor，生成 `RepairTrialV1`。
2. 将失败动作作为真实负 utility 纳入 Memory；未执行动作继续保持未知。
3. 构建严格分离的 Hunyuan calibration/test campaign。
4. 在真实 Trial 数据上重新训练，切换为 value-led selection。
"""
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": status, "json": str(args.output_json), "markdown": str(args.output_md)}, ensure_ascii=False))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
