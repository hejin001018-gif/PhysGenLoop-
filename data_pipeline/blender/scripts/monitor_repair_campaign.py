"""Persist hourly Repair campaign progress and a detailed terminal summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
from zoneinfo import ZoneInfo

import yaml


SHANGHAI = ZoneInfo("Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 60:
        parser.error("interval-seconds must be at least 60")
    return args


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def snapshot(campaign: Path) -> dict[str, Any]:
    state = load_json(campaign / "campaign_state.json", {})
    config = state.get("config", {})
    shards = state.get("shards", {})
    complete = {name: item for name, item in shards.items() if item.get("status") == "complete"}
    running = [name for name, item in shards.items() if item.get("status") == "running"]
    failed = [name for name, item in shards.items() if item.get("status") == "failed"]
    admitted_groups = sum(int(item.get("group_count", 0)) for item in complete.values())
    admitted_records = sum(int(item.get("record_count", 0)) for item in complete.values())
    # A cumulative continuation campaign intentionally has no local render shards;
    # its merged/verified source manifests are the authoritative admitted dataset.
    if not shards and state.get("status") in {"generated", "trained", "cleaned"}:
        admitted_groups = int(state.get("group_count", 0) or 0)
        admitted_records = int(state.get("record_count", 0) or 0)
    current_complete_groups = 0
    current_group = None
    for shard_name in running:
        groups_dir = campaign / "shards" / shard_name / "groups"
        if not groups_dir.is_dir():
            continue
        group_dirs = sorted(path for path in groups_dir.glob("group_*") if path.is_dir())
        current_complete_groups += sum(
            (path / "group_complete.json").is_file() for path in group_dirs
        )
        incomplete = [path.name for path in group_dirs if not (path / "group_complete.json").is_file()]
        if incomplete:
            current_group = incomplete[-1]

    total_groups = int(config.get("total_groups", state.get("group_count", 0)) or 0)
    generated_groups = admitted_groups + current_complete_groups
    elapsed = [float(item["elapsed_sec"]) for item in complete.values() if item.get("elapsed_sec")]
    elapsed_groups = sum(int(item.get("group_count", 0)) for item in complete.values())
    seconds_per_group = sum(elapsed) / elapsed_groups if elapsed and elapsed_groups else None
    remaining_groups = max(0, total_groups - generated_groups)
    eta_seconds = seconds_per_group * remaining_groups if seconds_per_group is not None else None

    campaign_bytes = sum(
        path.stat().st_size for path in campaign.rglob("*") if path.is_file()
    ) if campaign.is_dir() else 0
    disk = shutil.disk_usage(campaign if campaign.exists() else campaign.parent)
    gpu_raw = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu = None
    if gpu_raw:
        parts = [item.strip() for item in gpu_raw.splitlines()[0].split(",")]
        if len(parts) == 5:
            gpu = {
                "name": parts[0],
                "memory_used_mib": int(parts[1]),
                "memory_total_mib": int(parts[2]),
                "utilization_percent": int(parts[3]),
                "temperature_c": int(parts[4]),
            }
    processes = command_output(["pgrep", "-af", "run_training_stage|run_repair_campaign|train_repair_campaign"])
    now = datetime.now(timezone.utc)
    return {
        "timestamp_utc": now.isoformat(),
        "timestamp_asia_shanghai": now.astimezone(SHANGHAI).isoformat(),
        "status": state.get("status", "missing"),
        "total_groups": total_groups,
        "generated_groups": generated_groups,
        "admitted_groups": admitted_groups,
        "admitted_records": admitted_records,
        "progress_percent": round(100.0 * generated_groups / total_groups, 2) if total_groups else 0.0,
        "completed_shards": len(complete),
        "running_shards": running,
        "failed_shards": failed,
        "current_shard_completed_groups": current_complete_groups,
        "current_group": current_group,
        "mean_seconds_per_group": None if seconds_per_group is None else round(seconds_per_group, 3),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds),
        "eta_human": format_duration(eta_seconds),
        "campaign_bytes": campaign_bytes,
        "disk_free_bytes": disk.free,
        "gpu": gpu,
        "pipeline_processes": [] if processes is None else processes.splitlines(),
        "release_exists": (campaign / "repair_agent" / "release_manifest.json").is_file(),
    }


def progress_markdown(data: dict[str, Any]) -> str:
    gpu = data.get("gpu") or {}
    running = ", ".join(data["running_shards"]) or "none"
    failed = ", ".join(data["failed_shards"]) or "none"
    return f"""# Repair Agent 小时进度

- 时间（Asia/Shanghai）：{data['timestamp_asia_shanghai']}
- Campaign 状态：{data['status']}
- 数据进度：{data['generated_groups']} / {data['total_groups']} 组（{data['progress_percent']}%）
- 已通过门禁：{data['admitted_groups']} 组 / {data['admitted_records']} 条样本
- 已完成 shard：{data['completed_shards']}
- 当前 shard：{running}；当前组：{data['current_group'] or 'none'}
- 失败 shard：{failed}
- 平均速度：{data['mean_seconds_per_group']} 秒/组
- 预计剩余：{data['eta_human']}
- Campaign 占用：{data['campaign_bytes'] / 1024 / 1024:.2f} MiB
- 磁盘可用：{data['disk_free_bytes'] / 1024 / 1024 / 1024:.2f} GiB
- GPU：{gpu.get('name', 'unknown')}；利用率 {gpu.get('utilization_percent', 'unknown')}%；显存 {gpu.get('memory_used_mib', 'unknown')} / {gpu.get('memory_total_mib', 'unknown')} MiB；温度 {gpu.get('temperature_c', 'unknown')}°C
- Release 已生成：{data['release_exists']}
"""


def final_summary(campaign: Path, last: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    state = load_json(campaign / "campaign_state.json", {})
    if state.get("status") != "cleaned":
        return None
    selection = load_json(campaign / str(state.get("selection_report", "")), {})
    cleanup = load_json(campaign / str(state.get("cleanup_receipt", "")), {})
    release_dir = campaign / str(state.get("release", "repair_agent"))
    release_manifest = load_json(release_dir / "release_manifest.json", {})
    feature_schema = load_json(release_dir / "feature_schema.json", {})
    critic_snapshot = load_json(release_dir / "critic_snapshot.json", {})
    winner_config_path = campaign / "training" / "winner_config.yaml"
    winner_config = (
        yaml.safe_load(winner_config_path.read_text(encoding="utf-8")) or {}
        if winner_config_path.is_file()
        else {}
    )
    artifact_dir = campaign.parents[1] / "artifacts"
    archive = artifact_dir / f"repair_agent_{campaign.name}.tar.gz"
    checksum_file = Path(str(archive) + ".sha256")
    if not archive.is_file() or not checksum_file.is_file():
        return None
    archive_sha = sha256(archive)
    expected_archive_sha = checksum_file.read_text(encoding="utf-8").split()[0]
    if archive_sha != expected_archive_sha:
        raise ValueError("final Repair Agent archive checksum mismatch")

    runs = selection.get("runs", [])
    winner = selection.get("winner", {})
    test = selection.get("held_out_test", {})
    dataset_audit = selection.get("dataset_audit", {})
    training_config = winner_config.get("training", {})
    agent_config = winner_config.get("agent", {})
    known_limitations = [
        "当前监督标签来自 Blender 配对正常轨迹与策略代理映射，并非部署环境中 Executor 的真实修复试验回报。",
        "当前 Release 提供 Selector/Policy 与 Repair Memory；Prompt、全局重生成、局部编辑和 Reject 的执行后端需由部署侧适配器实现。",
        "训练输入绑定冻结的 Physics Critic 配置；CriticReport schema 或类别语义变化时必须重新做兼容性评估。",
        "Blender 场景覆盖受控刚体下落/接触族，迁移到 HunyuanVideo 后仍需采集真实生成分布上的闭环反馈。",
    ]
    team_next_tasks = [
        "实现统一 RepairExecutor 接口及 prompt_generator、video_generator、local_video_editor、candidate_selector 四个适配器。",
        "把部署端每次尝试写成 RepairTrial，记录 before/after physics、semantic、quality 和 cost，替换代理标签。",
        "在 HunyuanVideo 小规模验证集上做 Critic→Selector→Executor→Critic 的端到端闭环回归。",
        "保持 critic_snapshot.json 与 feature_schema.json 的版本门禁，防止特征漂移和静默不兼容。",
        "依据 held-out test 的最低动作 F1 与混淆方向构建下一轮难例/类别重采样续训集。",
    ]
    payload = {
        "status": "complete",
        "campaign": campaign.name,
        "data": {
            "groups": state.get("group_count", last.get("total_groups")),
            "samples": state.get("record_count"),
            "completed_shards": len(state.get("shards", {})),
            "semantic_gate": "all admitted shards passed",
        },
        "model_selection": {
            "metric": selection.get("selection_metric"),
            "test_policy": selection.get("test_policy"),
            "split_seed": selection.get("split_seed"),
            "runs": runs,
            "winner": winner,
        },
        "dataset_audit": dataset_audit,
        "reproducibility": {
            "campaign_config": state.get("config", {}),
            "assigned_manifest_sha256": selection.get("manifest_sha256"),
            "split_seed": selection.get("split_seed"),
            "critic_snapshot": critic_snapshot,
        },
        "architecture": {
            "policy": "structured CriticReport feature encoder + PyTorch MLP action/gain heads",
            "feature_dimension": len(feature_schema.get("feature_names", [])),
            "feature_schema": feature_schema,
            "training_config": training_config,
            "agent_config": agent_config,
            "memory": "cosine retrieval over the same structured feature space",
            "executor_included": False,
        },
        "held_out_test": test,
        "memory": selection.get("memory", {}),
        "release": {
            "model_id": release_manifest.get("model_id"),
            "file_count": len(release_manifest.get("files", {})),
            "manifest_sha256": state.get("release_manifest_sha256"),
            "smoke": selection.get("release_smoke", {}),
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
            "files": release_manifest.get("files", {}),
        },
        "cleanup": cleanup,
        "known_limitations": known_limitations,
        "team_next_tasks": team_next_tasks,
        "completed_at_asia_shanghai": datetime.now(timezone.utc).astimezone(SHANGHAI).isoformat(),
    }
    run_lines = "\n".join(
        f"| {item.get('seed')} | {item.get('validation', {}).get('macro_f1')} | "
        f"{item.get('validation', {}).get('accuracy')} | {item.get('best_epoch')} |"
        for item in runs
    )
    per_class = test.get("per_class", {})
    class_lines = "\n".join(
        f"| {name} | {metrics.get('precision')} | {metrics.get('recall')} | "
        f"{metrics.get('f1')} | {metrics.get('support')} |"
        for name, metrics in per_class.items()
    )
    action_counts = dataset_audit.get("action_counts", {})
    split_counts = dataset_audit.get("split_counts", {})
    release_file_lines = "\n".join(
        f"| `{name}` | {metadata.get('bytes')} | `{metadata.get('sha256')}` |"
        for name, metadata in sorted(release_manifest.get("files", {}).items())
    )
    limitation_lines = "\n".join(f"- {item}" for item in known_limitations)
    task_lines = "\n".join(
        f"{index}. {item}" for index, item in enumerate(team_next_tasks, 1)
    )
    markdown = f"""# Repair Agent 训练最终总结

## 结果概览

- 数据：{payload['data']['groups']} 个场景组，{payload['data']['samples']} 条监督样本
- 数据质量：所有纳入训练的 shard 均通过 artifact audit 与 Critic 语义门禁
- 最佳随机种子：{winner.get('seed')}
- 最佳验证 macro-F1：{winner.get('validation', {}).get('macro_f1')}
- Held-out test macro-F1：{test.get('macro_f1')}
- Held-out test accuracy：{test.get('accuracy')}
- Repair Memory：{payload['memory'].get('sample_count')} 条，仅来自 train split
- 模型：{payload['release']['model_id']}

## 数据集与可复现性

- Action 分布：`{json.dumps(action_counts, ensure_ascii=False)}`
- Split 分布：`{json.dumps(split_counts, ensure_ascii=False)}`
- Group leakage：`{json.dumps(dataset_audit.get('group_leakage', {}), ensure_ascii=False)}`
- 固定 split seed：`{selection.get('split_seed')}`
- Assigned manifest SHA256：`{selection.get('manifest_sha256')}`
- Critic config SHA256：`{critic_snapshot.get('config_sha256')}`
- Critic model ID：`{critic_snapshot.get('critic_model_id')}`

Campaign 配置及生成器指纹：

```json
{json.dumps(state.get('config', {}), ensure_ascii=False, indent=2)}
```

## Repair Agent 架构

- 输入：冻结 Physics Critic 输出的结构化 `CriticReport`
- 特征维度：{len(feature_schema.get('feature_names', []))}
- Policy：结构化特征编码器 + MLP action head + expected-gain head
- MLP hidden dims：`{training_config.get('hidden_dims')}`；dropout：`{training_config.get('dropout')}`
- 动作空间：`{feature_schema.get('action_order')}`
- Memory：同一特征空间上的 cosine retrieval，推理融合权重 `{agent_config.get('memory_weight')}`
- Executor：不在本模型包中，由部署端四类执行适配器负责

## 五种子验证集对比

| Seed | Macro-F1 | Accuracy | Best epoch |
|---:|---:|---:|---:|
{run_lines}

## Held-out test 分类结果

| Repair action | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{class_lines}

混淆矩阵（action order: {test.get('action_order')}）：

```json
{json.dumps(test.get('confusion_matrix'), ensure_ascii=False)}
```

## Release 与清理

- Release smoke：{payload['release']['smoke'].get('valid')}
- Release 文件数：{payload['release']['file_count']}
- Release manifest SHA256：`{payload['release']['manifest_sha256']}`
- 部署压缩包：`{payload['release']['archive']}`
- 压缩包 SHA256：`{payload['release']['archive_sha256']}`
- 已删除 Blender shards：{cleanup.get('deleted_bytes', 0) / 1024 / 1024:.2f} MiB

Release 文件清单：

| File | Bytes | SHA256 |
|---|---:|---|
{release_file_lines}

## 部署契约

```text
CriticReport + RepairContext
              ↓
LearningRepairAgent (Policy + Memory)
              ↓
Prompt Repair | Global Regeneration | Local Editing | Reject
              ↓
Deployment RepairExecutor adapter
```

推理入口：`python inference.py --critic-report critic_report.json --device cuda`

## 已知限制

{limitation_lines}

## 团队下一步任务

{task_lines}

最终部署包不依赖 Blender；保留 Physics Critic、Repair Agent 代码与该模型包即可进行策略推理。
"""
    return payload, markdown


def main() -> int:
    args = parse_args()
    campaign = args.campaign_root.resolve()
    monitor_dir = campaign / "monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    history_path = monitor_dir / "hourly_progress.jsonl"
    while True:
        data = snapshot(campaign)
        atomic_json(monitor_dir / "latest_progress.json", data)
        atomic_text(monitor_dir / "latest_progress.md", progress_markdown(data))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(
            f"{data['timestamp_asia_shanghai']} status={data['status']} "
            f"progress={data['progress_percent']}% eta={data['eta_human']}",
            flush=True,
        )
        summary = final_summary(campaign, data)
        if summary is not None:
            payload, markdown = summary
            atomic_json(monitor_dir / "final_training_summary.json", payload)
            atomic_text(monitor_dir / "final_training_summary.md", markdown)
            print("Final training summary verified and written.", flush=True)
            return 0
        if data["failed_shards"]:
            atomic_text(
                monitor_dir / "monitor_failure.txt",
                f"Campaign stopped with failed shards: {data['failed_shards']}\n",
            )
            return 2
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
