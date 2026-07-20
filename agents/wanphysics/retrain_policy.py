"""训练入口：从 trials.jsonl 重新训练 Action-Value Policy。

消费 run_trial_campaign.py 产出的 trials.jsonl（RepairTrialV1 格式），
调用已有的 train_policy() 函数训练并写出新 checkpoint。

用法：
  python agents/wanphysics/retrain_policy.py \\
    --manifest path/to/trials.jsonl \\
    --output-dir checkpoints/repair_agent/repair-agent-vX.Y-actual-YYYYMMDD \\
    [--epochs 80] \\
    [--device cuda]

输出目录结构（与 train_policy 一致）：
  model/best_action_value_policy.pt
  config/feature_config.json
  audit/split_report.json
  audit/eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/PhysGenLoop-")
sys.path.insert(0, "/root/PhysGenLoop-/src")

from physgenloop.learning_repair.training import TrainConfig, train_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="从 trials.jsonl 重新训练 Repair Policy")
    parser.add_argument("--manifest", required=True, help="trials.jsonl 路径（RepairExample 格式）")
    parser.add_argument("--output-dir", required=True, help="输出 checkpoint 目录")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--successful-only", action="store_true", default=True,
                        help="只使用 successful=True 的 Trial（默认开启）")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[retrain] ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        successful_only=args.successful_only,
    )

    print(f"[retrain] manifest: {manifest_path}")
    print(f"[retrain] output:   {output_dir}")
    print(f"[retrain] config:   epochs={config.epochs}, device={config.device}")

    metrics = train_policy(
        manifest_path=manifest_path,
        output_dir=output_dir,
        config=config,
    )

    print("[retrain] training complete")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
