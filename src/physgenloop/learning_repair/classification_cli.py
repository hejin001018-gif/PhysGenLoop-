"""Learning Repair Agent 的数据、训练、评估和推理命令。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .agent import LearningRepairAgent
from .contracts import RepairContext
from .dataset import (
    audit_dataset,
    collect_repair_samples,
    grouped_split,
    load_repair_manifest,
    select_split,
    write_repair_manifest,
)
from .memory import RepairMemory
from .policy import HeuristicRepairPolicy, TorchMLPRepairPolicy
from .release import export_release
from .training import evaluate_policy, load_train_config, train_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pavg-repair",
        description="Validate data, train, evaluate, and run the Learning Repair Agent.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Audit a repair manifest.")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--check-artifacts", action="store_true")
    validate.add_argument("--base-dir", type=Path)

    collect = commands.add_parser(
        "collect", help="Collect Blender repair_sample.json records into JSONL."
    )
    collect.add_argument("--root", required=True, type=Path)
    collect.add_argument("--output", required=True, type=Path)
    collect.add_argument("--record-name", default="repair_sample.json")

    split = commands.add_parser("split", help="Create leakage-safe group splits.")
    split.add_argument("--manifest", required=True, type=Path)
    split.add_argument("--output-dir", required=True, type=Path)
    split.add_argument("--validation-fraction", type=float, default=0.1)
    split.add_argument("--test-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Train a PyTorch Repair Policy.")
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--config", type=Path)

    export = commands.add_parser(
        "export", help="Export a Blender-free deployable Repair Agent bundle."
    )
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--config", type=Path)
    export.add_argument("--memory", type=Path)
    export.add_argument("--critic-config", type=Path)
    export.add_argument("--critic-model-id")
    export.add_argument("--overwrite", action="store_true")

    evaluate = commands.add_parser("evaluate", help="Evaluate a policy checkpoint.")
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="test"
    )
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output", type=Path)

    predict = commands.add_parser("predict", help="Choose a repair action for one report.")
    predict.add_argument("--critic-report", required=True, type=Path)
    predict.add_argument("--checkpoint", type=Path)
    predict.add_argument("--memory", type=Path)
    predict.add_argument("--prompt", default="")
    predict.add_argument("--context", type=Path)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--output", type=Path)
    return parser


def _write_or_print(payload: object, path: Path | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        audit = audit_dataset(
            load_repair_manifest(args.manifest),
            check_artifacts=args.check_artifacts,
            base_dir=args.base_dir or args.manifest.parent,
        )
        _write_or_print(audit.to_dict())
        return 0 if audit.valid else 2

    if args.command == "collect":
        records = collect_repair_samples(args.root, record_name=args.record_name)
        write_repair_manifest(records, args.output)
        _write_or_print(audit_dataset(records).to_dict())
        return 0

    if args.command == "split":
        records = grouped_split(
            load_repair_manifest(args.manifest),
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train", "validation", "test"):
            write_repair_manifest(
                select_split(records, name), args.output_dir / f"{name}.jsonl"
            )
        audit = audit_dataset(records)
        _write_or_print(audit.to_dict(), args.output_dir / "split_audit.json")
        return 0

    if args.command == "train":
        report = train_policy(
            args.manifest,
            args.output_dir,
            config=load_train_config(args.config),
        )
        _write_or_print(report)
        return 0

    if args.command == "export":
        destination = export_release(
            args.checkpoint,
            args.output_dir,
            config_path=args.config,
            memory_path=args.memory,
            critic_config_path=args.critic_config,
            critic_model_id=args.critic_model_id,
            overwrite=args.overwrite,
        )
        _write_or_print({"release_dir": str(destination), "status": "complete"})
        return 0

    if args.command == "evaluate":
        policy = (
            TorchMLPRepairPolicy.load(args.checkpoint, device=args.device)
            if args.checkpoint
            else HeuristicRepairPolicy()
        )
        records = load_repair_manifest(args.manifest)
        if args.split != "all":
            records = select_split(records, args.split)
        if not records:
            raise ValueError(f"selected split {args.split!r} contains no samples")
        metrics = evaluate_policy(policy, records)
        _write_or_print(metrics, args.output)
        return 0

    raw = json.loads(args.critic_report.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("report"), dict):
        raw = raw["report"]
    policy = (
        TorchMLPRepairPolicy.load(args.checkpoint, device=args.device)
        if args.checkpoint
        else HeuristicRepairPolicy()
    )
    memory = (
        RepairMemory.from_manifest(
            args.memory, encoder=getattr(policy, "encoder", None)
        )
        if args.memory
        else None
    )
    context = (
        RepairContext.from_dict(
            json.loads(args.context.read_text(encoding="utf-8"))
        )
        if args.context
        else RepairContext()
    )
    decision = LearningRepairAgent(policy, memory=memory).decide(
        critic_report=raw, prompt=args.prompt, context=context
    )
    _write_or_print(decision.to_dict(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
