"""Train, select, test, and export a Repair Agent from a generated campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import yaml

from physgenloop.learning_repair import (
    AgentConfig,
    LearningRepairAgent,
    RepairMemory,
    TorchMLPRepairPolicy,
    audit_dataset,
    evaluate_policy,
    export_release,
    grouped_split,
    load_repair_manifest,
    load_train_config,
    select_split,
    train_policy,
    write_repair_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--config", default="configs/repair_agent.yaml", type=Path)
    parser.add_argument("--critic-config", default="configs/default.yaml", type=Path)
    parser.add_argument("--critic-model-id", default="pavg-critic-0.3.0/configs-default")
    parser.add_argument("--seeds", default="17,23,42,73,101")
    parser.add_argument("--split-seed", type=int, default=20260716)
    parser.add_argument("--memory-size", type=int, default=512)
    args = parser.parse_args()
    args.seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be a non-empty unique comma-separated list")
    if args.memory_size < 4:
        parser.error("memory-size must be at least four")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def observed_signature(example) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(item.get("category", ""))
                for item in example.critic_report.get("violations", ())
                if isinstance(item, dict)
            }
        )
    )


def curate_memory(records: Iterable, limit: int):
    """Round-robin semantic buckets instead of shipping thousands of duplicates."""

    buckets = defaultdict(list)
    for item in records:
        key = (
            item.target_action.value,
            str(item.metadata.get("category", "unknown")),
            int(item.metadata.get("severity", 0)),
            str(item.critic_report.get("decision", "unknown")),
            observed_signature(item),
        )
        buckets[key].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.sample_id)
    selected = []
    keys = sorted(buckets, key=str)
    cursor = {key: 0 for key in keys}
    while len(selected) < limit:
        progressed = False
        for key in keys:
            index = cursor[key]
            bucket = buckets[key]
            if index >= len(bucket):
                continue
            # Walk each bucket from a different deterministic offset so adjacent
            # campaign groups do not dominate the compact episodic memory.
            selected.append(bucket[index])
            cursor[key] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    selected.sort(key=lambda item: item.sample_id)
    return tuple(selected)


def verify_release(release: Path) -> dict[str, Any]:
    manifest_path = release / "release_manifest.json"
    manifest = load_json(manifest_path)
    failures = []
    for relative, expected in manifest["files"].items():
        path = release / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(path) != expected["sha256"]:
            failures.append(f"checksum:{relative}")
        elif path.stat().st_size != expected["bytes"]:
            failures.append(f"size:{relative}")
    if failures:
        raise ValueError(f"release verification failed: {failures}")
    return {
        "manifest_sha256": sha256(manifest_path),
        "file_count": len(manifest["files"]),
        "model_id": manifest.get("model_id"),
    }


def release_smoke(release: Path, examples: Iterable, output_dir: Path) -> dict[str, Any]:
    """Exercise the exported entrypoint on one validation example per action."""

    representatives = {}
    for item in examples:
        representatives.setdefault(item.target_action.value, item)
    results = []
    for action, item in sorted(representatives.items()):
        input_path = output_dir / f"{action}.critic_report.json"
        context_path = output_dir / f"{action}.context.json"
        output_path = output_dir / f"{action}.decision.json"
        atomic_json(input_path, item.critic_report)
        # Action targets can depend on repair budget and previous attempts.  The
        # exported-entrypoint smoke must preserve the example's RepairContext;
        # otherwise a terminal multi-corruption sample labeled Reject is silently
        # evaluated as a first-attempt sample and can correctly choose Local Editing.
        atomic_json(context_path, item.context.to_dict())
        result = subprocess.run(
            [
                sys.executable,
                str(release / "inference.py"),
                "--critic-report",
                str(input_path),
                "--context",
                str(context_path),
                "--device",
                "cpu",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        predicted = None
        if result.returncode == 0 and output_path.is_file():
            predicted = load_json(output_path).get("action")
        runtime_valid = result.returncode == 0 and predicted in representatives
        results.append(
            {
                "sample_id": item.sample_id,
                "expected_action": action,
                "predicted_action": predicted,
                "returncode": result.returncode,
                "valid": runtime_valid,
                "matches_expected": predicted == action,
                "stderr": result.stderr[-500:],
            }
        )
    valid = (
        len(results) == 4
        and all(item["valid"] for item in results)
        and all(item["matches_expected"] for item in results)
    )
    report = {"valid": valid, "checked_actions": len(results), "results": results}
    atomic_json(output_dir / "release_smoke_report.json", report)
    if not valid:
        raise ValueError("exported release smoke test failed")
    return report


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    campaign = args.campaign_root.resolve()
    state_path = campaign / "campaign_state.json"
    state = load_json(state_path)
    manifest = campaign / str(state.get("merged_manifest", "campaign_manifest.jsonl"))
    if state.get("status") not in {"generated", "trained"}:
        raise ValueError(f"campaign is not ready for training: {state.get('status')!r}")
    if not manifest.is_file() or sha256(manifest) != state.get("merged_manifest_sha256"):
        raise ValueError("merged campaign manifest failed checksum verification")

    training_root = campaign / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    raw_records = load_repair_manifest(manifest)
    assigned = grouped_split(
        raw_records,
        validation_fraction=0.1,
        test_fraction=0.1,
        seed=args.split_seed,
    )
    assigned_manifest = training_root / "assigned_manifest.jsonl"
    write_repair_manifest(assigned, assigned_manifest)
    audit = audit_dataset(assigned)
    if not audit.valid or audit.group_leakage:
        raise ValueError(f"assigned dataset failed audit: {audit.to_dict()}")

    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    critic_config = (
        (repo / args.critic_config).resolve()
        if not args.critic_config.is_absolute()
        else args.critic_config
    )
    base_config = load_train_config(config_path)
    run_summaries = []
    for seed in args.seeds:
        run_dir = training_root / "seeds" / f"seed_{seed}"
        config = replace(base_config, seed=seed, evaluate_test=False)
        print(f"TRAIN seed={seed}", flush=True)
        report = train_policy(assigned_manifest, run_dir, config=config)
        if report.get("test") is not None:
            raise AssertionError("test metrics were exposed during model selection")
        run_summaries.append(
            {
                "seed": seed,
                "checkpoint": str((run_dir / "best_policy.pt").relative_to(campaign)),
                "best_epoch": report["best_epoch"],
                "epochs_completed": report["epochs_completed"],
                "validation": report["validation"],
            }
        )

    winner = max(
        run_summaries,
        key=lambda item: (
            float(item["validation"]["macro_f1"]),
            float(item["validation"]["accuracy"]),
            -float(item["validation"]["gain_mae"]),
            -int(item["seed"]),
        ),
    )
    winner_checkpoint = campaign / winner["checkpoint"]
    winner_policy = TorchMLPRepairPolicy.load(winner_checkpoint, device="auto")
    test_records = select_split(assigned, "test")
    if not test_records:
        raise ValueError("held-out test split is empty")

    train_records = select_split(assigned, "train")
    # Blender videos are intentionally deleted after a verified release.  Shipping
    # their soon-to-be-stale relative paths inside episodic memory would create a
    # misleading deployment contract, while retrieval itself only needs structured
    # report/context/outcome fields.
    memory_records = tuple(
        replace(
            item,
            artifacts={},
            metadata={**item.metadata, "artifacts_retention": "omitted_from_release_memory"},
        )
        for item in curate_memory(train_records, args.memory_size)
    )
    memory_path = training_root / "repair_memory.jsonl"
    write_repair_manifest(memory_records, memory_path)
    memory_audit = audit_dataset(memory_records)
    if not memory_audit.valid:
        raise ValueError(f"curated Repair Memory failed audit: {memory_audit.to_dict()}")

    raw_release_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    agent_config = AgentConfig(**dict(raw_release_config.get("agent", {})))
    memory = RepairMemory(
        memory_records,
        encoder=winner_policy.encoder,
    )
    selected_agent = LearningRepairAgent(
        winner_policy,
        memory=memory,
        config=agent_config,
    )

    class SelectedAgentAdapter:
        def predict(self, critic_report, context=None):
            return selected_agent.decide(
                critic_report=critic_report,
                context=context,
            )

    # This is the only held-out test evaluation in the campaign, and it measures
    # the actual exported policy+memory decision path rather than the bare MLP.
    test_metrics = evaluate_policy(SelectedAgentAdapter(), test_records)

    winning_config = replace(base_config, seed=int(winner["seed"]), evaluate_test=False)
    serialized_training = asdict(winning_config)
    serialized_training["hidden_dims"] = list(serialized_training["hidden_dims"])
    raw_release_config["training"] = serialized_training
    winner_config_path = training_root / "winner_config.yaml"
    winner_config_path.write_text(
        yaml.safe_dump(raw_release_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    release = campaign / "repair_agent"
    export_release(
        winner_checkpoint,
        release,
        config_path=winner_config_path,
        memory_path=memory_path,
        critic_config_path=critic_config,
        critic_model_id=args.critic_model_id,
        overwrite=True,
    )
    release_verification = verify_release(release)
    smoke = release_smoke(
        release,
        select_split(assigned, "validation"),
        training_root / "release_smoke",
    )

    selection_report = {
        "selection_metric": "validation.macro_f1",
        "test_policy": "winner_only_once",
        "split_seed": args.split_seed,
        "manifest_sha256": sha256(assigned_manifest),
        "dataset_audit": audit.to_dict(),
        "runs": run_summaries,
        "winner": winner,
        "held_out_test": test_metrics,
        "memory": {
            "sample_count": len(memory_records),
            "source_split": "train",
            "audit": memory_audit.to_dict(),
        },
        "release": release_verification,
        "release_smoke": smoke,
    }
    selection_path = training_root / "selection_report.json"
    atomic_json(selection_path, selection_report)
    state.update(
        {
            "status": "trained",
            "training_completed_at_unix": time.time(),
            "selected_seed": winner["seed"],
            "selection_report": str(selection_path.relative_to(campaign)),
            "selection_report_sha256": sha256(selection_path),
            "release": str(release.relative_to(campaign)),
            "release_manifest_sha256": release_verification["manifest_sha256"],
        }
    )
    atomic_json(state_path, state)
    print(
        json.dumps(
            {
                "status": "trained",
                "selected_seed": winner["seed"],
                "validation_macro_f1": winner["validation"]["macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "release": str(release),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
