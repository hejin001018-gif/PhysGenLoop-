"""Action-Value and Actual-Trial commands used by the canonical CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .baselines import CategoryOnlyPolicy, HeuristicDecisionPolicy
from .campaign import RewardSpec, targets_from_trials
from .cloud_campaign import load_backend_factory, run_frozen_campaign
from .compatibility import CompatibilityManifest, verify_proxy_baseline
from .evaluation import audit_targets, closed_loop_metrics, compare_policies
from .manifests import FrozenCampaignManifest, validate_campaign_artifacts
from .memory_policy import ActualTrialMemory, BlendedValuePredictor, MemoryValuePredictor
from .recording import read_targets, read_trials, write_targets
from .proxy_adapter import adapt_proxy_manifests, write_adaptation_report
from .review import build_integration_review, write_integration_review
from .value_policy import (
    ActionValueDecisionPolicy,
    TorchActionValuePolicy,
)
from .value_training import ValueTrainConfig, train_action_value_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pavg-repair",
        description="Learning Repair Action-Value and Actual-Trial tools.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("verify-baseline")
    baseline.add_argument("--manifest", required=True, type=Path)
    baseline.add_argument("--root", default=Path("."), type=Path)
    baseline.add_argument("--output", type=Path)

    compatibility = commands.add_parser("check-compatibility")
    compatibility.add_argument("--manifest", required=True, type=Path)
    compatibility.add_argument("--critic-config", required=True, type=Path)
    compatibility.add_argument("--critic-schema", required=True, type=Path)
    compatibility.add_argument("--feature-schema", required=True, type=Path)
    compatibility.add_argument("--critic-report", type=Path)
    compatibility.add_argument("--output", type=Path)

    campaign = commands.add_parser("validate-campaign")
    campaign.add_argument("--manifest", required=True, type=Path)
    campaign.add_argument("--base-dir", default=Path("."), type=Path)
    campaign.add_argument("--output", type=Path)

    run_campaign = commands.add_parser("run-campaign")
    run_campaign.add_argument("--manifest", required=True, type=Path)
    run_campaign.add_argument("--backend-factory", required=True)
    run_campaign.add_argument("--compatibility", required=True, type=Path)
    run_campaign.add_argument("--trials-output", required=True, type=Path)
    run_campaign.add_argument("--targets-output", required=True, type=Path)
    run_campaign.add_argument("--reward-config", type=Path)
    run_campaign.add_argument("--output", type=Path)

    build = commands.add_parser("build-targets")
    build.add_argument("--trials", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--reward-config", type=Path)

    audit = commands.add_parser("audit-targets")
    audit.add_argument("--targets", required=True, type=Path)
    audit.add_argument("--output", type=Path)

    train = commands.add_parser("train-values")
    train.add_argument("--targets", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--compatibility", required=True, type=Path)
    train.add_argument("--config", type=Path)

    adapt = commands.add_parser("adapt-proxy-targets")
    adapt.add_argument("--manifest", action="append", required=True, type=Path)
    adapt.add_argument("--output", required=True, type=Path)
    adapt.add_argument("--report", required=True, type=Path)
    adapt.add_argument("--reward-config", type=Path)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--targets", required=True, type=Path)
    evaluate.add_argument("--compatibility", required=True, type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--memory-targets", type=Path)
    evaluate.add_argument("--hunyuan-calibration-targets", type=Path)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output", type=Path)

    closed = commands.add_parser("closed-loop-report")
    closed.add_argument("--trials", required=True, type=Path)
    closed.add_argument("--output", type=Path)

    review = commands.add_parser("integration-review")
    review.add_argument("--compatibility", required=True, type=Path)
    review.add_argument("--executor-manifest", required=True, type=Path)
    review.add_argument("--baseline-evidence", required=True, type=Path)
    review.add_argument("--test-evidence", required=True, type=Path)
    review.add_argument("--evaluation-evidence", type=Path)
    review.add_argument("--output-dir", required=True, type=Path)
    return parser


def _load_object(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain an object")
    return dict(raw)


def _reward(path: Path | None) -> RewardSpec:
    if path is None:
        return RewardSpec()
    raw = _load_object(path)
    raw = dict(raw.get("reward", raw))
    return RewardSpec(**raw)


def _emit(payload: Any, path: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-baseline":
        _emit(
            verify_proxy_baseline(args.manifest, root=args.root),
            args.output,
        )
        return 0
    if args.command == "check-compatibility":
        manifest = CompatibilityManifest.load(args.manifest)
        manifest.verify_files(
            critic_config=args.critic_config,
            critic_schema=args.critic_schema,
            feature_schema=args.feature_schema,
        )
        if args.critic_report:
            manifest.assert_report(_load_object(args.critic_report))
        _emit(
            {"valid": True, **manifest.to_dict()},
            args.output,
        )
        return 0
    if args.command == "validate-campaign":
        manifest = FrozenCampaignManifest.load(args.manifest)
        result = validate_campaign_artifacts(manifest, base_dir=args.base_dir)
        _emit(result, args.output)
        return 0 if result["valid"] else 2
    if args.command == "run-campaign":
        manifest = FrozenCampaignManifest.load(args.manifest)
        compatibility = CompatibilityManifest.load(args.compatibility)
        bundle = load_backend_factory(
            args.backend_factory,
            manifest=manifest,
            compatibility=compatibility,
        )
        result = run_frozen_campaign(
            manifest,
            bundle,
            compatibility=compatibility,
            trials_output=args.trials_output,
            targets_output=args.targets_output,
            reward_spec=_reward(args.reward_config),
        )
        _emit(result, args.output)
        return 0
    if args.command == "build-targets":
        targets = targets_from_trials(
            read_trials(args.trials), reward_spec=_reward(args.reward_config)
        )
        write_targets(targets, args.output)
        print(json.dumps({"written": str(args.output), "sample_count": len(targets)}))
        return 0
    if args.command == "audit-targets":
        result = audit_targets(read_targets(args.targets))
        _emit(result, args.output)
        return 0 if result["valid"] else 2
    if args.command == "train-values":
        config = ValueTrainConfig()
        if args.config:
            raw = _load_object(args.config)
            config = ValueTrainConfig.from_dict(raw.get("training", raw))
        report = train_action_value_policy(
            args.targets,
            args.output_dir,
            compatibility_manifest=CompatibilityManifest.load(args.compatibility),
            config=config,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "model_id": report["model_id"],
                    "winner_seed": report["winner"]["seed"],
                }
            )
        )
        return 0
    if args.command == "adapt-proxy-targets":
        targets, report = adapt_proxy_manifests(
            args.manifest,
            reward_spec=_reward(args.reward_config),
        )
        write_targets(targets, args.output)
        write_adaptation_report(report, args.report)
        print(
            json.dumps(
                {
                    "targets": str(args.output),
                    "report": str(args.report),
                    "sample_count": len(targets),
                }
            )
        )
        return 0
    if args.command == "evaluate":
        targets = read_targets(args.targets)
        compatibility = CompatibilityManifest.load(args.compatibility)
        methods: dict[str, Any] = {
            "R0_category_only": CategoryOnlyPolicy(
                compatibility_id=compatibility.compatibility_id
            ),
            "R1_heuristic": HeuristicDecisionPolicy(
                compatibility_id=compatibility.compatibility_id
            ),
        }
        memory_records = read_targets(args.memory_targets or args.targets)
        memory = ActualTrialMemory(memory_records)
        methods["R3_memory_only"] = ActionValueDecisionPolicy(
            MemoryValuePredictor(
                memory, compatibility_id=compatibility.compatibility_id
            ),
            minimum_confidence=0.0,
        )
        if args.checkpoint:
            learned = TorchActionValuePolicy.load(
                args.checkpoint,
                device=args.device,
                compatibility_manifest=compatibility,
            )
            methods["R2_policy_only"] = ActionValueDecisionPolicy(
                learned, minimum_confidence=0.0
            )
            methods["R4_policy_plus_memory"] = ActionValueDecisionPolicy(
                BlendedValuePredictor(learned, memory), minimum_confidence=0.0
            )
            if args.hunyuan_calibration_targets:
                calibration = read_targets(args.hunyuan_calibration_targets)
                if any(item.domain != "hunyuan" for item in calibration):
                    raise ValueError("R5 calibration targets must all be Hunyuan domain")
                calibrated_memory = ActualTrialMemory((*memory_records, *calibration))
                methods["R5_hunyuan_calibrated"] = ActionValueDecisionPolicy(
                    BlendedValuePredictor(
                        learned,
                        calibrated_memory,
                        model_id="action-value+hunyuan-memory",
                    ),
                    minimum_confidence=0.0,
                )
        _emit(compare_policies(methods, targets), args.output)
        return 0
    if args.command == "closed-loop-report":
        _emit(closed_loop_metrics(read_trials(args.trials)), args.output)
        return 0
    if args.command == "integration-review":
        evaluation = (
            None if args.evaluation_evidence is None else _load_object(args.evaluation_evidence)
        )
        review = build_integration_review(
            compatibility=CompatibilityManifest.load(args.compatibility),
            executor_manifest=_load_object(args.executor_manifest),
            baseline_verification=_load_object(args.baseline_evidence),
            test_evidence=_load_object(args.test_evidence),
            evaluation_evidence=evaluation,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_integration_review(
            review,
            json_path=args.output_dir / "integration_review.json",
            markdown_path=args.output_dir / "integration_review.md",
        )
        print(json.dumps({"output_dir": str(args.output_dir), "research_entry_ready": review["research_entry_ready"]}))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
