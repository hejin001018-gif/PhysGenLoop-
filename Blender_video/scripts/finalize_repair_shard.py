"""Run the frozen Critic over Blender GT observations and build Repair examples."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from pavg_critic import CriticRequest, PhysicsCritic, PhysicsPlan, load_config
from pavg_critic.schemas import FrameState
from physgenloop.learning_repair import (
    RepairAction,
    RepairContext,
    RepairExample,
    audit_dataset,
    write_repair_manifest,
)


ACTION_BY_CATEGORY = {
    "physical": RepairAction.REJECT,
    "premature_rebound": RepairAction.PROMPT_REPAIR,
    "surface_penetration": RepairAction.LOCAL_EDITING,
    "object_disappearance": RepairAction.LOCAL_EDITING,
    "reverse_gravity": RepairAction.PROMPT_REPAIR,
    "teleportation": RepairAction.LOCAL_EDITING,
    "unknown": RepairAction.GLOBAL_REGENERATION,
    "multi_violation": RepairAction.REJECT,
}


STRATEGY_BY_CATEGORY = {
    "physical": "no_repair_for_physical_candidate",
    "premature_rebound": "make_floor_contact_precede_rebound",
    "surface_penetration": "replace_the_local_contact_interval",
    "object_disappearance": "restore_the_tracked_object_layer",
    "reverse_gravity": "make_downward_gravity_explicit",
    "teleportation": "replace_the_discontinuous_local_trajectory",
    "unknown": "regenerate_with_visible_target_and_stable_camera",
    "multi_violation": "reject_after_repair_budget_is_exhausted",
}


EXPECTED_REPORT_CATEGORY = {
    "premature_rebound": "premature_rebound",
    "surface_penetration": "surface_penetration",
    "object_disappearance": "object_disappearance",
    "reverse_gravity": "reverse_gravity",
    "teleportation": "teleportation",
}


def context_cases(category: str, base_action: RepairAction, profile: str):
    """Return policy-supervision cases for one frozen Critic report.

    ``standard`` preserves the v1/v2 contract.  ``hard-v1`` reuses each expensive
    Blender render under several realistic RepairContext states, so the selector
    must learn availability and remaining-budget constraints instead of memorizing
    a one-to-one error-category mapping.
    """

    if profile not in {"hard-v1", "hard-v1.1"}:
        attempt_index = 3 if category == "multi_violation" else 0
        return (
            (
                "standard",
                base_action,
                STRATEGY_BY_CATEGORY[category],
                RepairContext(
                    attempt_index=attempt_index,
                    max_attempts=3,
                    prompt_repair_available=True,
                    global_regeneration_available=True,
                    local_editor_available=True,
                    semantic_score=1.0,
                    quality_score=1.0,
                ),
            ),
        )

    if category == "physical":
        return (
            (
                "clean_physical",
                RepairAction.REJECT,
                STRATEGY_BY_CATEGORY[category],
                RepairContext(semantic_score=1.0, quality_score=1.0),
            ),
        )
    if category == "multi_violation":
        return (
            (
                "budget_exhausted",
                RepairAction.REJECT,
                STRATEGY_BY_CATEGORY[category],
                RepairContext(
                    attempt_index=3,
                    max_attempts=3,
                    semantic_score=0.9,
                    quality_score=0.85,
                    previous_actions=(
                        RepairAction.PROMPT_REPAIR,
                        RepairAction.LOCAL_EDITING,
                        RepairAction.GLOBAL_REGENERATION,
                    ),
                ),
            ),
        )

    nominal = (
        "nominal",
        base_action,
        STRATEGY_BY_CATEGORY[category],
        RepairContext(semantic_score=1.0, quality_score=1.0),
    )
    if base_action is RepairAction.PROMPT_REPAIR:
        fallback_context = RepairContext(
            attempt_index=1,
            max_attempts=3,
            prompt_repair_available=False,
            global_regeneration_available=True,
            local_editor_available=True,
            semantic_score=0.92,
            quality_score=0.9,
            previous_actions=(RepairAction.PROMPT_REPAIR,),
        )
        fallback = (
            "preferred_failed",
            RepairAction.GLOBAL_REGENERATION,
            "regenerate_after_prompt_repair_failed_or_became_unavailable",
            fallback_context,
        )
    elif base_action is RepairAction.LOCAL_EDITING:
        fallback_context = RepairContext(
            attempt_index=1,
            max_attempts=3,
            prompt_repair_available=True,
            global_regeneration_available=True,
            local_editor_available=False,
            semantic_score=0.92,
            quality_score=0.9,
            previous_actions=(RepairAction.LOCAL_EDITING,),
        )
        fallback = (
            "preferred_failed",
            RepairAction.GLOBAL_REGENERATION,
            "regenerate_after_local_editing_failed_or_became_unavailable",
            fallback_context,
        )
    else:
        fallback_context = RepairContext(
            attempt_index=1,
            max_attempts=3,
            prompt_repair_available=True,
            global_regeneration_available=False,
            local_editor_available=True,
            semantic_score=0.92,
            quality_score=0.9,
            previous_actions=(RepairAction.GLOBAL_REGENERATION,),
        )
        fallback = (
            "preferred_failed",
            RepairAction.REJECT,
            "reject_unknown_case_after_regeneration_failed_or_became_unavailable",
            fallback_context,
        )

    budget = (
        "budget_exhausted",
        RepairAction.REJECT,
        "reject_after_repair_budget_is_exhausted",
        RepairContext(
            attempt_index=3,
            max_attempts=3,
            semantic_score=0.88,
            quality_score=0.84,
            previous_actions=(base_action, RepairAction.GLOBAL_REGENERATION, base_action),
        ),
    )
    return nominal, fallback, budget


def repair_cost(action: RepairAction):
    return {
        RepairAction.PROMPT_REPAIR: 0.25,
        RepairAction.GLOBAL_REGENERATION: 1.0,
        RepairAction.LOCAL_EDITING: 0.5,
        RepairAction.REJECT: None,
    }[action]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(critic, metadata, video_path):
    plan = PhysicsPlan(
        objects=(metadata["object"],),
        # Use the Critic's canonical event vocabulary.  ``contact`` is natural
        # language, whereas the deterministic event emitted by the frozen Critic is
        # named ``floor_contact``.
        expected_events=("fall", "floor_contact", "rebound"),
    )
    request = CriticRequest(
        video_path=str(video_path),
        prompt=metadata["prompt"],
        physics_plan=plan,
    )
    observations = tuple(FrameState.from_dict(item) for item in metadata["observations"])
    return critic.analyze(request, observations=observations, floor_y=metadata["floor_y"])


def semantic_gate_result(metadata, report):
    """Check that injected semantics survive the frozen-Critic contract."""

    category = metadata["category"]
    observed = sorted({item.category for item in report.violations})
    reasons = []
    if category == "physical":
        if str(report.decision) != "physical":
            reasons.append(f"normal decision is {report.decision!s}, expected physical")
        if observed:
            reasons.append(f"normal report contains violations: {observed}")
    elif category == "unknown":
        if str(report.decision) != "unknown":
            reasons.append(f"unknown decision is {report.decision!s}, expected unknown")
        if observed:
            reasons.append(f"unknown report contains positive violations: {observed}")
    elif category == "multi_violation":
        if str(report.decision) != "violation":
            reasons.append(f"multi-corrupt decision is {report.decision!s}, expected violation")
        if len(observed) < 2:
            reasons.append(f"multi-corrupt exposed only {len(observed)} violation families")
    else:
        expected = EXPECTED_REPORT_CATEGORY[category]
        if str(report.decision) != "violation":
            reasons.append(f"injected anomaly decision is {report.decision!s}, expected violation")
        if expected not in observed:
            reasons.append(f"missing expected violation {expected!r}; observed {observed}")
    return {
        "sample_id": f"{metadata['group_id']}--{metadata['variant']}",
        "injected_category": category,
        "decision": str(report.decision),
        "observed_categories": observed,
        "valid": not reasons,
        "reasons": reasons,
    }


def main() -> int:
    args = parse_args()
    root = args.shard_root.resolve()
    output = args.output or root / "repair_manifest.jsonl"
    critic = PhysicsCritic(load_config(args.config))
    examples = []
    report_decisions = Counter()
    semantic_results = []
    group_dirs = sorted((root / "groups").glob("group_*"))
    if not group_dirs:
        raise ValueError(f"no generated groups found below {root}")
    for group_dir in group_dirs:
        normal_dir = group_dir / "normal"
        normal_metadata = read_json(normal_dir / "metadata.json")
        normal_video = root / normal_metadata["video"]
        normal_report = analyze(critic, normal_metadata, normal_video)
        (normal_dir / "critic_report.json").write_text(
            normal_report.to_json() + "\n", encoding="utf-8"
        )
        for variant_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            metadata_path = variant_dir / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = read_json(metadata_path)
            video_path = root / metadata["video"]
            report = normal_report if metadata["category"] == "physical" else analyze(
                critic, metadata, video_path
            )
            report_decisions[str(report.decision)] += 1
            semantic_results.append(semantic_gate_result(metadata, report))
            (variant_dir / "critic_report.json").write_text(
                report.to_json() + "\n", encoding="utf-8"
            )
            category = metadata["category"]
            profile = str(metadata.get("difficulty_profile", "standard"))
            variant_examples = []
            for case_name, action, strategy, context in context_cases(
                category, ACTION_BY_CATEGORY[category], profile
            ):
                reject = action is RepairAction.REJECT
                base_id = f"{metadata['group_id']}--{metadata['variant']}"
                sample_id = base_id if profile == "standard" else f"{base_id}--{case_name}"
                example = RepairExample(
                    sample_id=sample_id,
                    group_id=metadata["group_id"],
                    prompt=metadata["prompt"],
                    critic_report=report.to_dict(),
                    target_action=action,
                    strategy=strategy,
                    before_score=report.physics_score,
                    after_score=(
                        report.physics_score if reject else normal_report.physics_score
                    ),
                    successful=True,
                    semantic_score=None if reject else 1.0,
                    quality_score=None if reject else 1.0,
                    repair_cost=repair_cost(action),
                    context=context,
                    artifacts={
                        "wrong_video": metadata["video"],
                        "correct_video": normal_metadata["video"],
                        "metadata": str(metadata_path.relative_to(root).as_posix()),
                    },
                    metadata={
                        "domain": "blender",
                        "category": category,
                        "severity": metadata["severity"],
                        "seed": metadata["seed"],
                        "difficulty_profile": profile,
                        "context_case": case_name,
                        "critic_contract": "pavg-critic-0.3.0/configs-default",
                        "label_source": (
                            "blender_hard_context_proxy_v1_1"
                            if profile == "hard-v1.1"
                            else "blender_hard_context_proxy_v1"
                            if profile == "hard-v1"
                            else "blender_proxy_repair_trial"
                        ),
                    },
                )
                variant_examples.append(example)
                examples.append(example)
            record_path = (
                variant_dir / "repair_samples.json"
                if profile in {"hard-v1", "hard-v1.1"}
                else variant_dir / "repair_sample.json"
            )
            payload = (
                [item.to_dict() for item in variant_examples]
                if profile in {"hard-v1", "hard-v1.1"}
                else variant_examples[0].to_dict()
            )
            record_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    write_repair_manifest(examples, output)
    audit = audit_dataset(examples, check_artifacts=True, base_dir=root)
    semantic_failures = [item for item in semantic_results if not item["valid"]]
    semantic_gate = {
        "valid": not semantic_failures,
        "checked": len(semantic_results),
        "failure_count": len(semantic_failures),
        "failures": semantic_failures,
        "results": semantic_results,
    }
    summary = {
        "audit": audit.to_dict(),
        "semantic_gate": semantic_gate,
        "critic_decisions": dict(report_decisions),
        "manifest": str(output),
    }
    (root / "finalize_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if audit.valid and semantic_gate["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
