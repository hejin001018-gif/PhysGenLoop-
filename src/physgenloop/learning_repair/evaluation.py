"""Classification, regret, capability, and closed-loop evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from physgenloop.learning_repair.contracts import ACTION_ORDER, RepairAction, RepairContext

from .contracts import LearningTargetV1, RepairTrialV1


@dataclass(frozen=True)
class _CandidateStub:
    candidate_id: str = "evaluation-candidate"


def _classification_metrics(
    truth: list[RepairAction], predicted: list[RepairAction]
) -> dict[str, Any]:
    matrix = [[0 for _ in ACTION_ORDER] for _ in ACTION_ORDER]
    for expected, actual in zip(truth, predicted):
        matrix[ACTION_ORDER.index(expected)][ACTION_ORDER.index(actual)] += 1
    per_action = {}
    f1s = []
    recalls = []
    for index, action in enumerate(ACTION_ORDER):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(ACTION_ORDER)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(ACTION_ORDER)) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(matrix[index])
        per_action[action.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1s.append(f1)
            recalls.append(recall)
    return {
        "sample_count": len(truth),
        "accuracy": sum(a is b for a, b in zip(truth, predicted)) / len(truth) if truth else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "per_action": per_action,
        "confusion_matrix": matrix,
        "action_order": [item.value for item in ACTION_ORDER],
    }


def evaluate_decision_policy(
    policy: Any,
    targets: Iterable[LearningTargetV1],
) -> dict[str, Any]:
    records = tuple(targets)
    truth = []
    predictions = []
    regrets = []
    unavailable = 0
    fallback = 0
    for item in records:
        context = RepairContext(
            attempt_index=item.context.attempt_index,
            max_attempts=item.context.max_attempts,
            prompt_repair_available=item.available_actions[RepairAction.PROMPT_REPAIR],
            global_regeneration_available=item.available_actions[RepairAction.GLOBAL_REGENERATION],
            local_editor_available=item.available_actions[RepairAction.LOCAL_EDITING],
            semantic_score=item.context.semantic_score,
            quality_score=item.context.quality_score,
            previous_actions=item.context.previous_actions,
        )
        decision = policy.decide(
            critic_report=item.critic_report,
            candidate=_CandidateStub(),
            prompt=str(item.metadata.get("prompt", "")),
            context=context,
        )
        truth.append(item.target_action)
        predictions.append(decision.action)
        unavailable += not item.available_actions[decision.action]
        fallback += bool(decision.abstained)
        observed = [value for value in item.action_rewards.values() if value is not None]
        selected = item.action_rewards[decision.action]
        regrets.append(max(observed, default=0.0) - (0.0 if selected is None else selected))
    metrics = _classification_metrics(truth, predictions)
    metrics.update(
        {
            "mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
            "unavailable_action_rate": unavailable / len(records) if records else 0.0,
            "fallback_abstention_rate": fallback / len(records) if records else 0.0,
        }
    )
    return metrics


def compare_policies(
    policies: Mapping[str, Any],
    targets: Iterable[LearningTargetV1],
) -> dict[str, Any]:
    records = tuple(targets)
    by_domain = {
        domain: tuple(item for item in records if item.domain == domain)
        for domain in sorted({item.domain for item in records})
    }
    return {
        "method_ids": list(policies),
        "overall": {
            name: evaluate_decision_policy(policy, records)
            for name, policy in policies.items()
        },
        "by_domain": {
            domain: {
                name: evaluate_decision_policy(policy, domain_records)
                for name, policy in policies.items()
            }
            for domain, domain_records in by_domain.items()
        },
        "domain_warning": (
            "Blender and Hunyuan results are reported separately and must not be merged into a deployment claim."
            if len(by_domain) > 1
            else None
        ),
    }


def closed_loop_metrics(trials: Iterable[RepairTrialV1]) -> dict[str, Any]:
    records = tuple(trials)
    if not records:
        return {"sample_count": 0, "by_domain": {}}
    grouped = defaultdict(list)
    for trial in records:
        grouped[trial.domain].append(trial)

    def summarize(items: list[RepairTrialV1]) -> dict[str, Any]:
        gains = [item.physics_gain for item in items if item.physics_gain is not None]
        semantics = [
            item.after_scores.semantic
            for item in items
            if item.after_scores is not None and item.after_scores.semantic is not None
        ]
        qualities = [
            item.after_scores.quality
            for item in items
            if item.after_scores is not None and item.after_scores.quality is not None
        ]
        costs = [float(item.execution.get("cost", 0.0)) for item in items]
        latencies = [float(item.execution.get("latency_seconds", 0.0)) for item in items]
        new_errors = [bool(item.metadata.get("introduced_new_error")) for item in items]
        return {
            "trial_count": len(items),
            "repair_success_rate": sum(item.successful for item in items) / len(items),
            "mean_physics_gain": sum(gains) / len(gains) if gains else None,
            "mean_semantic_preservation": sum(semantics) / len(semantics) if semantics else None,
            "mean_visual_quality": sum(qualities) / len(qualities) if qualities else None,
            "introduced_error_rate": sum(new_errors) / len(new_errors) if new_errors else 0.0,
            "mean_cost": sum(costs) / len(costs),
            "mean_latency_seconds": sum(latencies) / len(latencies),
            "fallback_abstention_rate": sum(item.decision.abstained for item in items) / len(items),
            "action_counts": dict(Counter(item.decision.action.value for item in items)),
            "failure_reasons": dict(Counter(item.failure_reason for item in items if not item.successful)),
        }

    return {
        "trial_count": len(records),
        "by_domain": {domain: summarize(items) for domain, items in sorted(grouped.items())},
        "cross_domain_aggregation_prohibited": len(grouped) > 1,
    }


def audit_targets(targets: Iterable[LearningTargetV1]) -> dict[str, Any]:
    records = tuple(targets)
    group_splits: dict[str, set[str | None]] = defaultdict(set)
    for item in records:
        group_splits[item.group_id].add(item.split)
    leakage = {
        group: sorted("null" if value is None else value for value in splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    return {
        "valid": bool(records) and not leakage,
        "sample_count": len(records),
        "group_count": len(group_splits),
        "domains": dict(Counter(item.domain for item in records)),
        "actions": dict(Counter(item.target_action.value for item in records)),
        "splits": dict(Counter(str(item.split) for item in records)),
        "group_leakage": leakage,
        "proxy_label_count": sum(bool(item.metadata.get("proxy_label")) for item in records),
        "actual_trial_label_count": sum(not bool(item.metadata.get("proxy_label")) for item in records),
    }
