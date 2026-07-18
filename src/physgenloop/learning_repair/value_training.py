"""Group-safe, multi-seed training for action classification and per-action value."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from physgenloop.learning_repair.contracts import ACTION_ORDER
from physgenloop.learning_repair.features import FeatureConfig, ReportFeatureEncoder
from physgenloop.learning_repair.policy import require_torch, resolve_device

from .compatibility import CompatibilityManifest
from .contracts import LearningTargetV1
from .recording import read_targets, write_targets
from .value_policy import POLICY_FORMAT_VERSION, build_action_value_mlp


@dataclass(frozen=True)
class ValueTrainConfig:
    seeds: tuple[int, ...] = (17, 23, 42, 73, 101)
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.15
    value_loss_weight: float = 0.4
    patience: int = 12
    minimum_delta: float = 1e-4
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not self.seeds or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("seeds, epochs, and batch_size must be non-empty/positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0.0 <= self.dropout < 1.0 or self.value_loss_weight < 0:
            raise ValueError("invalid dropout or value_loss_weight")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.validation_fraction < 0 or self.test_fraction < 0:
            raise ValueError("split fractions must be non-negative")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation + test fraction must be below one")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ValueTrainConfig":
        return cls(
            seeds=tuple(int(item) for item in raw.get("seeds", (17, 23, 42, 73, 101))),
            epochs=int(raw.get("epochs", 80)),
            batch_size=int(raw.get("batch_size", 64)),
            learning_rate=float(raw.get("learning_rate", 3e-4)),
            weight_decay=float(raw.get("weight_decay", 1e-4)),
            hidden_dims=tuple(int(item) for item in raw.get("hidden_dims", (128, 64))),
            dropout=float(raw.get("dropout", 0.15)),
            value_loss_weight=float(raw.get("value_loss_weight", 0.4)),
            patience=int(raw.get("patience", 12)),
            minimum_delta=float(raw.get("minimum_delta", 1e-4)),
            validation_fraction=float(raw.get("validation_fraction", 0.1)),
            test_fraction=float(raw.get("test_fraction", 0.1)),
            device=str(raw.get("device", "auto")),
            num_workers=int(raw.get("num_workers", 0)),
        )


def assign_group_splits(
    targets: Iterable[LearningTargetV1],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[LearningTargetV1, ...]:
    records = tuple(targets)
    explicit = [item.split is not None for item in records]
    if any(explicit):
        if not all(explicit):
            raise ValueError("target manifest has partially assigned splits")
        _assert_no_group_leakage(records)
        return records
    groups = sorted({item.group_id for item in records})
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).hexdigest(),
    )
    validation_count = round(len(ranked) * validation_fraction)
    test_count = round(len(ranked) * test_fraction)
    if len(ranked) >= 3 and validation_fraction > 0:
        validation_count = max(1, validation_count)
    if len(ranked) >= 3 and test_fraction > 0:
        test_count = max(1, test_count)
    while validation_count + test_count >= len(ranked) and (validation_count or test_count):
        if test_count >= validation_count and test_count:
            test_count -= 1
        elif validation_count:
            validation_count -= 1
    validation = set(ranked[:validation_count])
    test = set(ranked[validation_count : validation_count + test_count])
    assigned = tuple(
        replace(
            item,
            split=(
                "validation"
                if item.group_id in validation
                else "test" if item.group_id in test else "train"
            ),
        )
        for item in records
    )
    _assert_no_group_leakage(assigned)
    return assigned


def _assert_no_group_leakage(records: Iterable[LearningTargetV1]) -> None:
    group_splits: dict[str, set[str | None]] = {}
    for item in records:
        group_splits.setdefault(item.group_id, set()).add(item.split)
    leakage = {key: value for key, value in group_splits.items() if len(value) > 1}
    if leakage:
        raise ValueError(f"group leakage detected: {leakage}")


def _loader(torch, records, encoder, config, *, shuffle: bool, seed: int):
    features = torch.tensor(
        [encoder.encode(item.critic_report, item.context) for item in records],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [ACTION_ORDER.index(item.target_action) for item in records], dtype=torch.long
    )
    rewards = torch.tensor(
        [
            [0.0 if item.action_rewards[action] is None else item.action_rewards[action] for action in ACTION_ORDER]
            for item in records
        ],
        dtype=torch.float32,
    )
    reward_mask = torch.tensor(
        [
            [item.action_rewards[action] is not None for action in ACTION_ORDER]
            for item in records
        ],
        dtype=torch.bool,
    )
    availability = torch.tensor(
        [[item.available_actions[action] for action in ACTION_ORDER] for item in records],
        dtype=torch.bool,
    )
    dataset = torch.utils.data.TensorDataset(
        features, labels, rewards, reward_mask, availability
    )
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, max(1, len(records))),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda") or config.device == "auto",
    )


def _metrics(
    true_actions: list[int],
    predicted_actions: list[int],
    regrets: list[float],
    value_errors: list[float],
) -> dict[str, Any]:
    size = len(ACTION_ORDER)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for truth, prediction in zip(true_actions, predicted_actions):
        matrix[truth][prediction] += 1
    per_action = {}
    f1_values = []
    recalls = []
    for index, action in enumerate(ACTION_ORDER):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(size) if row != index)
        fn = sum(matrix[index][column] for column in range(size) if column != index)
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
            f1_values.append(f1)
            recalls.append(recall)
    return {
        "sample_count": len(true_actions),
        "accuracy": (
            sum(a == b for a, b in zip(true_actions, predicted_actions)) / len(true_actions)
            if true_actions
            else 0.0
        ),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
        "value_mae": sum(value_errors) / len(value_errors) if value_errors else 0.0,
        "per_action": per_action,
        "confusion_matrix": matrix,
        "action_order": [item.value for item in ACTION_ORDER],
    }


def _evaluate(torch, model, loader, device) -> dict[str, Any]:
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    regrets: list[float] = []
    errors: list[float] = []
    with torch.inference_mode():
        for features, labels, rewards, reward_mask, availability in loader:
            logits, values = model(features.to(device))
            masked_logits = logits.masked_fill(~availability.to(device), -1e9)
            choices = masked_logits.argmax(dim=-1).cpu()
            truth.extend(int(item) for item in labels)
            predicted.extend(int(item) for item in choices)
            cpu_values = values.cpu()
            for row in range(len(labels)):
                observed = [
                    float(rewards[row, column])
                    for column in range(len(ACTION_ORDER))
                    if bool(reward_mask[row, column])
                ]
                chosen = int(choices[row])
                chosen_reward = (
                    float(rewards[row, chosen]) if bool(reward_mask[row, chosen]) else 0.0
                )
                regrets.append(max(observed, default=0.0) - chosen_reward)
                for column in range(len(ACTION_ORDER)):
                    if bool(reward_mask[row, column]):
                        errors.append(abs(float(rewards[row, column]) - float(cpu_values[row, column])))
    return _metrics(truth, predicted, regrets, errors)


def train_action_value_policy(
    target_manifest: str | Path,
    output_dir: str | Path,
    *,
    compatibility_manifest: CompatibilityManifest,
    config: ValueTrainConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> dict[str, Any]:
    """Train cloud-ready checkpoints; the caller chooses the machine/environment."""

    config = config or ValueTrainConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records = assign_group_splits(
        read_targets(target_manifest),
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seeds[0],
    )
    train_records = tuple(item for item in records if item.split == "train")
    validation_records = tuple(item for item in records if item.split == "validation") or train_records
    test_records = tuple(item for item in records if item.split == "test")
    if not train_records:
        raise ValueError("training split is empty")
    write_targets(train_records, output / "train.jsonl")
    write_targets(validation_records, output / "validation.jsonl")
    if test_records:
        write_targets(test_records, output / "test.jsonl")

    torch = require_torch()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(torch, config.device)
    encoder = ReportFeatureEncoder(feature_config)
    seed_results = []
    winner = None
    winner_state = None
    for seed in config.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = build_action_value_mlp(
            torch,
            input_dim=encoder.dimension,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
        train_loader = _loader(torch, train_records, encoder, config, shuffle=True, seed=seed)
        validation_loader = _loader(
            torch, validation_records, encoder, config, shuffle=False, seed=seed
        )
        counts = Counter(ACTION_ORDER.index(item.target_action) for item in train_records)
        class_weights = torch.tensor(
            [
                len(train_records) / (len(ACTION_ORDER) * counts[index])
                if counts[index]
                else 0.0
                for index in range(len(ACTION_ORDER))
            ],
            dtype=torch.float32,
            device=device,
        )
        classification_loss = torch.nn.CrossEntropyLoss(
            weight=class_weights, reduction="none"
        )
        value_loss = torch.nn.SmoothL1Loss(reduction="none")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        best_score = float("-inf")
        best_state = None
        best_epoch = 0
        stale = 0
        history = []
        for epoch in range(1, config.epochs + 1):
            model.train()
            losses = []
            for features, labels, rewards, reward_mask, availability in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                rewards = rewards.to(device)
                reward_mask = reward_mask.to(device)
                availability = availability.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, values = model(features)
                logits = logits.masked_fill(~availability, -1e9)
                class_term = classification_loss(logits, labels).mean()
                raw_value = value_loss(values, rewards)
                value_term = (raw_value * reward_mask).sum() / reward_mask.sum().clamp_min(1)
                loss = class_term + config.value_loss_weight * value_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics = _evaluate(torch, model, validation_loader, device)
            score = float(metrics["macro_f1"]) - 0.1 * max(0.0, float(metrics["mean_regret"]))
            history.append(
                {"epoch": epoch, "loss": sum(losses) / len(losses), "validation": metrics}
            )
            if score > best_score + config.minimum_delta:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        validation_metrics = _evaluate(torch, model, validation_loader, device)
        seed_result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "validation": validation_metrics,
            "history": history,
        }
        seed_results.append(seed_result)
        rank = (
            float(validation_metrics["macro_f1"]),
            -float(validation_metrics["mean_regret"]),
            -float(validation_metrics["value_mae"]),
            -seed,
        )
        if winner is None or rank > winner[0]:
            winner = (rank, seed_result)
            winner_state = best_state
    assert winner is not None and winner_state is not None
    target_manifest_sha256 = hashlib.sha256(
        Path(target_manifest).read_bytes()
    ).hexdigest()
    model_identity = {
        "format_version": POLICY_FORMAT_VERSION,
        "target_manifest_sha256": target_manifest_sha256,
        "compatibility_id": compatibility_manifest.compatibility_id,
        "training_config": asdict(config),
        "winner_seed": winner[1]["seed"],
    }
    model_id = "repair-value-" + hashlib.sha256(
        json.dumps(model_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    proxy_label_count = sum(bool(item.metadata.get("proxy_label")) for item in records)
    actual_trial_label_count = len(records) - proxy_label_count
    selection_mode = (
        "classification_proxy" if actual_trial_label_count == 0 else "action_value"
    )
    checkpoint = {
        "format_version": POLICY_FORMAT_VERSION,
        "model_id": model_id,
        "target_manifest_sha256": target_manifest_sha256,
        "compatibility_id": compatibility_manifest.compatibility_id,
        "action_order": [item.value for item in ACTION_ORDER],
        "feature_config": encoder.config.to_dict(),
        "feature_names": list(encoder.feature_names),
        "model_config": {
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
        },
        "state_dict": winner_state,
        "winner_seed": winner[1]["seed"],
        "selection_mode": selection_mode,
        "proxy_label_count": proxy_label_count,
        "actual_trial_label_count": actual_trial_label_count,
    }
    torch.save(checkpoint, output / "best_action_value_policy.pt")
    report = {
        "format_version": POLICY_FORMAT_VERSION,
        "model_id": model_id,
        "selection_metric": [
            "max validation.macro_f1",
            "min validation.mean_regret",
            "min validation.value_mae",
            "min seed",
        ],
        "compatibility": compatibility_manifest.to_dict(),
        "training_config": asdict(config),
        "sample_counts": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "domains": dict(Counter(item.domain for item in records)),
        "label_provenance": {
            "proxy_label_count": proxy_label_count,
            "actual_trial_label_count": actual_trial_label_count,
            "selection_mode": selection_mode,
        },
        "seeds": seed_results,
        "winner": winner[1],
    }
    if test_records:
        model = build_action_value_mlp(
            torch,
            input_dim=encoder.dimension,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(winner_state)
        report["held_out_test"] = _evaluate(
            torch,
            model,
            _loader(torch, test_records, encoder, config, shuffle=False, seed=config.seeds[0]),
            device,
        )
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
