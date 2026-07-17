"""PyTorch MLP Repair Policy 的可复现训练与评估。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import yaml

from .contracts import ACTION_ORDER, RepairExample
from .dataset import (
    audit_dataset,
    grouped_split,
    load_repair_manifest,
    select_split,
    write_repair_manifest,
)
from .features import FeatureConfig, ReportFeatureEncoder
from .policy import build_repair_mlp, require_torch, resolve_device


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.15
    gain_loss_weight: float = 0.2
    patience: int = 12
    minimum_delta: float = 1e-4
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    device: str = "auto"
    num_workers: int = 0
    successful_only: bool = True
    evaluate_test: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if self.gain_loss_weight < 0 or self.patience < 1:
            raise ValueError("gain_loss_weight must be non-negative and patience positive")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainConfig":
        return cls(
            seed=int(raw.get("seed", 42)),
            epochs=int(raw.get("epochs", 80)),
            batch_size=int(raw.get("batch_size", 64)),
            learning_rate=float(raw.get("learning_rate", 3e-4)),
            weight_decay=float(raw.get("weight_decay", 1e-4)),
            hidden_dims=tuple(int(item) for item in raw.get("hidden_dims", (128, 64))),
            dropout=float(raw.get("dropout", 0.15)),
            gain_loss_weight=float(raw.get("gain_loss_weight", 0.2)),
            patience=int(raw.get("patience", 12)),
            minimum_delta=float(raw.get("minimum_delta", 1e-4)),
            validation_fraction=float(raw.get("validation_fraction", 0.1)),
            test_fraction=float(raw.get("test_fraction", 0.1)),
            device=str(raw.get("device", "auto")),
            num_workers=int(raw.get("num_workers", 0)),
            successful_only=bool(raw.get("successful_only", True)),
            evaluate_test=bool(raw.get("evaluate_test", True)),
        )


def load_train_config(path: str | Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("repair training config must be a YAML object")
    training = raw.get("training", raw)
    if not isinstance(training, Mapping):
        raise ValueError("training config section must be an object")
    return TrainConfig.from_dict(training)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(
    true_actions: list[int],
    predicted_actions: list[int],
    true_gains: list[float],
    predicted_gains: list[float],
) -> dict[str, Any]:
    matrix = [[0 for _ in ACTION_ORDER] for _ in ACTION_ORDER]
    for truth, prediction in zip(true_actions, predicted_actions):
        matrix[truth][prediction] += 1
    per_class = {}
    f1_values = []
    present = set(true_actions)
    for index, action in enumerate(ACTION_ORDER):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(ACTION_ORDER)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(ACTION_ORDER)) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[action.value] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(matrix[index]),
        }
        if index in present:
            f1_values.append(f1)
    count = len(true_actions)
    accuracy = (
        sum(truth == prediction for truth, prediction in zip(true_actions, predicted_actions))
        / count
        if count
        else 0.0
    )
    gain_mae = (
        sum(abs(truth - prediction) for truth, prediction in zip(true_gains, predicted_gains))
        / len(true_gains)
        if true_gains
        else 0.0
    )
    return {
        "sample_count": count,
        "accuracy": round(accuracy, 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else 0.0,
        "gain_mae": round(gain_mae, 6),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "action_order": [item.value for item in ACTION_ORDER],
    }


def evaluate_policy(policy, examples: Iterable[RepairExample]) -> dict[str, Any]:
    """用统一指标评估任意 RepairPolicy，包括规则回退和已训练模型。"""

    records = tuple(examples)
    true_actions = []
    predicted_actions = []
    true_gains = []
    predicted_gains = []
    for item in records:
        prediction = policy.predict(item.critic_report, context=item.context)
        true_actions.append(ACTION_ORDER.index(item.target_action))
        predicted_actions.append(ACTION_ORDER.index(prediction.action))
        true_gains.append(item.score_gain)
        predicted_gains.append(prediction.expected_gain)
    return _metrics(true_actions, predicted_actions, true_gains, predicted_gains)


def _make_loader(torch, records, encoder, config, *, shuffle):
    features = torch.tensor(
        [encoder.encode(item.critic_report, item.context) for item in records],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [ACTION_ORDER.index(item.target_action) for item in records], dtype=torch.long
    )
    gains = torch.tensor([item.score_gain for item in records], dtype=torch.float32)
    weights = torch.tensor(
        [1.0 if item.successful else 0.25 for item in records], dtype=torch.float32
    )
    dataset = torch.utils.data.TensorDataset(features, labels, gains, weights)
    generator = torch.Generator().manual_seed(config.seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, max(1, len(records))),
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator if shuffle else None,
        pin_memory=config.device.startswith("cuda") or config.device == "auto",
    )


def _evaluate_model(torch, model, loader, device) -> dict[str, Any]:
    model.eval()
    true_actions: list[int] = []
    predicted_actions: list[int] = []
    true_gains: list[float] = []
    predicted_gains: list[float] = []
    with torch.inference_mode():
        for features, labels, gains, _weights in loader:
            logits, predicted_gain = model(features.to(device))
            predictions = logits.argmax(dim=-1).detach().cpu()
            true_actions.extend(int(item) for item in labels)
            predicted_actions.extend(int(item) for item in predictions)
            true_gains.extend(float(item) for item in gains)
            predicted_gains.extend(float(item) for item in predicted_gain.detach().cpu())
    return _metrics(true_actions, predicted_actions, true_gains, predicted_gains)


def train_policy(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: TrainConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> dict[str, Any]:
    """训练并写出自描述 checkpoint、数据切分和审计报告。"""

    torch = require_torch()
    config = config or TrainConfig()
    # CuBLAS requires this before the first CUDA kernel for reproducible GEMM.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:  # pragma: no cover - old torch fallback
        pass
    device = resolve_device(torch, config.device)

    manifest = Path(manifest_path)
    records = load_repair_manifest(manifest)
    if config.successful_only:
        records = tuple(item for item in records if item.successful)
    if not records:
        raise ValueError("no eligible training examples after successful_only filtering")
    assigned = [item.split is not None for item in records]
    if any(assigned):
        if not all(assigned):
            raise ValueError("manifest has partially assigned splits; assign all or none")
        assigned_audit = audit_dataset(records)
        if assigned_audit.group_leakage:
            raise ValueError(f"group leakage detected: {assigned_audit.group_leakage}")
    else:
        records = grouped_split(
            records,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=config.seed,
        )
    audit = audit_dataset(records)
    if audit.group_leakage:
        raise ValueError(f"group leakage detected: {audit.group_leakage}")
    train_records = select_split(records, "train")
    validation_records = select_split(records, "validation") or train_records
    test_records = select_split(records, "test")
    if not train_records:
        raise ValueError("training split is empty")

    encoder = ReportFeatureEncoder(feature_config)
    model = build_repair_mlp(
        torch,
        input_dim=encoder.dimension,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    train_loader = _make_loader(torch, train_records, encoder, config, shuffle=True)
    validation_loader = _make_loader(
        torch, validation_records, encoder, config, shuffle=False
    )
    test_loader = (
        _make_loader(torch, test_records, encoder, config, shuffle=False)
        if test_records
        else None
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
    gain_loss = torch.nn.SmoothL1Loss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, config.patience // 3)
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for features, labels, gains, sample_weights in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            gains = gains.to(device)
            sample_weights = sample_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_gain = model(features)
            per_sample = classification_loss(logits, labels)
            per_sample = per_sample + config.gain_loss_weight * gain_loss(
                predicted_gain, gains
            )
            loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_metrics = _evaluate_model(
            torch, model, validation_loader, device
        )
        current_f1 = float(validation_metrics["macro_f1"])
        scheduler.step(current_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(sum(losses) / len(losses), 8),
                "validation_macro_f1": current_f1,
                "validation_accuracy": validation_metrics["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if current_f1 > best_f1 + config.minimum_delta:
            best_f1 = current_f1
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_metrics = _evaluate_model(torch, model, validation_loader, device)
    # Multi-seed/model-selection campaigns must not repeatedly inspect the held-out
    # test set.  They disable this field, select exclusively on validation macro-F1,
    # and evaluate the winning checkpoint exactly once.
    test_metrics = (
        _evaluate_model(torch, model, test_loader, device)
        if test_loader and config.evaluate_test
        else None
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_id = f"repair-mlp-{_sha256(manifest)[:12]}"
    checkpoint = {
        "format_version": "1.0",
        "model_id": model_id,
        "state_dict": best_state,
        "feature_config": encoder.config.to_dict(),
        "feature_names": list(encoder.feature_names),
        "model_config": {
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
        },
        "action_order": [item.value for item in ACTION_ORDER],
        "training": {
            "manifest_sha256": _sha256(manifest),
            "best_epoch": best_epoch,
            "config": asdict(config),
        },
    }
    torch.save(checkpoint, output / "best_policy.pt")
    write_repair_manifest(train_records, output / "train.jsonl")
    write_repair_manifest(select_split(records, "validation"), output / "validation.jsonl")
    write_repair_manifest(test_records, output / "test.jsonl")
    report = {
        "model_id": model_id,
        "device": device,
        "feature_dimension": encoder.dimension,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "dataset_audit": audit.to_dict(),
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
