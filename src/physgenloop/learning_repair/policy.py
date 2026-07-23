"""Repair Policy 协议、确定性回退策略和 PyTorch 推理适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .contracts import ACTION_ORDER, PolicyPrediction, RepairAction, RepairContext
from .features import FeatureConfig, ReportFeatureEncoder


class RepairPolicy(Protocol):
    def predict(
        self, critic_report: Any, *, context: RepairContext | None = None
    ) -> PolicyPrediction: ...


def _report_value(report: Any, name: str, default: Any) -> Any:
    if hasattr(report, "to_dict"):
        report = report.to_dict()
    if isinstance(report, dict):
        return report.get(name, default)
    return default


class HeuristicRepairPolicy:
    """训练模型缺席或低置信时使用的可解释策略。"""

    def __init__(self, encoder: ReportFeatureEncoder | None = None) -> None:
        self.encoder = encoder or ReportFeatureEncoder()

    def predict(
        self, critic_report: Any, *, context: RepairContext | None = None
    ) -> PolicyPrediction:
        category = self.encoder.primary_category(critic_report)
        decision = str(_report_value(critic_report, "decision", "unknown"))
        score = float(_report_value(critic_report, "physics_score", 0.5))
        coverage = float(_report_value(critic_report, "coverage", 0.0))
        probabilities = {action: 0.05 for action in ACTION_ORDER}
        if decision == "unknown" or coverage < 0.25:
            probabilities[RepairAction.REJECT] = 0.8
        elif category in {
            "collision_violation",
            "trajectory_violation",
            "continuity_violation",
            "appearance_violation",
        }:
            probabilities[RepairAction.LOCAL_EDITING] = 0.7
            probabilities[RepairAction.PROMPT_REPAIR] = 0.15
        elif category in {
            "gravity_violation",
            "friction_violation",
            "contact_violation",
        }:
            probabilities[RepairAction.PROMPT_REPAIR] = 0.7
            probabilities[RepairAction.LOCAL_EDITING] = 0.15
        elif decision == "physical":
            probabilities[RepairAction.REJECT] = 0.8
        else:
            probabilities[RepairAction.PROMPT_REPAIR] = 0.55
            probabilities[RepairAction.REJECT] = 0.3
        return PolicyPrediction(
            probabilities,
            expected_gain=max(0.0, min(1.0, 1.0 - score)) * 0.5,
            model_id="heuristic-v1",
        )


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Learning Repair training/inference requires the train extra: "
            "`pip install -e .[train]`."
        ) from exc
    return torch


def resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {requested!r}, but CUDA is unavailable")
    return requested


def build_repair_mlp(
    torch,
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
):
    nn = torch.nn

    class RepairMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            current = input_dim
            for width in hidden_dims:
                layers.extend(
                    (
                        nn.Linear(current, width),
                        nn.LayerNorm(width),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
                current = width
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
            self.action_head = nn.Linear(current, len(ACTION_ORDER))
            self.gain_head = nn.Linear(current, 1)

        def forward(self, features):
            hidden = self.backbone(features)
            return self.action_head(hidden), torch.tanh(self.gain_head(hidden)).squeeze(-1)

    return RepairMLP()


class TorchMLPRepairPolicy:
    """从自描述 checkpoint 加载 MLP；导入包本身不要求安装 PyTorch。"""

    def __init__(
        self,
        model,
        *,
        encoder: ReportFeatureEncoder,
        torch,
        device: str,
        model_id: str,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.torch = torch
        self.device = device
        self.model_id = model_id

    @classmethod
    def load(
        cls, checkpoint_path: str | Path, *, device: str = "auto"
    ) -> "TorchMLPRepairPolicy":
        torch = require_torch()
        resolved_device = resolve_device(torch, device)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=resolved_device, weights_only=False
            )
        except TypeError:  # PyTorch < 2.6 compatibility
            checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        if checkpoint.get("format_version") != "1.0":
            raise ValueError("unsupported Repair Policy checkpoint format")
        encoder = ReportFeatureEncoder(
            FeatureConfig.from_dict(checkpoint["feature_config"])
        )
        model_config = checkpoint["model_config"]
        model = build_repair_mlp(
            torch,
            input_dim=encoder.dimension,
            hidden_dims=tuple(int(item) for item in model_config["hidden_dims"]),
            dropout=float(model_config["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(resolved_device)
        model.eval()
        return cls(
            model,
            encoder=encoder,
            torch=torch,
            device=resolved_device,
            model_id=str(checkpoint.get("model_id", Path(checkpoint_path).stem)),
        )

    def predict(
        self, critic_report: Any, *, context: RepairContext | None = None
    ) -> PolicyPrediction:
        features = self.encoder.encode(critic_report, context)
        tensor = self.torch.tensor(
            [features], dtype=self.torch.float32, device=self.device
        )
        with self.torch.inference_mode():
            logits, gain = self.model(tensor)
            probabilities = self.torch.softmax(logits[0], dim=-1).detach().cpu().tolist()
        return PolicyPrediction(
            {
                action: float(probability)
                for action, probability in zip(ACTION_ORDER, probabilities)
            },
            expected_gain=float(gain[0].detach().cpu()),
            model_id=self.model_id,
        )
