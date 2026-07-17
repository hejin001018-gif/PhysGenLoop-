"""Per-action value Policy used after actual RepairTrial collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physgenloop.learning_repair.contracts import ACTION_ORDER, RepairAction, RepairContext
from physgenloop.learning_repair.features import FeatureConfig, ReportFeatureEncoder
from physgenloop.learning_repair.policy import require_torch, resolve_device

from .baselines import HeuristicDecisionPolicy, _target
from .compatibility import CompatibilityManifest
from .contracts import RepairDecision
from .selector import RepairSelector


POLICY_FORMAT_VERSION = "repair-action-value-policy/2.1"


def build_action_value_mlp(
    torch,
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
):
    nn = torch.nn

    class ActionValueMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            width = input_dim
            for hidden in hidden_dims:
                layers.extend(
                    (
                        nn.Linear(width, hidden),
                        nn.LayerNorm(hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
                width = hidden
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
            self.action_head = nn.Linear(width, len(ACTION_ORDER))
            self.value_head = nn.Linear(width, len(ACTION_ORDER))

        def forward(self, features):
            hidden = self.backbone(features)
            # RewardSpec can produce utilities above 1.0 (physics gain plus
            # semantic/quality bonuses).  A tanh output silently made those valid
            # targets unrepresentable, so action values intentionally use a linear
            # head; capability masking still constrains executable actions.
            return self.action_head(hidden), self.value_head(hidden)

    return ActionValueMLP()


@dataclass(frozen=True)
class ActionValuePrediction:
    action_probabilities: dict[RepairAction, float]
    per_action_values: dict[RepairAction, float]
    model_id: str


class TorchActionValuePolicy:
    def __init__(
        self,
        model,
        *,
        encoder: ReportFeatureEncoder,
        torch,
        device: str,
        model_id: str,
        compatibility_id: str,
        selection_mode: str = "action_value",
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.torch = torch
        self.device = device
        self.model_id = model_id
        self.compatibility_id = compatibility_id
        self.selection_mode = str(selection_mode)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        compatibility_manifest: CompatibilityManifest | None = None,
    ) -> "TorchActionValuePolicy":
        torch = require_torch()
        resolved = resolve_device(torch, device)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=resolved, weights_only=False
            )
        except TypeError:  # pragma: no cover - PyTorch < 2.6
            checkpoint = torch.load(checkpoint_path, map_location=resolved)
        if checkpoint.get("format_version") != POLICY_FORMAT_VERSION:
            raise ValueError("unsupported action-value policy checkpoint")
        if tuple(checkpoint.get("action_order", ())) != tuple(
            item.value for item in ACTION_ORDER
        ):
            raise ValueError("checkpoint action order mismatch")
        if compatibility_manifest is not None:
            compatibility_manifest.assert_checkpoint(checkpoint)
        encoder = ReportFeatureEncoder(
            FeatureConfig.from_dict(checkpoint["feature_config"])
        )
        model_config = checkpoint["model_config"]
        model = build_action_value_mlp(
            torch,
            input_dim=encoder.dimension,
            hidden_dims=tuple(int(item) for item in model_config["hidden_dims"]),
            dropout=float(model_config["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(resolved)
        model.eval()
        return cls(
            model,
            encoder=encoder,
            torch=torch,
            device=resolved,
            model_id=str(checkpoint.get("model_id", Path(checkpoint_path).stem)),
            compatibility_id=str(checkpoint.get("compatibility_id", "unknown")),
            selection_mode=str(checkpoint.get("selection_mode", "action_value")),
        )

    def predict(
        self,
        critic_report: Any,
        *,
        context: RepairContext | None = None,
    ) -> ActionValuePrediction:
        features = self.encoder.encode(critic_report, context)
        tensor = self.torch.tensor(
            [features], dtype=self.torch.float32, device=self.device
        )
        with self.torch.inference_mode():
            logits, values = self.model(tensor)
            probabilities = self.torch.softmax(logits[0], dim=-1).cpu().tolist()
            predicted_values = values[0].cpu().tolist()
        return ActionValuePrediction(
            action_probabilities={
                action: float(value)
                for action, value in zip(ACTION_ORDER, probabilities)
            },
            per_action_values={
                action: float(value)
                for action, value in zip(ACTION_ORDER, predicted_values)
            },
            model_id=self.model_id,
        )


class ActionValueDecisionPolicy:
    """Select by calibrated per-action utility after applying backend masks."""

    def __init__(
        self,
        policy: TorchActionValuePolicy,
        *,
        probability_weight: float = 0.15,
        minimum_confidence: float = 0.35,
        fallback: HeuristicDecisionPolicy | None = None,
        selector: RepairSelector | None = None,
    ) -> None:
        self.policy = policy
        self.selector = selector or RepairSelector(
            probability_weight=probability_weight,
            minimum_confidence=minimum_confidence,
        )
        self.fallback = fallback or HeuristicDecisionPolicy(
            compatibility_id=policy.compatibility_id
        )

    def decide(
        self,
        *,
        critic_report: Any,
        candidate: Any,
        prompt: str,
        context: RepairContext,
    ) -> RepairDecision:
        prediction = self.policy.predict(critic_report, context=context)
        selection_mode = getattr(self.policy, "selection_mode", "action_value")
        selection = self.selector.select(
            action_probabilities=prediction.action_probabilities,
            per_action_values=prediction.per_action_values,
            context=context,
            selection_mode=selection_mode,
        )
        if selection.abstained:
            fallback = self.fallback.decide(
                critic_report=critic_report,
                candidate=candidate,
                prompt=prompt,
                context=context,
            )
            payload = fallback.to_dict()
            payload["fallback_reason"] = selection.fallback_reason
            payload["abstained"] = True
            return RepairDecision.from_dict(payload)
        return RepairDecision(
            action=selection.action,
            confidence=selection.confidence,
            instruction="Execute the highest calibrated, available repair action.",
            action_probabilities=selection.probabilities,
            per_action_values=prediction.per_action_values,
            parameters={"original_prompt": prompt, "selection": selection_mode},
            local_target=(
                _target(critic_report, candidate)
                if selection.action is RepairAction.LOCAL_EDITING
                else None
            ),
            source=prediction.model_id,
            compatibility_id=self.policy.compatibility_id,
        )
