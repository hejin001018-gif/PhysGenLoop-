"""Minimal repair contracts used by the WanPhysics V2 loop."""

from .base_contracts import ACTION_ORDER, PolicyPrediction, RepairAction, RepairContext
from .compatibility import CompatibilityError, CompatibilityManifest
from .contracts import (
    CandidateRecord,
    ExecutionRequest,
    ExecutionResult,
    LearningTargetV1,
    LocalEditTarget,
    RepairDecision,
    RepairDecisionV1,
    RepairExample,
    RepairRunResult,
    RepairTrialV1,
    ScoreBundle,
)
from .executors import ExecutorRegistry, RepairExecutor
from .features import FeatureConfig, ReportFeatureEncoder, normalize_category
from .policy import HeuristicRepairPolicy, RepairPolicy, require_torch, resolve_device
from .selector import RepairSelection, RepairSelector
from .value_policy import (
    ActionValueDecisionPolicy,
    ActionValuePrediction,
    TorchActionValuePolicy,
)

__all__ = [
    "ACTION_ORDER",
    "ActionValueDecisionPolicy",
    "ActionValuePrediction",
    "CandidateRecord",
    "CompatibilityError",
    "CompatibilityManifest",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutorRegistry",
    "FeatureConfig",
    "HeuristicRepairPolicy",
    "LearningTargetV1",
    "LocalEditTarget",
    "PolicyPrediction",
    "RepairAction",
    "RepairContext",
    "RepairDecision",
    "RepairDecisionV1",
    "RepairExample",
    "RepairExecutor",
    "RepairPolicy",
    "RepairRunResult",
    "RepairSelection",
    "RepairSelector",
    "RepairTrialV1",
    "ReportFeatureEncoder",
    "ScoreBundle",
    "TorchActionValuePolicy",
    "normalize_category",
    "require_torch",
    "resolve_device",
]
