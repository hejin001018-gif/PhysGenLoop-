"""Learning Repair Agent 的公开 API。"""

from .agent import AgentConfig, LearningRepairAgent, LearningRepairPromptAdapter
from .contracts import (
    ACTION_ORDER,
    CandidateRecord,
    ExecutionRequest,
    ExecutionResult,
    LearningTargetV1,
    LocalEditTarget,
    PolicyPrediction,
    RepairAction,
    RepairContext,
    RepairDecision,
    RepairDecisionV1,
    RepairExample,
    RepairRunResult,
    RepairTrialV1,
    ScoreBundle,
)
from .dataset import (
    DatasetAudit,
    audit_dataset,
    collect_repair_samples,
    grouped_split,
    load_repair_manifest,
    select_split,
    write_repair_manifest,
)
from .features import FeatureConfig, ReportFeatureEncoder, normalize_category
from .memory import MemoryMatch, RepairMemory
from .labeling import (
    RepairTrial,
    RewardConfig,
    TrialSelection,
    build_labeled_example,
    select_best_trial,
)
from .policy import HeuristicRepairPolicy, RepairPolicy, TorchMLPRepairPolicy
from .release import export_release
from .training import TrainConfig, evaluate_policy, load_train_config, train_policy
from .baselines import (
    CategoryOnlyPolicy,
    DecisionPolicy,
    HeuristicDecisionPolicy,
    adapt_legacy_decision,
)
from .campaign import ActualTrialCampaign, CampaignResult, RewardSpec, targets_from_trials
from .cloud_campaign import CloudBackendBundle, run_frozen_campaign
from .compatibility import CompatibilityError, CompatibilityManifest
from .executors import (
    ExecutorRegistry,
    GlobalRegenerationExecutor,
    LocalEditingExecutor,
    PromptRepairExecutor,
    RejectExecutor,
    RepairExecutor,
)
from .manifests import CampaignItem, FrozenCampaignManifest
from .memory_policy import ActualTrialMemory, BlendedValuePredictor, MemoryValuePredictor
from .recording import JsonlTrialRecorder, VersionedMemoryWriter
from .runner import LearningRepairLoopRunner, RunnerConfig
from .value_policy import (
    ActionValueDecisionPolicy,
    ActionValuePrediction,
    TorchActionValuePolicy,
)
from .selector import RepairSelection, RepairSelector

__all__ = [
    "ACTION_ORDER",
    "ActionValueDecisionPolicy",
    "ActionValuePrediction",
    "ActualTrialCampaign",
    "ActualTrialMemory",
    "AgentConfig",
    "BlendedValuePredictor",
    "CampaignItem",
    "CampaignResult",
    "CandidateRecord",
    "CategoryOnlyPolicy",
    "CloudBackendBundle",
    "CompatibilityError",
    "CompatibilityManifest",
    "DatasetAudit",
    "DecisionPolicy",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutorRegistry",
    "FeatureConfig",
    "FrozenCampaignManifest",
    "GlobalRegenerationExecutor",
    "HeuristicRepairPolicy",
    "HeuristicDecisionPolicy",
    "JsonlTrialRecorder",
    "LearningRepairAgent",
    "LearningRepairLoopRunner",
    "LearningRepairPromptAdapter",
    "LearningTargetV1",
    "LocalEditTarget",
    "LocalEditingExecutor",
    "MemoryMatch",
    "MemoryValuePredictor",
    "PolicyPrediction",
    "PromptRepairExecutor",
    "RepairAction",
    "RepairContext",
    "RepairDecision",
    "RepairDecisionV1",
    "RepairExample",
    "RepairExecutor",
    "RepairMemory",
    "RepairRunResult",
    "RepairSelection",
    "RepairSelector",
    "RepairTrial",
    "RepairTrialV1",
    "RepairPolicy",
    "RejectExecutor",
    "ReportFeatureEncoder",
    "RewardConfig",
    "RewardSpec",
    "RunnerConfig",
    "ScoreBundle",
    "TorchActionValuePolicy",
    "TorchMLPRepairPolicy",
    "TrainConfig",
    "TrialSelection",
    "VersionedMemoryWriter",
    "adapt_legacy_decision",
    "audit_dataset",
    "collect_repair_samples",
    "build_labeled_example",
    "evaluate_policy",
    "export_release",
    "grouped_split",
    "load_repair_manifest",
    "load_train_config",
    "normalize_category",
    "select_split",
    "select_best_trial",
    "run_frozen_campaign",
    "targets_from_trials",
    "train_policy",
    "write_repair_manifest",
]
