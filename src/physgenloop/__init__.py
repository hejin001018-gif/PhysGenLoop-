"""PhysGenLoop 跨组件编排层的稳定公开 API。"""

from .contracts import (
    CandidateEvaluation,
    GeneratedCandidate,
    LoopConfig,
    LoopResult,
    LoopRound,
)
from .controller import LoopController
from .critic_adapter import PhysicsCriticAdapter
from .generator import DeterministicFakeGenerator
from .repairer import InstructionPromptRepairer
from .selector import EvidenceAwareSelector
from .learning_repair import (
    ActionValueDecisionPolicy,
    ExecutorRegistry,
    HeuristicRepairPolicy,
    LearningRepairAgent,
    LearningRepairLoopRunner,
    LearningRepairPromptAdapter,
    RepairAction,
    RepairContext,
    RepairDecision,
    RepairMemory,
    TorchActionValuePolicy,
    TorchMLPRepairPolicy,
)

__all__ = [
    "CandidateEvaluation",
    "ActionValueDecisionPolicy",
    "DeterministicFakeGenerator",
    "EvidenceAwareSelector",
    "ExecutorRegistry",
    "GeneratedCandidate",
    "HeuristicRepairPolicy",
    "InstructionPromptRepairer",
    "LearningRepairAgent",
    "LearningRepairLoopRunner",
    "LearningRepairPromptAdapter",
    "LoopConfig",
    "LoopController",
    "LoopResult",
    "LoopRound",
    "PhysicsCriticAdapter",
    "RepairAction",
    "RepairContext",
    "RepairDecision",
    "RepairMemory",
    "TorchActionValuePolicy",
    "TorchMLPRepairPolicy",
]
