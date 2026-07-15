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

__all__ = [
    "CandidateEvaluation",
    "DeterministicFakeGenerator",
    "EvidenceAwareSelector",
    "GeneratedCandidate",
    "InstructionPromptRepairer",
    "LoopConfig",
    "LoopController",
    "LoopResult",
    "LoopRound",
    "PhysicsCriticAdapter",
]
