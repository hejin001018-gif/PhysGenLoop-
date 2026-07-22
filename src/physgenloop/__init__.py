"""Public API for the V2 WanPhysics loop.

The project now exposes only the contracts and selector required by the V2
full-chain runner. Legacy LoopController and fake-generator APIs were removed
from the public surface to keep one operational path.
"""

from .contracts import CandidateEvaluation, GeneratedCandidate
from .selector import EvidenceAwareSelector

__all__ = [
    "CandidateEvaluation",
    "EvidenceAwareSelector",
    "GeneratedCandidate",
]
