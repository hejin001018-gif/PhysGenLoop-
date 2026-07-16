"""生成、评价、修复与选择组件的可替换协议。"""

from __future__ import annotations

from typing import Protocol

from pavg_critic.planner import PhysicsPlanResolution
from pavg_critic.schemas import CriticReport, CriticRequest, PhysicsPlan

from .contracts import CandidateEvaluation, GeneratedCandidate


class VideoGenerator(Protocol):
    def generate(
        self, *, prompt: str, physics_plan: PhysicsPlan, seed: int
    ) -> GeneratedCandidate: ...


class PlanResolver(Protocol):
    def resolve(self, request: CriticRequest) -> PhysicsPlanResolution: ...


class CandidateCritic(Protocol):
    def evaluate(
        self,
        candidate: GeneratedCandidate,
        *,
        prompt: str,
        physics_plan: PhysicsPlan,
    ) -> CriticReport: ...


class PromptRepairer(Protocol):
    def repair(self, *, prompt: str, report: CriticReport) -> str: ...


class CandidateSelector(Protocol):
    def select(
        self, evaluations: tuple[CandidateEvaluation, ...]
    ) -> CandidateEvaluation: ...
