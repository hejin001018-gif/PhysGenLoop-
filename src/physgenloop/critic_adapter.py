"""将总控层候选转换为现有 PhysicsCritic 请求。"""

from __future__ import annotations

from pavg_critic import CriticRequest, PhysicsCritic
from pavg_critic.schemas import CriticReport, PhysicsPlan

from .contracts import GeneratedCandidate


class PhysicsCriticAdapter:
    """保持总控层不依赖 Critic 内部流水线细节。"""

    def __init__(self, critic: PhysicsCritic) -> None:
        self.critic = critic

    def evaluate(
        self,
        candidate: GeneratedCandidate,
        *,
        prompt: str,
        physics_plan: PhysicsPlan,
    ) -> CriticReport:
        return self.critic.analyze(
            CriticRequest(
                video_path=candidate.video_path,
                prompt=prompt,
                physics_plan=physics_plan,
            )
        )
