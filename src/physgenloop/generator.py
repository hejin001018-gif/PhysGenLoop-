"""视频生成器占位实现。"""

from __future__ import annotations

import hashlib
import json

from pavg_critic.schemas import PhysicsPlan

from .contracts import GeneratedCandidate


class DeterministicFakeGenerator:
    """不创建 MP4 的显式 fake，用于 CPU 测试和接口联调。"""

    def generate(
        self, *, prompt: str, physics_plan: PhysicsPlan, seed: int
    ) -> GeneratedCandidate:
        plan_json = json.dumps(
            physics_plan.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(
            f"{prompt}\0{plan_json}\0{seed}".encode("utf-8")
        ).hexdigest()[:12]
        return GeneratedCandidate(
            candidate_id=f"fake-{digest}",
            video_path=f"fake://{digest}.mp4",
            prompt=prompt,
            seed=seed,
            metadata={"backend": "deterministic_fake", "is_real_video": False},
        )
