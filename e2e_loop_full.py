import sys
sys.path.insert(0, "/root/PhysGenLoop-")
sys.path.insert(0, "/root/PhysGenLoop-/src")

from pavg_critic import PhysicsCritic
from physgenloop.contracts import LoopConfig
from physgenloop.controller import LoopController
from physgenloop.critic_adapter import PhysicsCriticAdapter
from physgenloop.repairer import InstructionPromptRepairer
from physgenloop.selector import EvidenceAwareSelector

from generators.wanphysics.adapter import WanPhysicsGenerator

# 不覆盖 num_frames/height/width/fps，使用 wan_generator.py 中的官方推荐默认值
generator = WanPhysicsGenerator(
    model_path="/root/PhysGenLoop-/models/wan2.2_ti2v_5b",
    device="cuda",
    output_root="/root/PhysGenLoop-/outputs",
)
critic = PhysicsCriticAdapter(PhysicsCritic())
repairer = InstructionPromptRepairer()
selector = EvidenceAwareSelector()

controller = LoopController(
    generator=generator,
    critic=critic,
    repairer=repairer,
    selector=selector,
    config=LoopConfig(max_rounds=1, candidates_per_round=1, acceptance_score=0.8, base_seed=42),
)

result = controller.run(prompt="a red ball rolling on a flat table")
print("stop_reason:", result.stop_reason)
print("best.candidate_id:", result.best.candidate.candidate_id)
print("best.video_path:", result.best.candidate.video_path)
print("best.report.decision:", result.best.report.decision)
print("best.report.physics_score:", result.best.report.physics_score)
