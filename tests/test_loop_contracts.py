import pytest

from pavg_critic.schemas import CriticReport, PhysicsPlan

from physgenloop.contracts import GeneratedCandidate, LoopConfig
from physgenloop.critic_adapter import PhysicsCriticAdapter
from physgenloop.generator import DeterministicFakeGenerator


def test_loop_config_rejects_unbounded_or_empty_runs():
    with pytest.raises(ValueError, match="max_rounds"):
        LoopConfig(max_rounds=0)
    with pytest.raises(ValueError, match="candidates_per_round"):
        LoopConfig(candidates_per_round=0)
    with pytest.raises(ValueError, match="acceptance_score"):
        LoopConfig(acceptance_score=1.1)


def test_generated_candidate_requires_identity_and_video_path():
    with pytest.raises(ValueError, match="candidate_id"):
        GeneratedCandidate(candidate_id="", video_path="fake://a", prompt="p", seed=1)
    with pytest.raises(ValueError, match="video_path"):
        GeneratedCandidate(candidate_id="a", video_path="", prompt="p", seed=1)


def test_fake_generator_is_deterministic_and_explicitly_fake():
    generator = DeterministicFakeGenerator()
    plan = PhysicsPlan(objects=("ball",), expected_events=("fall",))
    first = generator.generate(prompt="A ball falls.", physics_plan=plan, seed=42)
    second = generator.generate(prompt="A ball falls.", physics_plan=plan, seed=42)
    assert first == second
    assert first.video_path.startswith("fake://")
    assert first.metadata == {
        "backend": "deterministic_fake",
        "is_real_video": False,
    }


def test_critic_adapter_builds_request_from_candidate_and_plan():
    class RecordingCritic:
        def __init__(self):
            self.request = None

        def analyze(self, request):
            self.request = request
            return CriticReport(
                is_physical=True,
                decision="physical",
                physics_score=1.0,
                confidence=1.0,
            )

    critic = RecordingCritic()
    plan = PhysicsPlan(objects=("ball",), expected_events=("fall",))
    candidate = GeneratedCandidate("c1", "video.mp4", "candidate prompt", 7)
    result = PhysicsCriticAdapter(critic).evaluate(
        candidate, prompt="loop prompt", physics_plan=plan
    )
    assert result.decision == "physical"
    assert critic.request.video_path == "video.mp4"
    assert critic.request.prompt == "loop prompt"
    assert critic.request.physics_plan is plan
