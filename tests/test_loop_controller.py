from pavg_critic.schemas import CriticReport, PhysicsPlan, Violation

from physgenloop.contracts import (
    CandidateEvaluation,
    GeneratedCandidate,
    LoopConfig,
)
from physgenloop.controller import LoopController
from physgenloop.generator import DeterministicFakeGenerator
from physgenloop.repairer import InstructionPromptRepairer
from physgenloop.selector import EvidenceAwareSelector


def _report(
    decision: str,
    score: float,
    confidence: float,
    instruction: str = "",
) -> CriticReport:
    violations = ()
    if instruction:
        violations = (
            Violation(
                object="ball",
                category="gravity",
                start_frame=1,
                peak_frame=2,
                end_frame=3,
                critical_frames=(1, 2, 3),
                reason="bad",
                repair_instruction=instruction,
                evidence={},
            ),
        )
    return CriticReport(
        is_physical=decision == "physical",
        decision=decision,
        physics_score=score,
        confidence=confidence,
        violations=violations,
    )


def _candidate(name: str, report: CriticReport) -> CandidateEvaluation:
    generated = GeneratedCandidate(name, f"fake://{name}", "p", 1)
    return CandidateEvaluation(generated, report)


class ScriptedCritic:
    def __init__(self, reports):
        self.reports = iter(reports)

    def evaluate(self, candidate, *, prompt, physics_plan):
        return next(self.reports)


def test_repairer_appends_unique_structured_instructions():
    item_report = _report(
        "violation", 0.2, 0.9, "Keep the ball falling until contact."
    )
    repaired = InstructionPromptRepairer().repair(
        prompt="A ball falls.", report=item_report
    )
    assert repaired == (
        "A ball falls.\nPhysics correction: Keep the ball falling until contact."
    )


def test_repairer_keeps_prompt_when_no_actionable_instruction_exists():
    item_report = _report("unknown", 0.5, 0.1)
    assert (
        InstructionPromptRepairer().repair(prompt="A ball falls.", report=item_report)
        == "A ball falls."
    )


def test_selector_prefers_decision_then_score_and_is_stable():
    first = _candidate("first", _report("physical", 0.8, 0.6))
    second = _candidate("second", _report("physical", 0.8, 0.6))
    violation = _candidate("bad", _report("violation", 0.99, 1.0))
    assert EvidenceAwareSelector().select((first, second, violation)) is first


def test_selector_rejects_empty_evaluations():
    try:
        EvidenceAwareSelector().select(())
    except ValueError as error:
        assert "evaluations" in str(error)
    else:
        raise AssertionError("empty selection must fail")


def test_controller_stops_on_accepted_physical_candidate():
    controller = LoopController(
        generator=DeterministicFakeGenerator(),
        critic=ScriptedCritic((_report("physical", 0.9, 0.8),)),
        repairer=InstructionPromptRepairer(),
        selector=EvidenceAwareSelector(),
        config=LoopConfig(
            max_rounds=3,
            candidates_per_round=1,
            acceptance_score=0.8,
        ),
    )
    result = controller.run(prompt="A ball falls.")
    assert result.stop_reason == "accepted"
    assert len(result.history) == 1


def test_controller_resolves_prompt_plan_once_before_generation_and_critique():
    class RecordingGenerator:
        def __init__(self):
            self.plans = []

        def generate(self, *, prompt, physics_plan, seed):
            self.plans.append(physics_plan)
            return GeneratedCandidate("recorded", "fake://recorded", prompt, seed)

    class RecordingCritic:
        def __init__(self):
            self.plans = []

        def evaluate(self, candidate, *, prompt, physics_plan):
            self.plans.append(physics_plan)
            return _report("physical", 0.9, 0.8)

    generator = RecordingGenerator()
    critic = RecordingCritic()
    controller = LoopController(
        generator=generator,
        critic=critic,
        repairer=InstructionPromptRepairer(),
        selector=EvidenceAwareSelector(),
        config=LoopConfig(max_rounds=1, candidates_per_round=1),
    )
    result = controller.run(prompt="A ball falls.", physics_plan=PhysicsPlan())
    assert result.resolved_plan.expected_events == ("leave_support", "fall")
    assert generator.plans == [result.resolved_plan]
    assert critic.plans == [result.resolved_plan]


def test_controller_keeps_global_best_at_max_rounds():
    controller = LoopController(
        generator=DeterministicFakeGenerator(),
        critic=ScriptedCritic(
            (
                _report("violation", 0.4, 0.8, "Fix gravity."),
                _report("violation", 0.2, 0.9, "Fix contact."),
            )
        ),
        repairer=InstructionPromptRepairer(),
        selector=EvidenceAwareSelector(),
        config=LoopConfig(max_rounds=2, candidates_per_round=1),
    )
    plan = PhysicsPlan(objects=("ball",), expected_events=("fall",))
    result = controller.run(prompt="A ball falls.", physics_plan=plan)
    assert result.stop_reason == "max_rounds"
    assert result.best.report.physics_score == 0.4
    assert result.resolved_plan.objects == plan.objects
    assert result.resolved_plan.expected_events == plan.expected_events
    assert len(result.history) == 2
    assert result.history[1].prompt.endswith("Physics correction: Fix gravity.")


def test_controller_selects_best_of_k_within_one_round():
    controller = LoopController(
        generator=DeterministicFakeGenerator(),
        critic=ScriptedCritic(
            (
                _report("violation", 0.3, 0.9, "Fix gravity."),
                _report("physical", 0.9, 0.8),
            )
        ),
        repairer=InstructionPromptRepairer(),
        selector=EvidenceAwareSelector(),
        config=LoopConfig(max_rounds=1, candidates_per_round=2),
    )
    result = controller.run(prompt="A ball falls.", physics_plan=PhysicsPlan())
    assert len(result.history[0].evaluations) == 2
    assert result.best.report.decision == "physical"
    assert result.history[0].selected_candidate_id == result.best.candidate.candidate_id
