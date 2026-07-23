from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import ExecutionResult, LocalEditTarget, RepairDecision

from generators.wanphysics.v2.runner import ActionAwareRunnerV2, RunnerConfig


class _Violation:
    object = "ball"
    category = "gravity_violation"
    critical_frames = (1, 2)
    repair_instruction = "fix gravity"


class _Report:
    def __init__(self, physical=False, available=True):
        self.decision = "physical" if physical else "violation" if available else "unknown"
        self.physics_score = 0.9 if physical else 0.2
        self.confidence = 0.8
        self.coverage = 0.8
        self.diagnostics = {}
        self.evidence_bundles = ()
        self.violations = () if physical else (_Violation(),)

    def to_dict(self):
        return {
            "decision": self.decision,
            "physics_score": self.physics_score,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "diagnostics": self.diagnostics,
            "violations": [],
        }


class _Generator:
    def generate(self, *, prompt, seed):
        return GeneratedCandidate(f"c{seed}", f"/tmp/c{seed}.mp4", prompt, seed, {"num_frames": 10})


class _Critic:
    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = 0

    def evaluate(self, candidate, *, prompt):
        report = self.reports[min(self.calls, len(self.reports) - 1)]
        self.calls += 1
        return report


class _Selector:
    def select(self, evaluations):
        return max(evaluations, key=lambda item: item.report.physics_score)


class _Registry:
    def __init__(self, fail=False):
        self.fail = fail

    def execute(self, request):
        if request.decision.action is RepairAction.REJECT:
            return ExecutionResult(
                RepairAction.REJECT,
                "rejected",
                "reject",
                candidate=request.candidate,
                terminal=True,
            )
        if self.fail:
            return ExecutionResult(
                request.decision.action,
                "failed",
                "executor",
                failure_reason="boom",
            )
        candidate = _Generator().generate(prompt=request.prompt + " fixed", seed=request.seed)
        return ExecutionResult(
            request.decision.action,
            "succeeded",
            "executor",
            candidate=candidate,
            next_prompt=candidate.prompt,
        )


def _decision(action=RepairAction.PROMPT_REPAIR):
    probabilities = {item: 0.1 for item in RepairAction}
    probabilities[action] = 0.8
    total = sum(probabilities.values())
    probabilities = {item: value / total for item, value in probabilities.items()}
    target = None
    if action is RepairAction.LOCAL_EDITING:
        target = LocalEditTarget("c", ("ball",), 1, 2, (1, 2), "manifest.json")
    return RepairDecision(
        action,
        probabilities[action],
        "fix",
        probabilities,
        probabilities,
        local_target=target,
        source="test_policy",
    )


def _runner(critic, decider, *, caps=None, registry=None, max_rounds=2):
    return ActionAwareRunnerV2(
        generator=_Generator(),
        critic=critic,
        decider=decider,
        executor_registry=registry or _Registry(),
        selector=_Selector(),
        config=RunnerConfig(max_rounds=max_rounds),
        capability_fn=lambda: caps
        or {"prompt_repair": True, "local_editing": False, "reject": True},
        side_score_fn=lambda candidate: {
            "semantic_score": 0.9,
            "original_prompt_semantic_score": 0.9,
            "quality_score": 0.9,
        },
    )


def test_rejected_candidate_repairs_and_accepts_without_duplicate_critic():
    critic = _Critic([_Report(False), _Report(True)])
    calls = []
    runner = _runner(critic, lambda report, evaluation, context: calls.append(context) or _decision())
    result = runner.run(sample_id="s1", prompt="p")
    assert result.final_state == "ACCEPTED"
    assert critic.calls == 2
    assert len(calls) == 1
    assert result.rounds[0].critic_before is not None
    assert result.rounds[0].critic_after is not None


def test_unavailable_never_calls_policy():
    calls = []
    runner = _runner(
        _Critic([_Report(available=False)]),
        lambda *args: calls.append(1) or _decision(),
    )
    result = runner.run(sample_id="s1", prompt="p")
    assert result.final_state == "EVALUATION_FAILED"
    assert calls == []


def test_guard_blocked_action_uses_audited_reject():
    runner = _runner(
        _Critic([_Report(False)]),
        lambda *args: _decision(RepairAction.LOCAL_EDITING),
        caps={"prompt_repair": True, "local_editing": False, "reject": True},
    )
    result = runner.run(sample_id="s1", prompt="p")
    assert result.final_state == "REJECTED"
    assert result.rounds[0].policy_action == "local_editing"
    assert result.rounds[0].executed_action == "reject"


def test_execution_failure_is_not_completed():
    runner = _runner(
        _Critic([_Report(False)]),
        lambda *args: _decision(),
        registry=_Registry(fail=True),
    )
    assert runner.run(sample_id="s1", prompt="p").final_state == "EXECUTION_FAILED"


def test_max_rounds_uses_cached_after_evaluation():
    critic = _Critic([_Report(False)])
    runner = _runner(critic, lambda *args: _decision(), max_rounds=2)
    result = runner.run(sample_id="s1", prompt="p")
    assert result.final_state == "MAX_ROUNDS"
    assert result.final_candidate_disposition == "best_effort"
    assert critic.calls == 3
