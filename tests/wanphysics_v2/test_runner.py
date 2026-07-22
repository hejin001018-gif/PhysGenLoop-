"""V2 runner 状态机四动作 + critic 失败 + edited→current。"""
from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import ExecutionResult, RepairDecision, LocalEditTarget
from generators.wanphysics.v2.runner import ActionAwareRunnerV2, RunnerConfig
from generators.wanphysics.v2.guardrails import GateThresholds


class _Gen:
    def generate(self, *, prompt, physics_plan, seed):
        return GeneratedCandidate(candidate_id=f"g-{seed}", video_path=f"/tmp/g-{seed}.mp4", prompt=prompt, seed=seed, metadata={"num_frames": 8})


class _Rep:
    def __init__(self, physical):
        self.decision = "physical" if physical else "violation"
        self.is_physical = physical; self.physics_score = 0.9 if physical else 0.2
        self.confidence = 0.7; self.coverage = 0.8; self.diagnostics = {}; self.violations = ()


class _Violation:
    object = "ball"
    start_frame = 1
    end_frame = 1
    critical_frames = (1,)
    evidence = {}


class _LocalRep(_Rep):
    def __init__(self, physical=False):
        super().__init__(physical)
        self.violations = (_Violation(),)


class _Critic:
    def __init__(self, accept_at=99, fail=False):
        self.accept_at = accept_at; self.fail = fail; self.n = 0
    def evaluate(self, candidate, *, prompt, physics_plan):
        self.n += 1
        if self.fail:
            return None
        return _Rep(self.n >= self.accept_at)


class _Sel:
    def select(self, evals):
        return max(evals, key=lambda e: e.report.physics_score)


def _dec(action):
    lt = LocalEditTarget(parent_candidate_id="m", critical_frames=(1,), mask_uri="x.png") if action == "local_editing" else None
    return RepairDecision(action=RepairAction(action), confidence=0.5, instruction="i",
                          action_probabilities={a.value: 0.25 for a in RepairAction},
                          per_action_values={a.value: 0.0 for a in RepairAction}, local_target=lt)


class _Reg:
    def __init__(self, action): self.action = action
    def supports(self, a): return True
    def execute(self, request):
        if request.decision.action == RepairAction.REJECT:
            cand = request.history[-1].candidate if request.history else request.candidate
            return ExecutionResult(action=RepairAction.REJECT, status="rejected", backend_id="m", candidate=cand, terminal=True)
        new = GeneratedCandidate(candidate_id=f"e-{request.seed}", video_path=f"/tmp/e-{request.seed}.mp4", prompt=request.prompt, seed=request.seed, metadata={"num_frames": 8})
        return ExecutionResult(action=request.decision.action, status="succeeded", backend_id="m", candidate=new, next_prompt=request.prompt)


def _run(force, accept_at=99, caps=None):
    r = ActionAwareRunnerV2(
        generator=_Gen(), critic=_Critic(accept_at=accept_at), decider=lambda rep, ev: _dec(force),
        executor_registry=_Reg(force), selector=_Sel(),
        config=RunnerConfig(max_rounds=3, thresholds=GateThresholds(), default_total_frames=8),
        capability_fn=lambda: caps or {"prompt_repair": True, "global_regeneration": True, "local_editing": True, "reject": True},
        mask_valid_fn=lambda rep, c: force == "local_editing",
    )
    return r.run(sample_id="s1", prompt="a red ball rolls", physics_plan=None)


def test_accept_first_round():
    res = _run("prompt_repair", accept_at=1)
    assert res.stop_reason == "accepted"


def test_reject_terminates():
    res = _run("reject")
    assert res.stop_reason == "rejected"


def test_prompt_repair_runs_rounds():
    res = _run("prompt_repair")
    assert res.stop_reason == "max_rounds"
    assert len(res.rounds) >= 1


def test_local_editing_edited_becomes_current():
    res = _run("local_editing")
    # P0-01：after candidate 经 Re-Critic/Re-Gate（RE_EVALUATED），并写 before/after 配对
    re_rounds = [r for r in res.rounds if r.state == "RE_EVALUATED"]
    assert re_rounds, "应有 RE_EVALUATED 轮次"
    r0 = re_rounds[0]
    assert r0.before_candidate_id and r0.after_candidate_id
    assert r0.before_candidate_id != r0.after_candidate_id
    assert r0.execution_id is not None


def test_guard_local_override_injects_manifest_target():
    class _LocalCritic:
        def evaluate(self, candidate, *, prompt, physics_plan):
            return _LocalRep(False)

    class _CaptureReg:
        def __init__(self):
            self.seen = None
        def supports(self, a): return True
        def execute(self, request):
            self.seen = request.decision
            assert request.decision.action == RepairAction.LOCAL_EDITING
            assert request.decision.local_target is not None
            assert request.decision.local_target.mask_uri.endswith("mask_manifest.json")
            new = GeneratedCandidate(candidate_id=f"local-{request.seed}", video_path=f"/tmp/local-{request.seed}.mp4", prompt=request.prompt, seed=request.seed, metadata={"num_frames": 8})
            return ExecutionResult(action=request.decision.action, status="succeeded", backend_id="m", candidate=new, next_prompt=request.prompt)

    registry = _CaptureReg()
    target = LocalEditTarget(parent_candidate_id="g-42", objects=("ball",), critical_frames=(1,), mask_uri="/tmp/c1/mask_manifest.json")
    runner = ActionAwareRunnerV2(
        generator=_Gen(), critic=_LocalCritic(), decider=lambda rep, ev: _dec("prompt_repair"),
        executor_registry=registry, selector=_Sel(),
        config=RunnerConfig(max_rounds=2, thresholds=GateThresholds(), default_total_frames=8),
        capability_fn=lambda: {"prompt_repair": True, "global_regeneration": True, "local_editing": True, "reject": True},
        mask_valid_fn=lambda rep, c: True,
        local_target_fn=lambda rep, c: target,
    )
    runner.run(sample_id="s1", prompt="a red ball rolls", physics_plan=None)
    assert registry.seen is not None


def test_local_override_without_target_fails_closed():
    class _LocalCritic:
        def evaluate(self, candidate, *, prompt, physics_plan):
            return _LocalRep(False)

    class _NeverReg:
        def supports(self, a): return True
        def execute(self, request):  # pragma: no cover
            raise AssertionError("executor should not run without local target")

    runner = ActionAwareRunnerV2(
        generator=_Gen(), critic=_LocalCritic(), decider=lambda rep, ev: _dec("prompt_repair"),
        executor_registry=_NeverReg(), selector=_Sel(),
        config=RunnerConfig(max_rounds=2, thresholds=GateThresholds(), default_total_frames=8),
        capability_fn=lambda: {"prompt_repair": True, "global_regeneration": True, "local_editing": True, "reject": True},
        mask_valid_fn=lambda rep, c: True,
    )
    res = runner.run(sample_id="s1", prompt="a red ball rolls", physics_plan=None)
    assert res.stop_reason == "executor_failed"
    assert res.rounds[0].terminal_reason == "local_target_missing_or_invalid"


def test_critic_failure_stops():
    r = ActionAwareRunnerV2(
        generator=_Gen(), critic=_Critic(fail=True), decider=lambda rep, ev: _dec("prompt_repair"),
        executor_registry=_Reg("prompt_repair"), selector=_Sel(),
        config=RunnerConfig(max_rounds=2, default_total_frames=8),
    )
    res = r.run(sample_id="s1", prompt="p", physics_plan=None)
    assert res.stop_reason == "critic_failed"
