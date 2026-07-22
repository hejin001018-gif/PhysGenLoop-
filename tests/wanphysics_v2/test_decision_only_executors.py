"""decision-only executors：不二次调用 Policy + 四动作 + 失败。"""
from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import ExecutionRequest, RepairDecision, LocalEditTarget
from generators.wanphysics.v2.executors import (
    DecisionPromptRepairExecutor, OriginalPromptGlobalRegenerationExecutor,
    MaskSequenceLocalEditingExecutor, AuditedRejectExecutor,
)


class _Gen:
    def __init__(self):
        self.calls = 0
    def generate(self, *, prompt, physics_plan, seed):
        self.calls += 1
        return GeneratedCandidate(candidate_id=f"g-{seed}", video_path=f"/tmp/g-{seed}.mp4", prompt=prompt, seed=seed)


class _V:
    object = "ball"; category = "gravity"; start_frame = 1; end_frame = 3; critical_frames = (1, 2)
    reason = ""; repair_instruction = ""; evidence = {}


class _Report:
    violations = (_V(),)


def _decision(action, lt=None):
    return RepairDecision(action=RepairAction(action), confidence=0.5, instruction="do it",
                          action_probabilities={a.value: 0.25 for a in RepairAction},
                          per_action_values={a.value: 0.0 for a in RepairAction}, local_target=lt)


def _req(decision, **kw):
    return ExecutionRequest(decision=decision, candidate=GeneratedCandidate(candidate_id="src", video_path="/tmp/s.mp4", prompt="orig", seed=0),
                            critic_report=_Report(), prompt="current", physics_plan=None, seed=7, **kw)


def test_prompt_executor_no_second_policy_call():
    ex = DecisionPromptRepairExecutor(generator=_Gen())
    res = ex.execute(_req(_decision("prompt_repair")))
    assert res.status == "succeeded"
    assert ex.policy_call_count == 0
    assert res.metadata["policy_call_count"] == 0


def test_global_uses_original_prompt():
    ex = OriginalPromptGlobalRegenerationExecutor(generator=_Gen())
    res = ex.execute(_req(_decision("global_regeneration"), metadata={"original_prompt": "ORIG"}))
    assert res.next_prompt == "ORIG"


def test_local_requires_mask():
    ex = MaskSequenceLocalEditingExecutor(editor=lambda r: None)
    try:
        ex.execute(_req(_decision("local_editing", LocalEditTarget(parent_candidate_id="src", critical_frames=()))))
        assert False
    except ValueError:
        pass


def test_reject_terminal():
    class _Sel:
        def select(self, evals):
            return evals[-1]
    class _Eval:
        candidate = GeneratedCandidate(candidate_id="h1", video_path="/tmp/h.mp4", prompt="p", seed=1)
    ex = AuditedRejectExecutor(selector=_Sel())
    res = ex.execute(_req(_decision("reject"), history=(_Eval(),)))
    assert res.terminal and res.status == "rejected"
