"""P0 差距补全测试：因果链 / trial execution-first / checkpoint gate / semantic 结构化。"""
import sys
sys.path.insert(0, "/root/PhysGenLoop-")
sys.path.insert(0, "/root/PhysGenLoop-/src")

from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import ExecutionResult, RepairDecision, LocalEditTarget
from generators.wanphysics.v2.runner import ActionAwareRunnerV2, RunnerConfig


# ── mock 组件 ────────────────────────────────────────────────────────────────
class _Gen:
    def __init__(self): self.n = 0
    def generate(self, *, prompt, physics_plan, seed):
        self.n += 1
        return GeneratedCandidate(candidate_id=f"g{seed}-{self.n}", video_path=f"/tmp/g{seed}.mp4", prompt=prompt, seed=seed, metadata={"num_frames": 8})

class _Rep:
    def __init__(self, phys): self.decision = "physical" if phys >= 0.8 else "violation"; self.is_physical = phys >= 0.8
    def _f(self): return self
    physics_score = 0.2
    confidence = 0.7; coverage = 0.8; diagnostics = {}; violations = ()

def _rep(phys):
    r = _Rep(phys); r.physics_score = phys; return r

class _Critic:
    def __init__(self, seq): self.seq = list(seq); self.i = 0
    def evaluate(self, cand, *, prompt, physics_plan):
        v = self.seq[min(self.i, len(self.seq)-1)]; self.i += 1
        return _rep(v)

class _Sel:
    def select(self, evals): return max(evals, key=lambda e: e.report.physics_score)

class _Reg:
    def execute(self, req):
        new = GeneratedCandidate(candidate_id=f"after-{req.seed}", video_path=f"/tmp/after-{req.seed}.mp4", prompt=req.prompt, seed=req.seed, metadata={"num_frames": 8})
        return ExecutionResult(action=req.decision.action, status="succeeded", backend_id="m", candidate=new, next_prompt=req.prompt)

def _dec(action="prompt_repair"):
    probs = {a.value: (1.0 if a.value == action else 0.0) for a in RepairAction}
    tot = sum(probs.values()); probs = {k: v/tot for k, v in probs.items()}
    return RepairDecision(action=RepairAction(action), confidence=0.5, instruction="", action_probabilities=probs, per_action_values={a.value: 0.0 for a in RepairAction})


# ── P0-01 因果链：after candidate 成为 current，before/after 配对 ──────────────
def test_after_candidate_becomes_current_and_paired():
    gen = _Gen()
    runner = ActionAwareRunnerV2(
        generator=gen, critic=_Critic([0.2, 0.3, 0.4]), decider=lambda r, e: _dec("prompt_repair"),
        executor_registry=_Reg(), selector=_Sel(),
        config=RunnerConfig(max_rounds=3, default_total_frames=8),
    )
    res = runner.run(sample_id="s1", prompt="a red ball", physics_plan=None)
    # generator 只生成 1 次初始候选（后续由 executor 产 after）
    assert gen.n == 1, f"generator 应只调用1次，实际 {gen.n}"
    re_rounds = [r for r in res.rounds if r.state == "RE_EVALUATED"]
    assert re_rounds
    r0 = re_rounds[0]
    # before/after 配对且 after_physics 记录
    assert r0.before_candidate_id != r0.after_candidate_id
    assert r0.after_physics is not None
    assert r0.execution_id


def test_reject_no_after_candidate():
    class _RejReg:
        def execute(self, req):
            return ExecutionResult(action=RepairAction.REJECT, status="rejected", backend_id="m",
                                   candidate=req.candidate, terminal=True)
    runner = ActionAwareRunnerV2(
        generator=_Gen(), critic=_Critic([0.2]), decider=lambda r, e: _dec("reject"),
        executor_registry=_RejReg(), selector=_Sel(),
        config=RunnerConfig(max_rounds=3, default_total_frames=8),
    )
    res = runner.run(sample_id="s1", prompt="p", physics_plan=None)
    assert res.stop_reason == "rejected"
    rej = [r for r in res.rounds if r.state == "REJECTED"][0]
    assert rej.after_candidate_id is None  # reject 不产生 after
    assert rej.terminal_reason == "rejected_by_design"


# ── P0-02 plan gate ──────────────────────────────────────────────────────────
def test_plan_gate_enforce_blocks_empty_plan():
    # enforce + require_plan + 空 plan + physical → 不接受
    runner = ActionAwareRunnerV2(
        generator=_Gen(), critic=_Critic([0.9]), decider=lambda r, e: _dec("prompt_repair"),
        executor_registry=_Reg(), selector=_Sel(),
        config=RunnerConfig(max_rounds=2, default_total_frames=8, acceptance_mode="enforce", require_plan=True),
    )
    res = runner.run(sample_id="s1", prompt="a ball collides with a wall", physics_plan=None)
    # 空 plan 被 plan gate 阻断，不应首轮 accepted
    assert res.stop_reason != "accepted" or res.rounds[0].state != "ACCEPTED"


# ── P0-03 trial execution-first ──────────────────────────────────────────────
def test_trial_execution_first_pairing():
    from agents.wanphysics.run_videophy2_loop_v2 import _assemble_trials
    gen = _Gen()
    runner = ActionAwareRunnerV2(
        generator=gen, critic=_Critic([0.2, 0.9]), decider=lambda r, e: _dec("prompt_repair"),
        executor_registry=_Reg(), selector=_Sel(),
        config=RunnerConfig(max_rounds=3, default_total_frames=8),
    )
    res = runner.run(sample_id="s1", prompt="p", physics_plan=None)
    trials = _assemble_trials("s1", res, "p")
    assert trials
    t = trials[0]
    # trial_id == execution_id；不是 round 索引
    assert "exec" in t["trial_id"]
    # before/after 来自同一 execution
    assert t["execution"]["execution_id"] == t["trial_id"]
    # success 复合定义：physics 0.2→0.9 提升 → successful
    assert t["successful"] is True


# ── P0-08 checkpoint gate ────────────────────────────────────────────────────
def test_checkpoint_gate_blocks_deployment(tmp_path):
    from generators.wanphysics.v2.checkpoint_gate import evaluate_checkpoint_gate
    # 空目录 → deployment 拒绝
    r = evaluate_checkpoint_gate(tmp_path, mode="deployment", allow_proxy_override=False)
    assert r.allow_load is False
    assert r.proxy_only is True
    # proxy_research + override → 允许但 proxy_only
    r2 = evaluate_checkpoint_gate(tmp_path, mode="proxy_research", allow_proxy_override=True)
    assert r2.allow_load is True and r2.proxy_only is True
    # proxy_research 无 override → 拒绝
    r3 = evaluate_checkpoint_gate(tmp_path, mode="proxy_research", allow_proxy_override=False)
    assert r3.allow_load is False


# ── P0-06 semantic scorer 结构化 + 不可用 ────────────────────────────────────
def test_semantic_scorer_structured_unavailable():
    from generators.wanphysics.v2.scorers_semantic import VlmSemanticScorer
    sc = VlmSemanticScorer(base_url="http://localhost:19998/v1")
    res = sc.score_structured(prompt="p", video_path="/tmp/nonexist.mp4")
    assert res.available is False
    assert res.score is None
    assert res.degraded is True
    assert "vllm_unavailable" in res.degraded_reasons
    # 结构化 to_dict 有独立字段
    d = res.to_dict()
    assert "entity_preservation" in d and "backend" in d
