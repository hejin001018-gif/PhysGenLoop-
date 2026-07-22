"""新增功能测试：repair_trace / trial 组装 / memory_status / scorer 接线 / force_action。"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pytest
import yaml


# ── repair_trace via hooks ───────────────────────────────────────────────────
def test_repair_trace_appended_on_execution(tmp_path):
    from generators.wanphysics.v2.artifacts import RunArtifacts
    from generators.wanphysics.v2.build_backends import _ArtifactHooks

    art = RunArtifacts(tmp_path / "run")
    hooks = _ArtifactHooks(art)
    # simulate decision → execution cycle
    class _Guard:
        policy_action = "prompt_repair"; final_action = "prompt_repair"
        scope = "global"; override_reason = None
        def to_dict(self): return {"policy_action": self.policy_action, "final_action": self.final_action}
    class _Dec:
        def to_dict(self): return {"action": "prompt_repair"}
    hooks.on_decision("s1", "c1", _Dec(), _Guard())
    class _ExecResult:
        status = "succeeded"; backend_id = "mock"; failure_reason = None
        terminal = False; cost = 0.1; latency_seconds = 1.2
    hooks.on_execution("s1", _ExecResult())
    trace_path = art.run_dir / "s1" / "repair_trace.jsonl"
    assert trace_path.exists()
    records = [json.loads(ln) for ln in trace_path.read_text().strip().splitlines()]
    assert len(records) == 1
    assert records[0]["final_action"] == "prompt_repair"
    assert records[0]["execution_status"] == "succeeded"


# ── CRITIC_FAILED writes trace ───────────────────────────────────────────────
def test_critic_failed_writes_trace(tmp_path):
    from generators.wanphysics.v2.artifacts import RunArtifacts
    from generators.wanphysics.v2.build_backends import _ArtifactHooks

    art = RunArtifacts(tmp_path / "run")
    hooks = _ArtifactHooks(art)
    hooks.on_state("s1", "CRITIC_FAILED", 0)
    trace = art.run_dir / "s1" / "repair_trace.jsonl"
    assert trace.exists()
    r = json.loads(trace.read_text().strip())
    assert r["event"] == "CRITIC_FAILED"


# ── memory_status written ────────────────────────────────────────────────────
def test_memory_status_written(tmp_path):
    from generators.wanphysics.v2.artifacts import RunArtifacts
    art = RunArtifacts(tmp_path / "run")
    art.write_memory_status({"memory_enabled": False, "memory_format": "empty", "memory_records": 0, "memory_path": None})
    assert (tmp_path / "run" / "memory_status.json").exists()


# ── resource_metrics written ─────────────────────────────────────────────────
def test_resource_metrics_written(tmp_path):
    from generators.wanphysics.v2.artifacts import RunArtifacts
    art = RunArtifacts(tmp_path / "run")
    art.append_resource_metrics("s1", {"candidate_id": "c1", "gpu_memory_before_mb": None, "critic_seconds": 5.0})
    p = art.sample_dir("s1") / "resource_metrics.jsonl"
    assert p.exists()
    r = json.loads(p.read_text().strip())
    assert r["candidate_id"] == "c1"


# ── trial append ─────────────────────────────────────────────────────────────
def test_trial_append(tmp_path):
    from generators.wanphysics.v2.artifacts import RunArtifacts
    art = RunArtifacts(tmp_path / "run")
    art.append_trial("s1", {"trial_id": "t1", "action": "prompt_repair"})
    p = art.sample_dir("s1") / "trials.jsonl"
    assert p.exists()


# ── CpuQualityScorer接入gate(side_score_fn) ───────────────────────────────────
def test_side_score_fn_passed_to_gate():
    """runner 接受 side_score_fn 并传给 gate（无 GPU：mock runner 验证参数）。"""
    from generators.wanphysics.v2.runner import ActionAwareRunnerV2, RunnerConfig

    calls = []

    class _Gen:
        def generate(self, *, prompt, physics_plan, seed):
            from physgenloop.contracts import GeneratedCandidate
            return GeneratedCandidate(candidate_id=f"g{seed}", video_path=f"/tmp/g{seed}.mp4", prompt=prompt, seed=seed, metadata={"num_frames": 4})

    class _Rep:
        decision = "violation"; is_physical = False; physics_score = 0.2
        confidence = 0.4; coverage = 0.5; diagnostics = {}; violations = ()

    class _Critic:
        def evaluate(self, cand, *, prompt, physics_plan):
            return _Rep()

    class _Sel:
        def select(self, evals): return evals[0]

    class _Reg:
        def supports(self, a): return True
        def execute(self, req):
            from physgenloop.learning_repair.contracts import ExecutionResult
            from physgenloop.learning_repair.base_contracts import RepairAction
            new = _Gen().generate(prompt=req.prompt, physics_plan=None, seed=req.seed)
            return ExecutionResult(action=RepairAction.PROMPT_REPAIR, status="succeeded", backend_id="m", candidate=new, next_prompt=req.prompt)

    def _side(cand):
        calls.append(cand.candidate_id)
        return {"quality_score": 0.88, "semantic_score": None}

    from physgenloop.learning_repair.contracts import RepairDecision
    from physgenloop.learning_repair.base_contracts import RepairAction
    def _dec(rep, ev):
        probs = {a.value: 0.25 for a in RepairAction}
        return RepairDecision(action=RepairAction.PROMPT_REPAIR, confidence=0.5, instruction="", action_probabilities=probs, per_action_values=probs)

    runner = ActionAwareRunnerV2(
        generator=_Gen(), critic=_Critic(), decider=_dec,
        executor_registry=_Reg(), selector=_Sel(),
        config=RunnerConfig(max_rounds=2, default_total_frames=4),
        side_score_fn=_side,
    )
    result = runner.run(sample_id="s1", prompt="p", physics_plan=None)
    # side_score_fn was called for each evaluated candidate
    assert len(calls) > 0


# ── VlmSemanticScorer unavailable returns None ───────────────────────────────
def test_semantic_scorer_unavailable():
    from generators.wanphysics.v2.scorers_semantic import VlmSemanticScorer
    sc = VlmSemanticScorer(base_url="http://localhost:19999/v1")
    assert sc.available is False
    assert sc.score(prompt="p", video_path="/tmp/x.mp4") is None


# ── build_v2_runner includes new artifacts and scorers ───────────────────────
def test_build_v2_runner_wires_scorers(tmp_path):
    pytest.importorskip("torch")
    with open(_ROOT / "configs/loop_v2.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ckpt = Path(
        cfg.get("paths", {})
        .get("checkpoints", {})
        .get("repair_agent", _ROOT / "checkpoints" / "repair_agent" / "repair-agent-v3.1-proxy-20260717")
    )
    if not ckpt.exists():
        pytest.skip("real V2 backend assembly requires the proxy repair checkpoint")
    from generators.wanphysics.v2.build_backends import build_v2_runner
    runner, critic, artifacts, preflight = build_v2_runner(
        cfg=cfg, run_dir=str(tmp_path), sample_dir=str(tmp_path / "s1"),
        sample_id="s1", allow_proxy_policy=True,
    )
    # quality scorer wired (scorers.quality_enabled=true by default)
    assert critic._quality_scorer is not None
    # memory_status written
    assert (tmp_path / "memory_status.json").exists()
    # side_score_fn set on runner
    assert runner.side_score_fn is not None
