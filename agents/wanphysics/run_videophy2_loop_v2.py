"""WanPhysics V2 videophy2 入口。

默认不 enable 直接退出。``--dry-run`` 用 mock 组件跑完整 V2 状态机（CPU 可测），
``--enable`` + GPU 授权才装配真实后端。旧 run_videophy2_loop.py 不受影响。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = "/root/PhysGenLoop-"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, _ROOT + "/src")

from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import RepairDecision

from generators.wanphysics.v2.artifacts import RunArtifacts, SampleStatus, pending_samples
from generators.wanphysics.v2.guardrails import GateThresholds
from generators.wanphysics.v2.runner import ActionAwareRunnerV2, RunnerConfig, RunnerHooks


def _uniform_decision(action: str) -> RepairDecision:
    from physgenloop.learning_repair.contracts import LocalEditTarget

    probs = {a.value: 0.25 for a in RepairAction}
    vals = {a.value: 0.0 for a in RepairAction}
    local_target = None
    if action == "local_editing":
        local_target = LocalEditTarget(
            parent_candidate_id="mock",
            objects=("ball",),
            start_frame=1,
            end_frame=3,
            critical_frames=(1, 2, 3),
            mask_uri="mock_mask.png",
        )
    return RepairDecision(
        action=RepairAction(action),
        confidence=0.5,
        instruction="mock",
        action_probabilities=probs,
        per_action_values=vals,
        local_target=local_target,
        source="dry_run",
    )


class _MockGenerator:
    def generate(self, *, prompt, physics_plan, seed):
        cid = f"mock-{seed}"
        return GeneratedCandidate(candidate_id=cid, video_path=f"/tmp/{cid}.mp4", prompt=prompt, seed=seed, metadata={"num_frames": 8})


class _MockReport:
    def __init__(self, physical: bool):
        self.decision = "physical" if physical else "violation"
        self.is_physical = physical
        self.physics_score = 0.9 if physical else 0.2
        self.confidence = 0.7
        self.coverage = 0.8
        self.diagnostics = {}
        from physgenloop.learning_repair.base_contracts import RepairAction as _RA  # noqa
        self.violations = ()


class _MockCritic:
    def __init__(self, accept_after: int):
        self.accept_after = accept_after
        self.n = 0

    def evaluate(self, candidate, *, prompt, physics_plan):
        self.n += 1
        return _MockReport(self.n >= self.accept_after)


class _MockSelector:
    def select(self, evals):
        return max(evals, key=lambda e: getattr(e.report, "physics_score", 0.0))


class _MockRegistry:
    def __init__(self, force_action: str):
        self.force_action = force_action

    def supports(self, action):
        return True

    def execute(self, request):
        from physgenloop.learning_repair.contracts import ExecutionResult
        action = request.decision.action
        if action == RepairAction.REJECT:
            cand = request.history[-1].candidate if request.history else request.candidate
            return ExecutionResult(action=action, status="rejected", backend_id="mock", candidate=cand, terminal=True)
        new = GeneratedCandidate(candidate_id=f"edited-{request.seed}", video_path=f"/tmp/edited-{request.seed}.mp4", prompt=request.prompt, seed=request.seed, metadata={"num_frames": 8})
        return ExecutionResult(action=action, status="succeeded", backend_id="mock", candidate=new, next_prompt=request.prompt)


class _ArtifactHooks(RunnerHooks):
    def __init__(self, artifacts: RunArtifacts):
        self.artifacts = artifacts

    def on_state(self, sample_id, state, round_index):
        try:
            self.artifacts.set_status(SampleStatus(sample_id=sample_id, state=state, round_index=round_index))
        except ValueError:
            pass


def _dry_run(args) -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"v2_run_{ts}_dryrun"
    artifacts = RunArtifacts(run_dir)
    force = args.force_action or "prompt_repair"
    runner = ActionAwareRunnerV2(
        generator=_MockGenerator(),
        critic=_MockCritic(accept_after=99 if force != "accept" else 1),
        decider=lambda report, ev: _uniform_decision(force),
        executor_registry=_MockRegistry(force),
        selector=_MockSelector(),
        config=RunnerConfig(max_rounds=(args.max_rounds or 2), thresholds=GateThresholds(), default_total_frames=8),
        capability_fn=lambda: {"prompt_repair": True, "global_regeneration": True, "local_editing": True, "reject": True},
        mask_valid_fn=lambda report, cand: force == "local_editing",
        hooks=_ArtifactHooks(artifacts),
    )
    result = runner.run(sample_id="dryrun-0001", prompt="a red ball rolls on a table", physics_plan=None)
    artifacts.write_loop_result("dryrun-0001", result.to_dict())
    artifacts.write_summary({"mode": "dry_run", "force_action": force, "stop_reason": result.stop_reason})
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WanPhysics V2 videophy2 loop")
    p.add_argument("--config", default=f"{_ROOT}/configs/loop_v2.yaml")
    p.add_argument("--manifest", default=f"{_ROOT}/evaluation/manifests/videophy2_pilot300.json")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--enable", action="store_true", help="启用 V2（GPU 运行需另行授权）")
    p.add_argument("--dry-run", action="store_true", help="mock 组件跑状态机，CPU 可用")
    p.add_argument("--force-action", choices=["prompt_repair", "global_regeneration", "local_editing", "reject", "accept"], default=None)
    p.add_argument("--allow-proxy-policy", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--trace-level", default="full")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--output-root", default=f"{_ROOT}/outputs")
    return p


def _load_samples(manifest: str) -> list[dict]:
    with open(manifest, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("samples", []))


def _assemble_trials(sample_id: str, result: Any, prompt: str) -> list[dict]:
    """P0-03：execution-first 组装 WanRepairTrialV2。

    以每个有 execution_id 的 RoundRecord 为主键，before/after 直接取该 round 自身
    因果配对字段（runner 已保证 after 是本次 execution 的产物），不再窥探相邻 round。
    - reject / executor_failed：无 after，successful=False，记 terminal_reason；
    - success 复合定义：execution succeeded 且 after_physics > before_physics。
    """
    from generators.wanphysics.v2.trials import WanRepairTrialV2
    from physgenloop.learning_repair.contracts import CandidateRecord, ScoreBundle, RepairDecision
    from physgenloop.learning_repair.base_contracts import RepairAction

    trials = []
    for rnd in result.rounds:
        if not rnd.execution_id or rnd.final_action is None:
            continue
        try:
            status = (rnd.execution or {}).get("status", "unknown")
            action_str = rnd.final_action or "prompt_repair"
            before_physics = float(rnd.before_physics if rnd.before_physics is not None else 0.0)
            has_after = rnd.after_candidate_id is not None and rnd.after_physics is not None
            after_physics = float(rnd.after_physics) if has_after else None

            # 复合 success：真实执行成功 + after 物理提升。
            physics_improved = has_after and after_physics > before_physics
            successful = bool(status == "succeeded" and physics_improved)

            # 概率不伪造：无真实 policy 概率时置 0 分布 + 标注 source。
            probs = {a.value: (1.0 if a.value == action_str else 0.0) for a in RepairAction}
            tot = sum(probs.values()) or 1.0
            probs = {k: v / tot for k, v in probs.items()}
            decision = RepairDecision(
                action=RepairAction(action_str), confidence=0.5, instruction="",
                action_probabilities=probs,
                per_action_values={a.value: 0.0 for a in RepairAction},
                source="runner_round_record",
            )
            src = CandidateRecord(
                candidate_id=rnd.before_candidate_id or rnd.candidate_id,
                video_path=f"{rnd.before_candidate_id or rnd.candidate_id}-v01.mp4",
                prompt=prompt, seed=rnd.round_index,
            )
            failure_reason = None
            if not successful:
                failure_reason = (
                    (rnd.execution or {}).get("failure_reason")
                    or rnd.terminal_reason
                    or ("no_physics_gain" if has_after else "no_after_candidate")
                )
            trial = WanRepairTrialV2(
                trial_id=rnd.execution_id,
                group_id=sample_id,
                source_candidate=src,
                prompt=prompt,
                critic_before=rnd.gate or {},
                decision=decision,
                execution={**(rnd.execution or {}), "execution_id": rnd.execution_id,
                           "before_candidate_id": rnd.before_candidate_id,
                           "after_candidate_id": rnd.after_candidate_id},
                before_scores=ScoreBundle(physics=before_physics),
                critic_after=rnd.after_gate,
                after_scores=ScoreBundle(physics=after_physics) if has_after else None,
                successful=successful,
                failure_reason=failure_reason,
            )
            trials.append(trial.to_dict())
        except Exception:  # noqa: BLE001
            pass
    return trials


def _real_run(args) -> int:
    import yaml
    from generators.wanphysics.v2.build_backends import build_v2_runner
    from generators.wanphysics.v2.artifacts import RunArtifacts, append_jsonl

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("runtime", {}).get("enabled", False) and not args.enable:
        print("[V2] runtime.enabled=false 且未传 --enable，退出。")
        return 0

    policy_cfg = cfg.get("policy", {})
    if policy_cfg.get("require_explicit_proxy_override", True) and not args.allow_proxy_policy:
        print("[V2] proxy checkpoint 需 --allow-proxy-policy 才能加载（research 模式）。退出。")
        return 0

    if args.max_rounds is not None:
        cfg.setdefault("loop", {})["max_rounds"] = args.max_rounds

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"v2_run_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(args.manifest)[: args.limit]
    sample_ids = [str(s.get("sample_id") or f"sample-{i:04d}") for i, s in enumerate(samples)]

    done = set()
    if args.resume:
        from generators.wanphysics.v2.artifacts import pending_samples as _pending
        pending = set(_pending(run_root, sample_ids))
        done = set(sample_ids) - pending

    summaries = []
    for i, sample in enumerate(samples):
        sid = sample_ids[i]
        if sid in done:
            print(f"[V2] skip completed {sid}")
            continue
        prompt = str(sample.get("prompt") or sample.get("caption") or "").strip()
        sample_dir = str(run_root / sid)
        # P0-04：force_action 通过 build_v2_runner 装配，记录 proposed vs forced。
        runner, critic, artifacts, preflight = build_v2_runner(
            cfg=cfg, run_dir=str(run_root), sample_dir=sample_dir, sample_id=sid,
            allow_proxy_policy=args.allow_proxy_policy,
            force_action=args.force_action,
        )

        artifacts.write_run_manifest({
            "run_id": run_root.name, "sample_id": sid,
            "resolution": [cfg.get("generator", {}).get("width", 832), cfg.get("generator", {}).get("height", 480)],
            "gpu_mode": cfg.get("runtime", {}).get("gpu_mode"),
            "preflight": preflight.to_dict(),
            "force_action": args.force_action,
        })
        try:
            result = runner.run(sample_id=sid, prompt=prompt, physics_plan=None)
            artifacts.write_loop_result(sid, result.to_dict())
            # G2：写 trials.jsonl。
            trials = _assemble_trials(sid, result, prompt)
            for t in trials:
                artifacts.append_trial(sid, t)
            summaries.append({
                "sample_id": sid,
                "stop_reason": result.stop_reason,
                "best_physics_score": result.best_physics_score,
                "trials_written": len(trials),
            })
        finally:
            critic.shutdown()

    root_artifacts = RunArtifacts(run_root)
    root_artifacts.write_summary({
        "samples": len(summaries), "results": summaries,
        "resolution": [cfg.get("generator", {}).get("width", 832), cfg.get("generator", {}).get("height", 480)],
    })
    print(json.dumps({"run_dir": str(run_root), "samples": len(summaries)}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    if not args.enable:
        print("[V2] runtime not enabled; pass --enable (GPU authorization required) or --dry-run. Exiting.")
        return 0
    return _real_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
