"""WanPhysics V2 强制四动作 Trial 采集入口。

对同一 broken candidate 分别执行 prompt_repair / global_regeneration /
local_editing / reject，产出 WanRepairTrialV2；未执行/不可用动作记 null，不伪造失败。
默认 dry-run（mock，CPU 可测）；真实 GPU 采集需另行授权。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "src"))

from physgenloop.learning_repair.contracts import CandidateRecord, RepairDecision, ScoreBundle
from physgenloop.learning_repair.base_contracts import RepairAction

from generators.wanphysics.v2.trials import WanRepairTrialV2
from generators.wanphysics.v2.artifacts import append_jsonl

ACTIONS = ("prompt_repair", "global_regeneration", "local_editing", "reject")


def _decision(action: str) -> RepairDecision:
    from physgenloop.learning_repair.contracts import LocalEditTarget
    lt = None
    if action == "local_editing":
        lt = LocalEditTarget(parent_candidate_id="src", objects=("ball",), start_frame=1, end_frame=3, critical_frames=(1, 2, 3), mask_uri="mask_manifest.json")
    return RepairDecision(
        action=RepairAction(action),
        confidence=0.5,
        instruction="mock",
        action_probabilities={a.value: 0.25 for a in RepairAction},
        per_action_values={a.value: 0.0 for a in RepairAction},
        local_target=lt,
        source="forced_trial",
    )


def _mock_trial(action: str, available: bool) -> WanRepairTrialV2:
    src = CandidateRecord(candidate_id="src-1", video_path="/tmp/src.mp4", prompt="p", seed=1)
    if not available:
        return WanRepairTrialV2(
            trial_id=f"t-{action}", group_id="g1", source_candidate=src, prompt="p",
            critic_before={"decision": "violation"}, decision=_decision(action),
            execution={"status": "unavailable"}, before_scores=ScoreBundle(physics=0.2),
            successful=False, failure_reason="action_unavailable",
        )
    return WanRepairTrialV2(
        trial_id=f"t-{action}", group_id="g1", source_candidate=src, prompt="p",
        critic_before={"decision": "violation"}, decision=_decision(action),
        execution={"status": "succeeded"}, before_scores=ScoreBundle(physics=0.2),
        critic_after={"decision": "physical"}, after_scores=ScoreBundle(physics=0.85),
        successful=True,
    )


def _unavailable_trial(
    *,
    action: str,
    group_id: str,
    prompt: str,
    reason: str,
    preflight,
    seed: int,
) -> WanRepairTrialV2:
    src = CandidateRecord(
        candidate_id=f"{group_id}-source-unavailable",
        video_path="unavailable://not-generated",
        prompt=prompt,
        seed=seed,
    )
    return WanRepairTrialV2(
        trial_id=f"{group_id}-unavailable",
        group_id=group_id,
        source_candidate=src,
        prompt=prompt,
        critic_before={
            "status": "not_run",
            "reason": reason,
            "preflight": preflight.to_dict() if hasattr(preflight, "to_dict") else None,
        },
        decision=_decision(action),
        execution={
            "action": action,
            "status": "unavailable",
            "backend_id": "v2-preflight-capability-mask",
            "reason": reason,
        },
        before_scores=ScoreBundle(physics=0.0),
        successful=False,
        failure_reason=reason,
        metadata={"forced_action": action, "unavailable": True},
    )


def _dry_run(args) -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_root) / f"v2_trials_{ts}_dryrun"
    out.mkdir(parents=True, exist_ok=True)
    trials_path = out / "trials.jsonl"
    for action in ACTIONS:
        available = True  # dry-run 是 schema/mock 验证；真实可用性由 _real_trials preflight 决定。
        append_jsonl(trials_path, _mock_trial(action, available).to_dict())
    print(json.dumps({"mode": "dry_run", "trials": str(trials_path), "actions": list(ACTIONS)}, ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WanPhysics V2 forced four-action trials")
    p.add_argument("--config", default=str(_ROOT / "configs" / "loop_v2.yaml"))
    p.add_argument("--manifest", default=str(_ROOT / "evaluation" / "manifests" / "videophy2_smoke20.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--enable", action="store_true")
    p.add_argument("--allow-proxy-policy", action="store_true")
    p.add_argument("--output-root", default=str(_ROOT / "outputs"))
    return p


def _real_trials(args) -> int:
    """P0-04：对同一 broken candidate 分别强制四动作，产 WanRepairTrialV2。

    同源保证：四个动作用同一 cfg、同一 base_seed、同一 prompt，Wan 子进程按 seed
    确定性生成，故四条 trial 的 before candidate（candidate_id 由 prompt+seed 派生）
    相同，只有动作与执行产物不同。
    """
    import yaml
    from generators.wanphysics.v2.build_backends import build_v2_runner

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("loop", {})["max_rounds"] = 2
    cfg["loop"]["base_seed"] = int(cfg.get("loop", {}).get("base_seed", 42))  # 固定同源 seed

    with open(args.manifest, "r", encoding="utf-8") as f:
        samples = json.load(f).get("samples", [])[:1]
    if not samples:
        print("[V2] no samples"); return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_root) / f"v2_trials_{ts}"
    sample = samples[0]
    prompt = str(sample.get("prompt") or sample.get("caption") or "")

    for action in ACTIONS:
        sid = f"forced_{action}"
        runner, critic, artifacts, preflight = build_v2_runner(
            cfg=cfg, run_dir=str(out), sample_dir=str(out / sid),
            sample_id=sid, allow_proxy_policy=args.allow_proxy_policy,
            force_action=action,
        )
        if action == "local_editing" and not runner.capability_fn().get("local_editing", False):
            # ProPainter 不可用：记录 unavailable，不伪造执行（P0-03/P0-04）。
            reason = "local_editing capability masked by config/preflight"
            artifacts.append_trial(
                sid,
                _unavailable_trial(
                    action=action,
                    group_id=sid,
                    prompt=prompt,
                    reason=reason,
                    preflight=preflight,
                    seed=int(cfg.get("loop", {}).get("base_seed", 42)),
                ).to_dict(),
            )
            print(f"[V2] {action}: capability masked (config/preflight)")
            continue
        try:
            result = runner.run(sample_id=sid, prompt=prompt, physics_plan=None)
            from agents.wanphysics.run_videophy2_loop_v2 import _assemble_trials
            for t in _assemble_trials(sid, result, prompt):
                artifacts.append_trial(sid, t)
            print(f"[V2] {action}: {result.stop_reason}")
        except Exception as exc:
            print(f"[V2] {action}: ERROR {exc}")
        finally:
            critic.shutdown()

    print(json.dumps({"out": str(out), "actions": list(ACTIONS)}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    if not args.enable:
        print("[V2] trials not enabled; pass --enable (GPU authorization required) or --dry-run. Exiting.")
        return 0
    return _real_trials(args)


if __name__ == "__main__":
    raise SystemExit(main())
