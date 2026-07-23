"""The only active WanPhysics V2 entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import (
    CandidateRecord,
    ExecutionResult,
    RepairDecision,
    ScoreBundle,
)

from generators.wanphysics.v2.artifacts import (
    RunArtifacts,
    SampleStatus,
    pending_samples,
    rebuild_summary,
)
from generators.wanphysics.v2.guardrails import GateThresholds
from generators.wanphysics.v2.runner import ActionAwareRunnerV2, RunnerConfig, RunnerHooks


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    roots = (
        _ROOT / "agents" / "wanphysics",
        _ROOT / "generators" / "wanphysics",
        _ROOT / "src" / "physgenloop",
        _ROOT / "src" / "pavg_critic",
        _ROOT / "schemas",
    )
    files = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"}
    )
    for path in files:
        digest.update(path.relative_to(_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def _run_lock(run_root: Path):
    import fcntl

    path = run_root / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"RUN_ROOT is already locked: {run_root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WanPhysics strict V2 loop")
    parser.add_argument("--config", default=str(_ROOT / "configs" / "loop_v2.yaml"))
    parser.add_argument(
        "--manifest",
        default=str(_ROOT / "evaluation" / "manifests" / "videophy2_pilot300.json"),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Explicit final RUN_ROOT; omitted creates outputs/v2_run_<timestamp>.",
    )
    return parser


def _load_samples(manifest: str) -> list[dict[str, Any]]:
    with open(manifest, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("samples", []))


def _run_root(args: argparse.Namespace, *, dry_run: bool = False) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else ""
    return (_ROOT / "outputs" / f"v2_run_{timestamp}{suffix}").resolve()


def _manifest_payload(
    *,
    run_root: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    sample_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "v2-run-manifest/2.0",
        "run_id": run_root.name,
        "run_root": str(run_root),
        "created_at": _utc_now(),
        "entrypoint": "agents/wanphysics/run_videophy2_loop_v2.py",
        "decision_source": "three_action_policy",
        "project_root": str(_ROOT),
        "source_revision": _source_revision(),
        "source_fingerprint": _source_fingerprint(),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": _sha256(args.config),
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "sample_ids": sample_ids,
        "limit": args.limit,
        "max_rounds": int(cfg["loop"]["max_rounds"]),
        "acceptance_mode": "enforce",
        "critic_profile": str(cfg["critic"]["profile"]),
        "action_order": ["prompt_repair", "local_editing", "reject"],
    }


def _validate_resume(saved: dict[str, Any], current: dict[str, Any]) -> None:
    keys = (
        "source_revision",
        "source_fingerprint",
        "config_sha256",
        "manifest_sha256",
        "sample_ids",
        "limit",
        "max_rounds",
        "acceptance_mode",
        "critic_profile",
        "action_order",
    )
    mismatches = [key for key in keys if saved.get(key) != current.get(key)]
    if mismatches:
        raise RuntimeError(f"RESUME_COMPATIBILITY_FAILED: {mismatches}")


def _score_bundle(raw: dict[str, Any]) -> ScoreBundle:
    return ScoreBundle(
        physics=float(raw["physics"]),
        semantic=None if raw.get("semantic") is None else float(raw["semantic"]),
        original_prompt_semantic=(
            None
            if raw.get("original_prompt_semantic") is None
            else float(raw["original_prompt_semantic"])
        ),
        quality=None if raw.get("quality") is None else float(raw["quality"]),
    )


def _drop_ok(before: float | None, after: float | None, limit: float) -> bool:
    return before is not None and after is not None and after - before >= -float(limit)


def _assemble_trials(
    sample_id: str,
    result: Any,
    original_prompt: str,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate and move real RoundRecord facts; never reconstruct Policy/Critic."""

    from generators.wanphysics.v2.trials import WanRepairTrialV3

    acceptance = cfg.get("acceptance", {})
    semantic_limit = float(acceptance.get("max_semantic_drop", 0.03))
    original_limit = float(
        acceptance.get("max_original_prompt_semantic_drop", 0.03)
    )
    quality_limit = float(acceptance.get("max_quality_drop", 0.05))
    trials: list[dict[str, Any]] = []
    for record in result.rounds:
        if not record.execution_id or record.decision is None or record.execution is None:
            continue
        if record.before_candidate is None or record.critic_before is None or record.before_scores is None:
            raise RuntimeError(f"trial_evidence_incomplete:{record.execution_id}:before")
        decision = RepairDecision.from_dict(record.decision)
        before = _score_bundle(record.before_scores)
        after = None if record.after_scores is None else _score_bundle(record.after_scores)
        execution_status = str(record.execution.get("status", "unknown"))
        physics_gain = None if after is None else after.physics - before.physics
        side_ok = bool(
            after is not None
            and _drop_ok(before.semantic, after.semantic, semantic_limit)
            and _drop_ok(
                before.original_prompt_semantic,
                after.original_prompt_semantic,
                original_limit,
            )
            and _drop_ok(before.quality, after.quality, quality_limit)
        )
        repair_improved = bool(
            execution_status == "succeeded"
            and physics_gain is not None
            and physics_gain > 0.0
            and side_ok
        )
        after_gate_status = str((record.after_gate or {}).get("status", "")).upper()
        successful = repair_improved and after_gate_status == "ACCEPTED"
        failure_reason = None
        if not successful:
            failure_reason = (
                record.execution.get("failure_reason")
                or record.terminal_reason
                or ("after_gate_unavailable" if after_gate_status == "UNAVAILABLE" else None)
                or ("after_gate_rejected" if after_gate_status == "REJECTED" else None)
                or ("physics_not_improved" if physics_gain is not None and physics_gain <= 0 else None)
                or "repair_not_strictly_accepted"
            )
        trial = WanRepairTrialV3(
            trial_id=record.execution_id,
            group_id=sample_id,
            source_candidate=CandidateRecord.from_dict(record.before_candidate),
            original_prompt=original_prompt,
            prompt=str(record.before_prompt or record.before_candidate.get("prompt", "")),
            critic_before=dict(record.critic_before),
            decision=decision,
            guard=dict(record.guard or {}),
            execution={**record.execution, "execution_id": record.execution_id},
            before_scores=before,
            critic_after=None if record.critic_after is None else dict(record.critic_after),
            after_scores=after,
            repair_improved=repair_improved,
            successful=successful,
            failure_reason=failure_reason,
            metadata={
                "gates": {"before": record.gate, "after": record.after_gate},
                "candidate_paths": {
                    "before": record.before_candidate.get("video_path"),
                    "after": None
                    if record.after_candidate is None
                    else record.after_candidate.get("video_path"),
                },
            },
        )
        trials.append(trial.to_dict())
    return trials


class _MockViolation:
    object = "red_ball"
    category = "gravity_violation"
    critical_frames = (1, 2)
    repair_instruction = "Keep downward acceleration physically consistent."


class _MockReport:
    def __init__(self, physical: bool) -> None:
        self.decision = "physical" if physical else "violation"
        self.physics_score = 0.9 if physical else 0.2
        self.confidence = 0.8
        self.coverage = 0.8
        self.diagnostics: dict[str, Any] = {}
        self.evidence_bundles = ()
        self.violations = () if physical else (_MockViolation(),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "physics_score": self.physics_score,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "diagnostics": self.diagnostics,
            "violations": [
                {
                    "object": item.object,
                    "category": item.category,
                    "critical_frames": list(item.critical_frames),
                    "repair_instruction": item.repair_instruction,
                }
                for item in self.violations
            ],
        }


class _MockGenerator:
    def generate(self, *, prompt: str, seed: int) -> GeneratedCandidate:
        candidate_id = f"mock-{seed}"
        return GeneratedCandidate(
            candidate_id,
            f"/tmp/{candidate_id}.mp4",
            prompt,
            seed,
            {"num_frames": 8},
        )


class _MockCritic:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, candidate: Any, *, prompt: str) -> _MockReport:
        self.calls += 1
        return _MockReport(self.calls >= 2)


class _MockSelector:
    def select(self, evaluations):
        return max(evaluations, key=lambda item: item.report.physics_score)


class _MockRegistry:
    def execute(self, request):
        candidate = _MockGenerator().generate(prompt=request.prompt + " corrected", seed=request.seed)
        return ExecutionResult(
            action=request.decision.action,
            status="succeeded",
            backend_id="mock-prompt-repair",
            candidate=candidate,
            next_prompt=candidate.prompt,
        )


def _mock_decision(report, evaluation, context) -> RepairDecision:
    probabilities = {
        RepairAction.PROMPT_REPAIR: 0.8,
        RepairAction.LOCAL_EDITING: 0.1,
        RepairAction.REJECT: 0.1,
    }
    return RepairDecision(
        action=RepairAction.PROMPT_REPAIR,
        confidence=0.8,
        instruction="Keep downward acceleration physically consistent.",
        action_probabilities=probabilities,
        per_action_values={action: probabilities[action] for action in RepairAction},
        source="three_action_dry_run",
        compatibility_id="three-action-dry-run/1.0",
    )


def _dry_run(args: argparse.Namespace) -> int:
    run_root = _run_root(args, dry_run=True)
    artifacts = RunArtifacts(run_root)
    artifacts.create_run_manifest(
        {
            "schema_version": "v2-run-manifest/2.0",
            "run_id": run_root.name,
            "run_root": str(run_root),
            "created_at": _utc_now(),
            "entrypoint": "agents/wanphysics/run_videophy2_loop_v2.py",
            "mode": "dry_run",
            "decision_source": "three_action_policy",
            "action_order": ["prompt_repair", "local_editing", "reject"],
            "acceptance_mode": "enforce",
        }
    )
    attempt_id, _ = artifacts.start_attempt("dryrun-0001", reason="dry_run")
    runner = ActionAwareRunnerV2(
        generator=_MockGenerator(),
        critic=_MockCritic(),
        decider=_mock_decision,
        executor_registry=_MockRegistry(),
        selector=_MockSelector(),
        config=RunnerConfig(max_rounds=args.max_rounds or 2),
        capability_fn=lambda: {
            "prompt_repair": True,
            "local_editing": False,
            "reject": True,
        },
        side_score_fn=lambda candidate: {
            "semantic_score": 0.9,
            "original_prompt_semantic_score": 0.9,
            "quality_score": 0.9,
        },
    )
    result = runner.run(sample_id="dryrun-0001", prompt="a red ball falls")
    artifacts.write_loop_result("dryrun-0001", result.to_dict())
    dry_cfg = {
        "acceptance": {
            "max_semantic_drop": 0.03,
            "max_original_prompt_semantic_drop": 0.03,
            "max_quality_drop": 0.05,
        }
    }
    for trial in _assemble_trials(
        "dryrun-0001",
        result,
        "a red ball falls",
        dry_cfg,
    ):
        artifacts.append_trial("dryrun-0001", trial)
    artifacts.set_status(
        SampleStatus(
            "dryrun-0001",
            result.final_state,
            len(result.rounds),
            {"attempt_id": attempt_id},
        )
    )
    artifacts.write_summary(rebuild_summary(run_root, ["dryrun-0001"]))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _real_run(args: argparse.Namespace) -> int:
    import yaml

    from generators.wanphysics.v2.build_backends import build_v2_runner

    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.max_rounds is not None:
        cfg.setdefault("loop", {})["max_rounds"] = args.max_rounds
    if str(cfg.get("acceptance", {}).get("mode", "")) != "enforce":
        raise ValueError("active runtime requires acceptance.mode=enforce")

    samples = _load_samples(args.manifest)[: args.limit]
    sample_ids = [
        str(sample.get("sample_id") or f"sample-{index:04d}")
        for index, sample in enumerate(samples)
    ]
    run_root = _run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(run_root)
    current_manifest = _manifest_payload(
        run_root=run_root,
        args=args,
        cfg=cfg,
        sample_ids=sample_ids,
    )

    with _run_lock(run_root):
        manifest_path = run_root / "run_manifest.json"
        previous_resume_count = 0
        status_path = run_root / "run_status.json"
        if status_path.exists():
            try:
                previous_resume_count = int(
                    json.loads(status_path.read_text(encoding="utf-8")).get(
                        "resume_count", 0
                    )
                )
            except Exception:  # noqa: BLE001
                previous_resume_count = 0
        resume_count = previous_resume_count + (1 if args.resume else 0)
        if args.resume:
            if not manifest_path.exists():
                raise RuntimeError("--resume requires an initialized run_manifest.json")
            _validate_resume(artifacts.read_run_manifest(), current_manifest)
        else:
            artifacts.create_run_manifest(current_manifest)
        artifacts.write_run_status(
            {
                "schema_version": "v2-run-status/1.0",
                "run_id": run_root.name,
                "state": "RESUMING" if args.resume else "RUNNING",
                "pid": os.getpid(),
                "resume_count": resume_count,
                "updated_at": _utc_now(),
            }
        )

        pending = set(
            pending_samples(
                run_root,
                sample_ids,
                retry_failed=args.retry_failed,
            )
        )
        for index, sample in enumerate(samples):
            sample_id = sample_ids[index]
            if sample_id not in pending:
                print(f"[V2] skip terminal sample {sample_id}")
                continue
            prompt = str(sample.get("prompt") or sample.get("caption") or "").strip()
            attempt_id, attempt_dir = artifacts.start_attempt(
                sample_id,
                reason=("retry_failed" if args.retry_failed else "resume" if args.resume else "initial"),
            )
            critic = None
            try:
                runner, critic, sample_artifacts, preflight = build_v2_runner(
                    cfg=cfg,
                    run_dir=str(run_root),
                    sample_dir=str(attempt_dir),
                    sample_id=sample_id,
                    original_prompt=prompt,
                )
                if not preflight.all_ok:
                    sample_artifacts.write_loop_result(
                        sample_id,
                        {
                            "sample_id": sample_id,
                            "final_state": "PREFLIGHT_FAILED",
                            "preflight": preflight.to_dict(),
                        },
                    )
                    sample_artifacts.set_status(
                        SampleStatus(
                            sample_id,
                            "PREFLIGHT_FAILED",
                            0,
                            {"attempt_id": attempt_id, "missing": list(preflight.missing)},
                        )
                    )
                    continue
                result = runner.run(sample_id=sample_id, prompt=prompt)
                sample_artifacts.write_loop_result(sample_id, result.to_dict())
                trials = _assemble_trials(sample_id, result, prompt, cfg)
                for trial in trials:
                    sample_artifacts.append_trial(sample_id, trial)
                sample_artifacts.set_status(
                    SampleStatus(
                        sample_id,
                        result.final_state,
                        len(result.rounds),
                        {
                            "attempt_id": attempt_id,
                            "stop_reason": result.stop_reason,
                            "trials_written": len(trials),
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                artifacts.write_loop_result(
                    sample_id,
                    {
                        "sample_id": sample_id,
                        "final_state": "EXECUTION_FAILED",
                        "stop_reason": "runtime_or_artifact_failure",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                artifacts.set_status(
                    SampleStatus(
                        sample_id,
                        "EXECUTION_FAILED",
                        0,
                        {
                            "attempt_id": attempt_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
            finally:
                if critic is not None:
                    critic.shutdown()

        summary = rebuild_summary(run_root, sample_ids)
        artifacts.write_summary(summary)
        failed = sum(
            int(summary.get(name, 0))
            for name in ("evaluation_failed", "execution_failed", "preflight_failed")
        )
        rejected = int(summary.get("rejected", 0)) + int(summary.get("max_rounds", 0))
        state = "FAILED" if failed else "COMPLETED_WITH_REJECTIONS" if rejected else "COMPLETED"
        artifacts.write_run_status(
            {
                "schema_version": "v2-run-status/1.0",
                "run_id": run_root.name,
                "state": state,
                "pid": os.getpid(),
                "resume_count": resume_count,
                "updated_at": _utc_now(),
                "summary": summary,
            }
        )
    print(json.dumps({"run_root": str(run_root), "summary": summary}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    if not args.enable:
        print("[V2] pass --enable for real runtime or --dry-run for CPU validation")
        return 0
    return _real_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
