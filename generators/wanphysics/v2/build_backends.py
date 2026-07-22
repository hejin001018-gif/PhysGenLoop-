"""V2 真实后端装配（只新增，不改旧入口）。

把 :class:`ActionAwareRunnerV2` 接到真实后端，并落实双卡角色分工 + 低分辨率：

  generator : WanSubprocessGenerator(gpu_id=wan_gpu, height/width 来自配置)  → Wan 独占 GPU0
  critic    : V2 无损 codec 评分 + Sam2VlmSubprocessCritic 管理 vLLM(独占 GPU1，常驻)
  decider   : ActionValueRepairer.repair_with_decision（Policy 只决策一次）
  executors : V2 decision-only 四动作
  capability: preflight 决定（ProPainter 缺失则 local_editing 掩掉）
  mask      : 从候选目录 sam2_masks + violations 构建 mask_manifest 校验

真实 GPU 运行仍受入口 ``--enable`` 与 C 级授权约束；本模块只负责"可装配"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from physgenloop.learning_repair import ExecutorRegistry
from physgenloop.selector import EvidenceAwareSelector

from ..adapter import WanSubprocessGenerator
from ..repairer import load_action_value_repairer
from ..sam2_vlm_critic import Sam2VlmSubprocessCritic
from .artifacts import RunArtifacts, SampleStatus
from .critic_backend import V2SubprocessCritic
from .executors import (
    AuditedRejectExecutor,
    DecisionPromptRepairExecutor,
    MaskSequenceLocalEditingExecutor,
    OriginalPromptGlobalRegenerationExecutor,
)
from .guardrails import GateThresholds, CpuQualityScorer
from .mask_manifest import build_local_edit_target, build_manifest, has_valid_masks, verify_manifest
from .memory_adapter import inspect_memory
from .preflight import run_preflight
from .runner import ActionAwareRunnerV2, RunnerConfig, RunnerHooks
from .scorers_semantic import VlmSemanticScorer

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value: str | None, root: Path) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(root / path)


class _V2Critic:
    """无损评分 + vLLM 生命周期（双卡：GPU1 常驻）+ 完整报告/mask 落盘。

    返回给 runner 的是 CriticReport（解析失败返回 None → runner 记 critic_failed）。
    """

    def __init__(
        self,
        *,
        python: str,
        eval_step: str,
        vllm_lifecycle: Sam2VlmSubprocessCritic,
        requested_profile: str,
        artifacts: RunArtifacts,
        sample_id: str,
        dual_gpu: bool,
        quality_scorer: Any = None,
        semantic_scorer: Any = None,
    ) -> None:
        self._lifecycle = vllm_lifecycle
        self._dual_gpu = dual_gpu
        self._backend = V2SubprocessCritic(
            python=python,
            eval_step_path=eval_step,
            requested_profile=requested_profile,
            prepare_hook=vllm_lifecycle.start_vllm,  # 幂等：已健康则跳过
        )
        self._artifacts = artifacts
        self._sample_id = sample_id
        self._mask_valid: dict[str, bool] = {}
        self._mask_manifest: dict[str, Any] = {}
        self._mask_manifest_path: dict[str, Path] = {}
        self._quality_scorer = quality_scorer
        self._semantic_scorer = semantic_scorer
        # shadow 阶段记录的旁路分数，供 runner gate 读取。
        self._scores: dict[str, dict[str, float | None]] = {}

    def prepare_for_generation(self) -> None:
        # 双卡：Wan 在 GPU0，不动 GPU1 上常驻的 vLLM。
        self._lifecycle.prepare_for_generation()

    def _score_side_channels(self, candidate: Any, prompt: str) -> dict[str, float | None]:
        """CPU quality + VLM semantic（shadow 记录，不改判）。"""
        quality = semantic = None
        if self._quality_scorer is not None:
            try:
                q = self._quality_scorer.score(candidate.video_path)
                quality = q.score if hasattr(q, "score") else q
            except Exception:  # noqa: BLE001
                quality = None
        if self._semantic_scorer is not None and getattr(self._semantic_scorer, "available", True):
            try:
                semantic = self._semantic_scorer.score(prompt=prompt, video_path=candidate.video_path)
            except Exception:  # noqa: BLE001
                semantic = None
        return {"quality_score": quality, "semantic_score": semantic}

    def scores_for(self, candidate: Any) -> dict[str, float | None]:
        return self._scores.get(str(getattr(candidate, "candidate_id", "")), {"quality_score": None, "semantic_score": None})

    def evaluate(self, candidate: Any, *, prompt: str, physics_plan: Any):
        import time as _time

        cid = str(candidate.candidate_id)
        gpu_before = self._lifecycle._gpu_memory_used_mb() if hasattr(self._lifecycle, "_gpu_memory_used_mb") else None
        t0 = _time.perf_counter()
        decoded = self._backend.evaluate(candidate, prompt=prompt, physics_plan=physics_plan)
        critic_seconds = _time.perf_counter() - t0
        gpu_after = self._lifecycle._gpu_memory_used_mb() if hasattr(self._lifecycle, "_gpu_memory_used_mb") else None
        try:
            plan_dict = physics_plan.to_dict() if hasattr(physics_plan, "to_dict") else {}
        except Exception:  # noqa: BLE001
            plan_dict = {}
        # 资源指标（§19）：查不到记 null。
        self._artifacts.append_resource_metrics(
            self._sample_id,
            {
                "candidate_id": cid,
                "gpu_memory_before_mb": gpu_before,
                "gpu_memory_after_mb": gpu_after,
                "critic_seconds": round(critic_seconds, 2),
            },
        )
        doc = decoded.critic_report_document(
            candidate_id=cid, video_path=candidate.video_path, prompt=prompt, physics_plan=plan_dict
        )
        # 旁路 scorer（shadow）：写进 critic_report 供审计。
        side = self._score_side_channels(candidate, prompt)
        self._scores[cid] = side
        doc["side_scores"] = side
        self._artifacts.write_critic_report(self._sample_id, cid, doc)
        if not decoded.ok:
            self._artifacts.write_raw_payload(self._sample_id, cid, decoded.decode.raw_payload, decoded.decode.error or "roundtrip_failed")
            self._mask_valid[cid] = False
            return None
        report = decoded.report
        # 构建 mask manifest（对齐 sam2_masks/{object}_{frame:05d}.png）。
        mask_dir = Path(candidate.video_path).resolve().parent / "sam2_masks"
        meta = getattr(candidate, "metadata", {}) or {}
        manifest = build_manifest(
            candidate_id=cid,
            video_path=candidate.video_path,
            mask_dir=mask_dir,
            violations=getattr(report, "violations", ()) or (),
            video_width=meta.get("width"),
            video_height=meta.get("height"),
            video_frames=meta.get("num_frames"),
        )
        manifest_path = self._artifacts.write_mask_manifest(self._sample_id, cid, manifest.to_dict())
        self._mask_manifest[cid] = manifest
        self._mask_manifest_path[cid] = manifest_path
        manifest_ok, _manifest_problems = verify_manifest(manifest, check_sha=True)
        self._mask_valid[cid] = bool(
            manifest_ok
            and has_valid_masks(manifest)
            and self.local_target(report, candidate) is not None
        )
        return report

    def mask_valid(self, report: Any, candidate: Any) -> bool:
        return bool(self._mask_valid.get(str(getattr(candidate, "candidate_id", "")), False))

    def local_target(self, report: Any, candidate: Any):
        cid = str(getattr(candidate, "candidate_id", ""))
        manifest = self._mask_manifest.get(cid)
        manifest_path = self._mask_manifest_path.get(cid)
        if manifest is None or manifest_path is None:
            return None
        for violation in tuple(getattr(report, "violations", ()) or ()):
            target = build_local_edit_target(
                parent_candidate_id=cid,
                violation=violation,
                manifest=manifest,
                manifest_uri=str(manifest_path),
            )
            if target is not None:
                return target
        return None

    def shutdown(self) -> None:
        self._lifecycle.shutdown()


class _ArtifactHooks(RunnerHooks):
    def __init__(self, artifacts: RunArtifacts) -> None:
        self._artifacts = artifacts
        self._pending: dict[str, dict] = {}

    def on_state(self, sample_id: str, state: str, round_index: int) -> None:
        try:
            self._artifacts.set_status(SampleStatus(sample_id=sample_id, state=state, round_index=round_index))
        except ValueError:
            pass
        # critic 失败也写一条 trace（§12：不静默）。
        if state in {"CRITIC_FAILED", "MAX_ROUNDS"}:
            self._artifacts.append_repair_trace(sample_id, {"round_index": round_index, "event": state})

    def on_decision(self, sample_id, candidate_id, decision, guard) -> None:
        try:
            payload = {"decision": decision.to_dict() if hasattr(decision, "to_dict") else str(decision), "guard": guard.to_dict()}
            self._artifacts.write_decision(sample_id, candidate_id, payload)
            # 缓存供 on_execution 合并成一条完整 trace。
            self._pending[sample_id] = {
                "candidate_id": candidate_id,
                "policy_action": guard.policy_action,
                "final_action": guard.final_action,
                "scope": guard.scope,
                "override_reason": guard.override_reason,
            }
        except Exception:  # noqa: BLE001
            pass

    def on_execution(self, sample_id: str, exec_result: Any) -> None:
        # G1：每个动作执行后写 repair_trace.jsonl（append-only，动作级审计）。
        try:
            base = self._pending.pop(sample_id, {})
            record = {
                **base,
                "execution_status": getattr(exec_result, "status", "unknown"),
                "backend_id": getattr(exec_result, "backend_id", None),
                "failure_reason": getattr(exec_result, "failure_reason", None),
                "terminal": getattr(exec_result, "terminal", False),
                "cost": getattr(exec_result, "cost", None),
                "latency_seconds": getattr(exec_result, "latency_seconds", None),
            }
            self._artifacts.append_repair_trace(sample_id, record)
        except Exception:  # noqa: BLE001
            pass


def build_v2_runner(
    *,
    cfg: dict,
    run_dir: str,
    sample_dir: str,
    sample_id: str,
    allow_proxy_policy: bool,
    force_action: str | None = None,
):
    """按 loop_v2.yaml 装配真实 V2 runner。返回 (runner, critic, artifacts, preflight)。"""

    paths = cfg.get("paths", {})
    loop = cfg.get("loop", {})
    runtime = cfg.get("runtime", {})
    vllm_cfg = cfg.get("vllm", {})
    accept = cfg.get("acceptance", {})
    local_cfg = cfg.get("local_editing", {})

    gpu_mode = str(runtime.get("gpu_mode", "dual_gpu"))
    dual_gpu = gpu_mode == "dual_gpu"
    wan_gpu = runtime.get("generator_gpu", 0) if dual_gpu else None
    vllm_gpu = runtime.get("critic_gpu", 1) if dual_gpu else None

    gen_cfg = cfg.get("generator", {}) or {}
    height = gen_cfg.get("height", 480)
    width = gen_cfg.get("width", 832)
    num_frames = gen_cfg.get("num_frames", 81)

    project_root = Path(paths.get("root") or _PROJECT_ROOT).resolve()

    python = _resolve_path(paths.get("envs", {}).get("main"), project_root)
    vllm_python = _resolve_path(paths.get("envs", {}).get("vllm"), project_root)
    eval_step = _resolve_path(paths.get("eval_step"), project_root)

    artifacts = RunArtifacts(run_dir)

    generator = WanSubprocessGenerator(
        python=python,
        model_path=_resolve_path(paths.get("models", {}).get("wan"), project_root) or str(project_root / "models" / "wan2.2_ti2v_5b"),
        output_root=sample_dir,
        num_frames=num_frames,
        height=height,
        width=width,
        gpu_id=wan_gpu,
    )

    vllm_lifecycle = Sam2VlmSubprocessCritic(
        python=python,
        vllm_python=vllm_python,
        vllm_model=_resolve_path(paths.get("models", {}).get("vllm"), project_root) or str(project_root / "models" / "Qwen3-VL-8B-Instruct"),
        vllm_served_name=vllm_cfg.get("served_name", "qwen3-vl-8b-instruct"),
        vllm_log=str(Path(run_dir) / "vllm.log"),
        vllm_gpu_util=vllm_cfg.get("gpu_memory_utilization", 0.85),
        vllm_max_model_len=vllm_cfg.get("max_model_len", 16384),
        vllm_gpu_id=vllm_gpu,
        dual_gpu=dual_gpu,
    )
    scorers_cfg = cfg.get("scorers", {}) or {}
    quality_scorer = CpuQualityScorer() if scorers_cfg.get("quality_enabled", True) else None
    semantic_scorer = None
    if scorers_cfg.get("semantic_enabled", False):
        env = {}
        try:
            from dotenv import dotenv_values

            env = dotenv_values(str(project_root / ".env"))
        except Exception:  # noqa: BLE001
            env = {}
        semantic_scorer = VlmSemanticScorer(
            base_url=env.get("BASE_URL", "http://localhost:8000/v1"),
            api_key=env.get("API_KEY", "local"),
            model=env.get("VLM_MODEL", "qwen3-vl-8b-instruct"),
        )

    critic = _V2Critic(
        python=python,
        eval_step=eval_step,
        vllm_lifecycle=vllm_lifecycle,
        requested_profile=cfg.get("critic", {}).get("profile", "sam2_seeded_rules"),
        artifacts=artifacts,
        sample_id=sample_id,
        dual_gpu=dual_gpu,
        quality_scorer=quality_scorer,
        semantic_scorer=semantic_scorer,
    )

    max_rounds = loop.get("max_rounds", 2)
    ckpt_root = paths.get("checkpoints", {}).get("repair_agent") if isinstance(paths.get("checkpoints"), dict) else None
    ckpt_root = _resolve_path(ckpt_root, project_root)
    ckpt_root = ckpt_root or str(project_root / "checkpoints" / "repair_agent" / "repair-agent-v3.1-proxy-20260717")

    # G3：memory 格式识别，整轮写一次 memory_status（默认 disabled）。
    mem_cfg = cfg.get("memory", {}) or {}
    try:
        mem_status = inspect_memory(
            Path(ckpt_root) / "memory/proxy_memory_train.jsonl",
            enable=bool(mem_cfg.get("enable_proxy", False)),
        )
        artifacts.write_memory_status(mem_status.to_dict())
    except Exception:  # noqa: BLE001
        pass

    # G4：vLLM PID owner（在首次 start_vllm 后写，用 lambda 延迟）。
    _orig_start = vllm_lifecycle.start_vllm

    def _start_and_record():
        _orig_start()
        try:
            import subprocess as _sp

            pid_out = _sp.run(
                ["pgrep", "-f", "vllm.entrypoints.openai.api_server"], capture_output=True, text=True
            ).stdout.strip().splitlines()
            pid = int(pid_out[0]) if pid_out else -1
            artifacts.write_owner({"run_id": Path(run_dir).name, "pid": pid, "port": vllm_cfg.get("port", 8000)})
        except Exception:  # noqa: BLE001
            pass

    vllm_lifecycle.start_vllm = _start_and_record  # type: ignore[method-assign]

    # P0-08：checkpoint 分层硬门禁——加载前判定，写 checkpoint_gate.json。
    from .checkpoint_gate import evaluate_checkpoint_gate, MODE_PROXY_RESEARCH
    policy_cfg = cfg.get("policy", {}) or {}
    ckpt_mode = policy_cfg.get("mode", MODE_PROXY_RESEARCH)
    gate_result = evaluate_checkpoint_gate(
        ckpt_root, mode=ckpt_mode, allow_proxy_override=bool(allow_proxy_policy),
    )
    artifacts.write_json(str(Path(run_dir) / "checkpoint_gate.json"), gate_result.to_dict()) if hasattr(artifacts, "write_json") else None
    try:
        from .artifacts import write_json as _wj
        _wj(Path(run_dir) / "checkpoint_gate.json", gate_result.to_dict())
    except Exception:  # noqa: BLE001
        pass
    if not gate_result.allow_load:
        raise RuntimeError(
            f"checkpoint gate blocked load: mode={gate_result.mode} reasons={gate_result.reasons}"
        )

    repairer = load_action_value_repairer(ckpt_root, max_attempts=max_rounds, local_editor_available=True)

    def _decider(report, evaluation):
        _prompt, decision = repairer.repair_with_decision(prompt=getattr(evaluation.candidate, "prompt", ""), report=report)
        return decision

    # preflight → capability mask（ProPainter 缺失自动掩掉 local_editing）。
    preflight = run_preflight(
        propainter_repo=_resolve_path(local_cfg.get("propainter_repo"), project_root) or str(project_root / "models" / "ProPainter"),
        vllm_host=vllm_cfg.get("host", "127.0.0.1"),
        vllm_port=vllm_cfg.get("port", 18000),
        require_local_editing=bool(local_cfg.get("enabled", False)),
    )
    caps = dict(preflight.capability_mask)
    if not bool(local_cfg.get("enabled", False)):
        caps["local_editing"] = False

    if bool(local_cfg.get("strict_manifest", True)):
        from .propainter_strict_editor import StrictProPainterLocalEditor

        editor_cls = StrictProPainterLocalEditor
    else:
        from ..local_editor import ProPainterLocalEditor

        editor_cls = ProPainterLocalEditor
    editor = editor_cls(
        propainter_repo=_resolve_path(local_cfg.get("propainter_repo"), project_root) or str(project_root / "models" / "ProPainter"),
        python=python,
        output_root=sample_dir,
    )
    registry = ExecutorRegistry(
        executors=[
            DecisionPromptRepairExecutor(generator=generator),
            OriginalPromptGlobalRegenerationExecutor(generator=generator),
            MaskSequenceLocalEditingExecutor(editor=editor),
            AuditedRejectExecutor(selector=EvidenceAwareSelector()),
        ]
    )

    # P0-04：force_action 时用 forced decider（记录 proposed vs forced）。
    active_decider = _decider
    if force_action:
        from physgenloop.learning_repair.base_contracts import RepairAction
        from physgenloop.learning_repair.contracts import RepairDecision, LocalEditTarget

        def _forced_decider(report, evaluation, _fa=force_action):
            proposed = None
            try:
                proposed = getattr(_decider(report, evaluation), "action", None)
            except Exception:  # noqa: BLE001
                proposed = None
            lt = None
            if _fa == "local_editing":
                lt = critic.local_target(report, evaluation.candidate)
            probs = {a.value: (1.0 if a.value == _fa else 0.0) for a in RepairAction}
            tot = sum(probs.values()) or 1.0
            probs = {k: v / tot for k, v in probs.items()}
            return RepairDecision(
                action=RepairAction(_fa), confidence=0.99, instruction=f"forced:{_fa}",
                action_probabilities=probs,
                per_action_values={a.value: 0.0 for a in RepairAction},
                local_target=lt, source="force_action_override",
                parameters={"proposed_action": (proposed.value if proposed else None), "forced_action": _fa},
            )
        active_decider = _forced_decider

    runner = ActionAwareRunnerV2(
        generator=generator,
        critic=critic,
        decider=active_decider,
        executor_registry=registry,
        selector=EvidenceAwareSelector(),
        config=RunnerConfig(
            max_rounds=max_rounds,
            candidates_per_round=loop.get("candidates_per_round", 1),
            base_seed=loop.get("base_seed", 42),
            acceptance_mode=accept.get("mode", "shadow"),
            error_scope_threshold=loop.get("error_scope_threshold", 0.4),
            default_total_frames=num_frames,
            thresholds=GateThresholds.from_mapping(accept),
            fail_on_degraded_critic=bool(runtime.get("fail_on_degraded_critic", False)),
            force_action=force_action,
            require_plan=bool(cfg.get("acceptance", {}).get("require_plan", False)),
        ),
        capability_fn=lambda: caps,
        mask_valid_fn=critic.mask_valid,
        local_target_fn=critic.local_target,
        side_score_fn=critic.scores_for,
        hooks=_ArtifactHooks(artifacts),
    )
    return runner, critic, artifacts, preflight
