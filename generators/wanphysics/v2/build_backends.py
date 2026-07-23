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
from physgenloop.learning_repair.baselines import HeuristicDecisionPolicy
from physgenloop.learning_repair.executors import PromptRepairExecutor
from physgenloop.repairer import InstructionPromptRepairer
from physgenloop.selector import EvidenceAwareSelector

from ..adapter import WanSubprocessGenerator
from ..sam2_vlm_critic import Sam2VlmSubprocessCritic
from .artifacts import RunArtifacts, SampleStatus
from .critic_backend import V2SubprocessCritic
from .executors import (
    AuditedRejectExecutor,
    MaskSequenceLocalEditingExecutor,
)
from .guardrails import GateThresholds, CpuQualityScorer
from .mask_manifest import build_local_edit_target, build_manifest, has_valid_masks, verify_manifest
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


class _CoordinatedGenerator:
    def __init__(self, backend: Any, prepare: Any) -> None:
        self._backend = backend
        self._prepare = prepare

    def generate(self, *, prompt: str, seed: int):
        self._prepare()
        return self._backend.generate(prompt=prompt, seed=seed)


class _CoordinatedEditor:
    def __init__(self, backend: Any, prepare: Any) -> None:
        self._backend = backend
        self._prepare = prepare

    def edit(self, **kwargs):
        self._prepare()
        return self._backend.edit(**kwargs)


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
        original_prompt: str,
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
        self._original_prompt = str(original_prompt)
        # shadow 阶段记录的旁路分数，供 runner gate 读取。
        self._scores: dict[str, dict[str, float | None]] = {}

    def prepare_for_generation(self) -> None:
        # 双卡：Wan 在 GPU0，不动 GPU1 上常驻的 vLLM。
        self._lifecycle.prepare_for_generation()

    def _score_side_channels(self, candidate: Any, prompt: str) -> dict[str, float | None]:
        """CPU quality + VLM semantic（shadow 记录，不改判）。"""
        quality = semantic = original_semantic = None
        if self._quality_scorer is not None:
            try:
                q = self._quality_scorer.score(candidate.video_path)
                quality = q.score if hasattr(q, "score") else q
            except Exception:  # noqa: BLE001
                quality = None
        if self._semantic_scorer is not None and getattr(self._semantic_scorer, "available", True):
            try:
                semantic = self._semantic_scorer.score(prompt=prompt, video_path=candidate.video_path)
                original_semantic = self._semantic_scorer.score(
                    prompt=self._original_prompt,
                    video_path=candidate.video_path,
                )
            except Exception:  # noqa: BLE001
                semantic = None
        return {
            "quality_score": quality,
            "semantic_score": semantic,
            "original_prompt_semantic_score": original_semantic,
        }

    def scores_for(self, candidate: Any) -> dict[str, float | None]:
        return self._scores.get(
            str(getattr(candidate, "candidate_id", "")),
            {
                "quality_score": None,
                "semantic_score": None,
                "original_prompt_semantic_score": None,
            },
        )

    def evaluate(self, candidate: Any, *, prompt: str):
        import time as _time

        cid = str(candidate.candidate_id)
        gpu_before = self._lifecycle._gpu_memory_used_mb() if hasattr(self._lifecycle, "_gpu_memory_used_mb") else None
        t0 = _time.perf_counter()
        try:
            decoded = self._backend.evaluate(candidate, prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            self._artifacts.write_raw_payload(
                self._sample_id,
                cid,
                {},
                f"{type(exc).__name__}: {exc}",
            )
            self._mask_valid[cid] = False
            return None
        critic_seconds = _time.perf_counter() - t0
        gpu_after = self._lifecycle._gpu_memory_used_mb() if hasattr(self._lifecycle, "_gpu_memory_used_mb") else None
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
            candidate_id=cid,
            video_path=candidate.video_path,
            prompt=prompt,
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
        if state in {
            "ACCEPTED",
            "REJECTED",
            "MAX_ROUNDS",
            "EVALUATION_FAILED",
            "EXECUTION_FAILED",
            "PREFLIGHT_FAILED",
        }:
            # The entrypoint commits terminal status only after loop_result and
            # Trial artifacts are durable and schema-valid.
            return
        try:
            self._artifacts.set_status(SampleStatus(sample_id=sample_id, state=state, round_index=round_index))
        except ValueError:
            pass
        # critic 失败也写一条 trace（§12：不静默）。
        if state in {"EVALUATION_FAILED", "EXECUTION_FAILED", "MAX_ROUNDS"}:
            self._artifacts.append_repair_trace(sample_id, {"round_index": round_index, "event": state})

    def on_decision(
        self,
        sample_id,
        candidate_id,
        decision,
        guard,
        round_index,
        execution_id,
    ) -> None:
        try:
            payload = {"decision": decision.to_dict() if hasattr(decision, "to_dict") else str(decision), "guard": guard.to_dict()}
            self._artifacts.write_decision(sample_id, candidate_id, payload)
            # 缓存供 on_execution 合并成一条完整 trace。
            self._pending[sample_id] = {
                "candidate_id": candidate_id,
                "round_index": round_index,
                "state": "EXECUTED",
                "execution_id": execution_id,
                "policy_action": guard.policy_action,
                "guard_status": guard.status,
                "executed_action": guard.final_action,
                "scope": guard.scope,
                "blocked_reason": guard.blocked_reason,
                "decision_source": "three_action_policy",
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
    original_prompt: str,
):
    """按 loop_v2.yaml 装配真实 V2 runner。返回 (runner, critic, artifacts, preflight)。"""

    paths = cfg.get("paths", {})
    loop = cfg.get("loop", {})
    runtime = cfg.get("runtime", {})
    vllm_cfg = cfg.get("vllm", {})
    accept = cfg.get("acceptance", {})
    local_cfg = cfg.get("local_editing", {})
    if str(accept.get("mode", "")) != "enforce":
        raise ValueError("V2 active runtime requires acceptance.mode=enforce")
    critic_cfg = cfg.get("critic", {}) or {}
    if not bool(critic_cfg.get("formal_profile_required", False)):
        raise ValueError("V2 active runtime requires critic.formal_profile_required=true")
    if str(critic_cfg.get("profile", "")) != "sam2_seeded_rules":
        raise ValueError("unsupported active critic profile")

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
        original_prompt=original_prompt,
    )
    coordinated_generator = _CoordinatedGenerator(
        generator,
        critic.prepare_for_generation,
    )

    max_rounds = loop.get("max_rounds", 2)
    # G4：vLLM PID owner（在首次 start_vllm 后写，用 lambda 延迟）。
    _orig_start = vllm_lifecycle.start_vllm

    def _start_and_record():
        _orig_start()
        try:
            pid = vllm_lifecycle.owned_pid
            artifacts.write_owner(
                {
                    "run_id": Path(run_dir).name,
                    "pid": pid,
                    "owned": pid is not None,
                    "port": vllm_cfg.get("port", 8000),
                }
            )
        except Exception:  # noqa: BLE001
            pass

    vllm_lifecycle.start_vllm = _start_and_record  # type: ignore[method-assign]

    # preflight → capability mask（ProPainter 缺失自动掩掉 local_editing）。
    preflight = run_preflight(
        propainter_repo=_resolve_path(local_cfg.get("propainter_repo"), project_root) or str(project_root / "models" / "ProPainter"),
        wan_model=_resolve_path(paths.get("models", {}).get("wan"), project_root)
        or str(project_root / "models" / "wan2.2_ti2v_5b"),
        vllm_model=_resolve_path(paths.get("models", {}).get("vllm"), project_root)
        or str(project_root / "models" / "Qwen3-VL-8B-Instruct"),
        sam2_ckpt=_resolve_path(paths.get("models", {}).get("sam2"), project_root)
        or str(project_root / "models" / "sam2.1_hiera_base_plus.pt"),
        env_file=str(project_root / ".env"),
        vllm_host=vllm_cfg.get("host", "127.0.0.1"),
        vllm_port=vllm_cfg.get("port", 8000),
        require_local_editing=bool(local_cfg.get("enabled", False)),
    )
    caps = dict(preflight.capability_mask)
    if not bool(local_cfg.get("enabled", False)):
        caps["local_editing"] = False
    caps = {
        "prompt_repair": bool(caps.get("prompt_repair", True)),
        "local_editing": bool(caps.get("local_editing", False)),
        "reject": True,
    }

    policy = HeuristicDecisionPolicy(
        minimum_coverage=float(cfg.get("policy", {}).get("minimum_coverage", 0.25)),
        compatibility_id="three-action-heuristic/1.0",
    )

    def _decider(report, evaluation, context):
        return policy.decide(
            critic_report=report,
            candidate=evaluation.candidate,
            prompt=getattr(evaluation.candidate, "prompt", original_prompt),
            context=context,
        )

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
    coordinated_editor = _CoordinatedEditor(editor, critic.prepare_for_generation)
    registry = ExecutorRegistry(
        executors=[
            PromptRepairExecutor(
                prompt_rewriter=InstructionPromptRepairer(),
                generator=coordinated_generator,
                backend_id="legacy-prompt-rewriter+video-generator",
            ),
            MaskSequenceLocalEditingExecutor(editor=coordinated_editor),
            AuditedRejectExecutor(selector=EvidenceAwareSelector()),
        ]
    )

    runner = ActionAwareRunnerV2(
        generator=coordinated_generator,
        critic=critic,
        decider=_decider,
        executor_registry=registry,
        selector=EvidenceAwareSelector(),
        config=RunnerConfig(
            max_rounds=max_rounds,
            candidates_per_round=loop.get("candidates_per_round", 1),
            base_seed=loop.get("base_seed", 42),
            acceptance_mode=accept.get("mode", "enforce"),
            error_scope_threshold=loop.get("error_scope_threshold", 0.4),
            default_total_frames=num_frames,
            thresholds=GateThresholds.from_mapping(accept),
            fail_on_degraded_critic=bool(runtime.get("fail_on_degraded_critic", True)),
        ),
        capability_fn=lambda: caps,
        mask_valid_fn=critic.mask_valid,
        local_target_fn=critic.local_target,
        side_score_fn=critic.scores_for,
        hooks=_ArtifactHooks(artifacts),
    )
    return runner, critic, artifacts, preflight
