from __future__ import annotations

import sys

import paramiko

HOST = "px-cloud1.matpool.com"
PORT = 27323
USER = "root"
PASSWORD = r"dXm#hEFBUa@@f*N}"
ROOT = "/root/PhysGenLoop-"


def read_file(sftp: paramiko.SFTPClient, path: str) -> str:
    with sftp.open(path, "r") as handle:
        return handle.read().decode("utf-8")


def write_file(sftp: paramiko.SFTPClient, path: str, content: str) -> None:
    with sftp.open(path, "w") as handle:
        handle.write(content.encode("utf-8"))


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    sftp = client.open_sftp()
    try:
        contracts = """\"\"\"PhysGenLoop 跨组件共享的不可变数据契约。\"\"\"

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pavg_critic.schemas import CriticReport, PhysicsPlan


@dataclass(frozen=True)
class LoopConfig:
    \"\"\"控制 Best-of-K 和最大反馈轮数的最小配置。\"\"\"

    max_rounds: int = 3
    candidates_per_round: int = 2
    acceptance_score: float = 0.8
    base_seed: int = 42
    error_scope_threshold: float = 0.4
    default_total_frames: int | None = None

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if self.candidates_per_round < 1:
            raise ValueError("candidates_per_round must be at least 1")
        if not 0.0 <= self.acceptance_score <= 1.0:
            raise ValueError("acceptance_score must be within [0, 1]")
        if not 0.0 <= self.error_scope_threshold <= 1.0:
            raise ValueError("error_scope_threshold must be within [0, 1]")
        if self.default_total_frames is not None and self.default_total_frames < 1:
            raise ValueError("default_total_frames must be positive when provided")


@dataclass(frozen=True)
class GeneratedCandidate:
    \"\"\"生成器返回的候选视频引用；不要求视频已加载到内存。\"\"\"

    candidate_id: str
    video_path: str
    prompt: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if not self.video_path.strip():
            raise ValueError("video_path must not be empty")


@dataclass(frozen=True)
class CandidateEvaluation:
    \"\"\"一个候选与其结构化 Critic 报告。\"\"\"

    candidate: GeneratedCandidate
    report: CriticReport


@dataclass(frozen=True)
class LoopRound:
    \"\"\"一次生成—评价—选择的完整审计记录。\"\"\"

    round_index: int
    prompt: str
    evaluations: tuple[CandidateEvaluation, ...]
    selected_candidate_id: str


@dataclass(frozen=True)
class LoopResult:
    \"\"\"有界循环的最终结果与全部历史。\"\"\"

    best: CandidateEvaluation
    history: tuple[LoopRound, ...]
    stop_reason: str
    resolved_plan: PhysicsPlan
"""
        write_file(sftp, f"{ROOT}/src/physgenloop/contracts.py", contracts)

        error_scope = """\"\"\"基于 critical_frames 占比的错误范围启发式。\"\"\"

from __future__ import annotations

from typing import Any, Mapping


def _violations(report: Any) -> tuple[Any, ...]:
    return tuple(getattr(report, "violations", ()) or ())


def extract_critical_frames(report: Any) -> tuple[int, ...]:
    frames: set[int] = set()
    for violation in _violations(report):
        for frame in getattr(violation, "critical_frames", ()) or ():
            try:
                value = int(frame)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                frames.add(value)
    return tuple(sorted(frames))


def has_local_editing_evidence(report: Any) -> bool:
    for violation in _violations(report):
        evidence = getattr(violation, "evidence", {}) or {}
        if getattr(violation, "critical_frames", ()) or (
            isinstance(evidence, Mapping) and evidence.get("mask_uri")
        ):
            return True
    return False


def classify_error_scope(report: Any, total_frames: int, local_threshold: float = 0.4) -> str:
    if total_frames <= 0:
        return "global"
    critical_frames = extract_critical_frames(report)
    ratio = len(critical_frames) / float(total_frames)
    if ratio == 0.0 or ratio >= float(local_threshold):
        return "global"
    return "local"
"""
        write_file(sftp, f"{ROOT}/generators/wanphysics/error_scope.py", error_scope)

        repairer = """\"\"\"ActionValueDecisionPolicy 的 PromptRepairer 协议适配器。\"\"\"
from __future__ import annotations

from pathlib import Path

from physgenloop.learning_repair import (
    ActionValueDecisionPolicy,
    CompatibilityManifest,
    RepairAction,
    RepairContext,
    RepairMemory,
    TorchActionValuePolicy,
)
from physgenloop.contracts import GeneratedCandidate
from pavg_critic.schemas import CriticReport

_ACTION_PREFIX = {
    RepairAction.PROMPT_REPAIR: "Physics correction",
    RepairAction.GLOBAL_REGENERATION: "Regeneration constraint",
    RepairAction.LOCAL_EDITING: "Local-edit fallback constraint",
    RepairAction.REJECT: "Replacement constraint",
}


class ActionValueRepairer:
    def __init__(
        self,
        decision_policy: ActionValueDecisionPolicy,
        max_attempts: int = 2,
        proxy_memory: RepairMemory | None = None,
        memory_weight: float = 0.25,
        local_editor_available: bool = False,
    ) -> None:
        self._policy = decision_policy
        self._max_attempts = max_attempts
        self._attempt_index = 0
        self._previous_actions: list[RepairAction] = []
        self._proxy_memory = proxy_memory
        self._memory_weight = memory_weight
        self._local_editor_available = local_editor_available

    def repair_with_decision(self, *, prompt: str, report: CriticReport):
        from physgenloop.learning_repair.baselines import _target
        from physgenloop.learning_repair.contracts import RepairDecision

        context = RepairContext(
            attempt_index=self._attempt_index,
            max_attempts=self._max_attempts,
            local_editor_available=self._local_editor_available,
            previous_actions=tuple(self._previous_actions),
        )
        placeholder = GeneratedCandidate(
            candidate_id="repair-placeholder",
            video_path="pending://",
            prompt=prompt,
            seed=self._attempt_index,
        )
        decision = self._policy.decide(
            critic_report=report,
            candidate=placeholder,
            prompt=prompt,
            context=context,
        )

        if self._proxy_memory is not None:
            matches = self._proxy_memory.retrieve(report, context=context)
            if matches:
                mem_dist = self._proxy_memory.action_distribution(matches)
                w = self._memory_weight
                blended_values = {
                    action: (1.0 - w) * decision.per_action_values.get(action, 0.0)
                    + w * mem_dist.get(action, 0.0)
                    for action in RepairAction
                }
                best_action = max(blended_values, key=lambda a: blended_values[a])
                if best_action != decision.action:
                    decision = RepairDecision(
                        action=best_action,
                        confidence=decision.confidence,
                        instruction=decision.instruction,
                        action_probabilities=decision.action_probabilities,
                        per_action_values=blended_values,
                        parameters=decision.parameters,
                        local_target=decision.local_target,
                        source=f"{decision.source}+proxy-memory",
                        compatibility_id=decision.compatibility_id,
                    )

        if decision.local_target is None:
            candidate_target = _target(report, placeholder)
            if candidate_target.mask_uri or candidate_target.critical_frames:
                decision = RepairDecision(
                    action=decision.action,
                    confidence=decision.confidence,
                    instruction=decision.instruction,
                    action_probabilities=decision.action_probabilities,
                    per_action_values=decision.per_action_values,
                    parameters=decision.parameters,
                    local_target=candidate_target,
                    source=decision.source,
                    abstained=decision.abstained,
                    fallback_reason=decision.fallback_reason,
                    compatibility_id=decision.compatibility_id,
                )

        self._previous_actions.append(decision.action)
        self._attempt_index += 1

        instruction = decision.instruction.strip()
        if not instruction:
            return prompt, decision
        prefix = _ACTION_PREFIX[decision.action]
        return f"{prompt}\\n{prefix}: {instruction}", decision

    def repair(self, *, prompt: str, report: CriticReport) -> str:
        repaired, _ = self.repair_with_decision(prompt=prompt, report=report)
        return repaired


def load_action_value_repairer(
    ckpt_root: str,
    max_attempts: int = 2,
    local_editor_available: bool = False,
) -> ActionValueRepairer:
    ckpt_path = Path(ckpt_root)
    compatibility = CompatibilityManifest.load(
        str(ckpt_path / "config/critic_compatibility_v1.json")
    )

    if not compatibility.deployment_ready:
        print(
            "[repairer] WARNING: checkpoint compatibility manifest has deployment_ready=False "
            "(source_revision='unknown'). This is a proxy-trained checkpoint; "
            "actual_trial_count=0. Proceeding as proxy mode.",
        )

    _ROOT = Path("/root/PhysGenLoop-")
    _LOCAL_COMPAT = _ROOT / "configs/learning_repair/critic_compatibility_v1.json"
    if _LOCAL_COMPAT.exists():
        try:
            local_compat = CompatibilityManifest.load(str(_LOCAL_COMPAT))
            local_compat.verify_files(
                critic_config=_ROOT / "configs/default.yaml",
                critic_schema=_ROOT / "schemas/critic_output.schema.json",
                feature_schema=_ROOT / "configs/learning_repair/feature_schema.json",
            )
        except Exception as exc:
            print(
                f"[repairer] WARNING: Critic file hash mismatch ({exc}). "
                "Policy was trained on a different Critic revision.",
            )

    learned_policy = TorchActionValuePolicy.load(
        str(ckpt_path / "model/best_action_value_policy.pt"),
        device="cpu",
        compatibility_manifest=compatibility,
    )
    decision_policy = ActionValueDecisionPolicy(learned_policy, minimum_confidence=0.35)

    proxy_memory: RepairMemory | None = None
    memory_jsonl = ckpt_path / "memory/proxy_memory_train.jsonl"
    if memory_jsonl.exists():
        try:
            proxy_memory = RepairMemory.from_manifest(memory_jsonl)
            print(
                f"[repairer] loaded proxy memory: {len(proxy_memory)} examples from {memory_jsonl}",
            )
        except Exception as exc:
            print(f"[repairer] WARNING: failed to load proxy memory ({exc}), proceeding without memory")

    return ActionValueRepairer(
        decision_policy,
        max_attempts=max_attempts,
        proxy_memory=proxy_memory,
        memory_weight=0.25,
        local_editor_available=local_editor_available,
    )
"""
        write_file(sftp, f"{ROOT}/generators/wanphysics/repairer.py", repairer)

        executor_factory = read_file(sftp, f"{ROOT}/generators/wanphysics/executor_factory.py")
        executor_factory = executor_factory.replace(
            "    repairer = load_action_value_repairer(ckpt_root, max_attempts=max_attempts)\n",
            "    repairer = load_action_value_repairer(\n"
            "        ckpt_root,\n"
            "        max_attempts=max_attempts,\n"
            "        local_editor_available=True,\n"
            "    )\n",
        )
        write_file(sftp, f"{ROOT}/generators/wanphysics/executor_factory.py", executor_factory)

        sam2 = read_file(sftp, f"{ROOT}/src/pavg_critic/sam2_detector.py")
        sam2 = sam2.replace(
            '        jpeg_quality: int = 85,\n        prompt: str = "",\n    ) -> None:\n',
            '        jpeg_quality: int = 85,\n        prompt: str = "",\n        mask_output_dir: str | None = None,\n    ) -> None:\n',
        )
        sam2 = sam2.replace(
            '        self._object_names: dict[int, str] = {}\n        self._width: int | None = None\n        self._height: int | None = None\n        self._prompt = str(prompt or "").strip()\n',
            '        self._object_names: dict[int, str] = {}\n        self._width: int | None = None\n        self._height: int | None = None\n        self._prompt = str(prompt or "").strip()\n        self._mask_output_dir = None if mask_output_dir is None else Path(mask_output_dir)\n        self._mask_cache: dict[tuple[str, int], Any] = {}\n',
        )
        sam2 = sam2.replace(
            '                    detection = Detection(\n'
            '                        frame=out_frame_idx,\n'
            '                        timestamp_sec=timestamp,\n'
            '                        object=name,\n'
            '                        center=center,\n'
            '                        bbox=bbox,\n'
            '                        confidence=0.85,\n'
            '                        track_id=f"sam2:{obj_id_int}",\n'
            '                    )\n'
            '                    frame_dets.append(detection)\n',
            '                    detection = Detection(\n'
            '                        frame=out_frame_idx,\n'
            '                        timestamp_sec=timestamp,\n'
            '                        object=name,\n'
            '                        center=center,\n'
            '                        bbox=bbox,\n'
            '                        confidence=0.85,\n'
            '                        track_id=f"sam2:{obj_id_int}",\n'
            '                    )\n'
            '                    if self._mask_output_dir is not None:\n'
            '                        self._mask_cache[(name, out_frame_idx)] = (mask.astype("uint8") * 255)\n'
            '                    frame_dets.append(detection)\n',
        )
        sam2 = sam2.replace(
            '    @staticmethod\n    def _extract_video_frames(\n',
            '    def materialize_masks(self, violations: Sequence[Any]) -> dict[tuple[str, int], str]:\n'
            '        """仅为最终违规帧落盘 mask，供局部修复消费。"""\n'
            '        if self._mask_output_dir is None:\n'
            '            return {}\n'
            '        self._mask_output_dir.mkdir(parents=True, exist_ok=True)\n'
            '        try:\n'
            '            import cv2\n'
            '        except ImportError as exc:\n'
            '            raise RuntimeError("OpenCV is required to write SAM2 masks") from exc\n'
            '\n'
            '        paths: dict[tuple[str, int], str] = {}\n'
            '        for violation in violations:\n'
            '            object_name = str(getattr(violation, "object", "")).strip()\n'
            '            for frame in getattr(violation, "critical_frames", ()) or ():\n'
            '                key = (object_name, int(frame))\n'
            '                mask = self._mask_cache.get(key)\n'
            '                if mask is None:\n'
            '                    continue\n'
            '                filename = f"{object_name or \'object\'}_{int(frame):05d}.png"\n'
            '                path = self._mask_output_dir / filename\n'
            '                if not path.exists():\n'
            '                    ok = cv2.imwrite(str(path), mask)\n'
            '                    if not ok:\n'
            '                        raise ValueError(f"cannot encode SAM2 mask for {filename}")\n'
            '                paths[key] = str(path)\n'
            '        return paths\n'
            '\n'
            '    @staticmethod\n    def _extract_video_frames(\n',
        )
        write_file(sftp, f"{ROOT}/src/pavg_critic/sam2_detector.py", sam2)

        eval_step = """\"\"\"一次性评估步骤：SAM2+VLM 评分，写出 CriticReport JSON 后退出。\"\"\"
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/PhysGenLoop-")
sys.path.insert(0, "/root/PhysGenLoop-/src")

from dotenv import load_dotenv

load_dotenv("/root/PhysGenLoop-/.env")

from pavg_critic import OpenAIChatModel, PhysicsCritic, SAM2ObjectDetector, CriticRequest
from pavg_critic.schemas import CriticReport, PhysicsPlan, Violation
from physgenloop.contracts import GeneratedCandidate


def _build_critic(vlm, video_path: str, prompt: str, physics_plan: PhysicsPlan):
    mask_output_dir = str(Path(video_path).resolve().parent / "sam2_masks")
    try:
        detector = SAM2ObjectDetector(
            vlm,
            video_path,
            model_ckpt=os.environ.get(
                "SAM2_CHECKPOINT", "/root/PhysGenLoop-/models/sam2.1_hiera_base_plus.pt"
            ),
            prompt=prompt,
            mask_output_dir=mask_output_dir,
        )
        return PhysicsCritic(detector=detector), "sam2+vlm", detector
    except Exception as exc:
        print(
            f"  ⚠ SAM2+VLM 不可用（{type(exc).__name__}: {exc}），降级为默认规则 Critic",
            file=sys.stderr,
        )
        return PhysicsCritic(), "rules_fallback", None


def _attach_mask_uris(report: CriticReport, detector: SAM2ObjectDetector | None) -> CriticReport:
    if detector is None or not getattr(report, "violations", None):
        return report
    paths = detector.materialize_masks(report.violations)
    if not paths:
        return report

    violations: list[Violation] = []
    changed = False
    for violation in report.violations:
        evidence = dict(violation.evidence or {})
        object_name = str(violation.object).strip()
        mask_paths = [
            paths[(object_name, int(frame))]
            for frame in violation.critical_frames
            if (object_name, int(frame)) in paths
        ]
        if mask_paths:
            evidence["mask_uri"] = mask_paths[0]
            evidence["mask_uris"] = mask_paths
            changed = True
        violations.append(
            Violation(
                object=violation.object,
                category=violation.category,
                start_frame=violation.start_frame,
                peak_frame=violation.peak_frame,
                end_frame=violation.end_frame,
                critical_frames=violation.critical_frames,
                reason=violation.reason,
                repair_instruction=violation.repair_instruction,
                evidence=evidence,
            )
        )
    if not changed:
        return report
    return CriticReport(
        is_physical=report.is_physical,
        physics_score=report.physics_score,
        confidence=report.confidence,
        violations=tuple(violations),
        graph_evaluation=report.graph_evaluation,
        node_results=report.node_results,
        decision=report.decision,
        coverage=report.coverage,
        score_breakdown=dict(report.score_breakdown),
        diagnostics=dict(report.diagnostics),
        model_versions=dict(report.model_versions),
        evidence_bundles=report.evidence_bundles,
        schema_version=report.schema_version,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    with open(args.candidate_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    candidate = GeneratedCandidate(
        candidate_id=raw["candidate_id"],
        video_path=raw["video_path"],
        prompt=raw["prompt"],
        seed=raw["seed"],
        metadata=raw.get("metadata", {}),
    )

    raw_plan = raw.get("physics_plan", {})
    try:
        physics_plan = PhysicsPlan.from_dict(raw_plan) if hasattr(PhysicsPlan, "from_dict") and raw_plan else PhysicsPlan()
    except Exception:
        physics_plan = PhysicsPlan()

    vlm = OpenAIChatModel(
        api_key=os.environ["API_KEY"],
        model=os.environ["VLM_MODEL"],
        base_url=os.environ["BASE_URL"],
        strict_json_schema=os.environ.get("VLM_STRICT_SCHEMA", "").lower() == "true",
    )
    critic, backend, detector = _build_critic(vlm, candidate.video_path, candidate.prompt, physics_plan)
    report = critic.analyze(
        CriticRequest(
            video_path=candidate.video_path,
            prompt=candidate.prompt,
            physics_plan=physics_plan,
        )
    )
    report = _attach_mask_uris(report, detector)

    payload = {
        "candidate_id": candidate.candidate_id,
        "video_path": candidate.video_path,
        "detector_backend": backend,
        "report": report.to_dict(),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        "EVAL_OK", candidate.candidate_id,
        "decision=", report.decision,
        "score=", report.physics_score,
        "backend=", backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
        write_file(sftp, f"{ROOT}/agents/wanphysics/eval_step.py", eval_step)

        controller = """\"\"\"PhysGenLoop 的有界 Best-of-K 反馈循环。\"\"\"

from __future__ import annotations

from pavg_critic.planner import PhysicsPlanResolver, TemplatePhysicsPlanner
from pavg_critic.schemas import CriticRequest, PhysicsPlan

from generators.wanphysics.error_scope import classify_error_scope, has_local_editing_evidence

from .contracts import CandidateEvaluation, LoopConfig, LoopResult, LoopRound
from .interfaces import CandidateCritic, CandidateSelector, PlanResolver, PromptRepairer, VideoGenerator


class LoopController:
    def __init__(
        self,
        *,
        generator: VideoGenerator,
        critic: CandidateCritic,
        repairer: PromptRepairer,
        selector: CandidateSelector,
        plan_resolver: PlanResolver | None = None,
        config: LoopConfig | None = None,
        executor_registry=None,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.repairer = repairer
        self.selector = selector
        self.plan_resolver = plan_resolver or PhysicsPlanResolver(TemplatePhysicsPlanner())
        self.config = config or LoopConfig()
        self.executor_registry = executor_registry

    def _total_frames(self, evaluation: CandidateEvaluation) -> int:
        metadata = getattr(evaluation.candidate, "metadata", {}) or {}
        raw = metadata.get("num_frames", self.config.default_total_frames)
        try:
            total = int(raw)
        except (TypeError, ValueError):
            return 0
        return total if total > 0 else 0

    def run(self, *, prompt: str, physics_plan: PhysicsPlan | None = None) -> LoopResult:
        explicit_plan = physics_plan if physics_plan is not None else PhysicsPlan()
        resolved_plan = self.plan_resolver.resolve(
            CriticRequest(
                video_path="pending://generation",
                prompt=prompt,
                physics_plan=explicit_plan,
            )
        ).plan
        current_prompt = prompt
        original_prompt = prompt
        history: list[LoopRound] = []
        round_winners: list[CandidateEvaluation] = []

        for round_index in range(self.config.max_rounds):
            evaluations: list[CandidateEvaluation] = []
            for offset in range(self.config.candidates_per_round):
                seed = self.config.base_seed + round_index * self.config.candidates_per_round + offset
                candidate = self.generator.generate(
                    prompt=current_prompt,
                    physics_plan=resolved_plan,
                    seed=seed,
                )
                report = self.critic.evaluate(
                    candidate,
                    prompt=current_prompt,
                    physics_plan=resolved_plan,
                )
                evaluations.append(CandidateEvaluation(candidate, report))

            frozen_evaluations = tuple(evaluations)
            selected = self.selector.select(frozen_evaluations)
            round_winners.append(selected)
            history.append(
                LoopRound(
                    round_index=round_index,
                    prompt=current_prompt,
                    evaluations=frozen_evaluations,
                    selected_candidate_id=selected.candidate.candidate_id,
                )
            )

            if selected.report.decision == "physical" and selected.report.physics_score >= self.config.acceptance_score:
                return LoopResult(
                    best=selected,
                    history=tuple(history),
                    stop_reason="accepted",
                    resolved_plan=resolved_plan,
                )

            if round_index == self.config.max_rounds - 1:
                break

            if hasattr(self.repairer, "repair_with_decision"):
                next_prompt, decision = self.repairer.repair_with_decision(
                    prompt=current_prompt,
                    report=selected.report,
                )
                policy_action = str(getattr(decision, "action", "")).lower()
                final_action = policy_action
                total_frames = self._total_frames(selected)
                scope = classify_error_scope(
                    selected.report,
                    total_frames,
                    self.config.error_scope_threshold,
                )
                local_evidence = has_local_editing_evidence(selected.report)
                registry_supports_local = bool(
                    self.executor_registry is not None and hasattr(self.executor_registry, "supports")
                ) and self.executor_registry.supports("local_editing")
                if scope == "local" and local_evidence and registry_supports_local:
                    final_action = "local_editing"
                elif scope == "global" and "local_editing" in policy_action:
                    final_action = "global_regeneration"
                selected.report.diagnostics["error_scope"] = {
                    "round_index": round_index,
                    "policy_action": policy_action,
                    "final_action": final_action,
                    "scope": scope,
                    "total_frames": total_frames,
                    "threshold": self.config.error_scope_threshold,
                    "has_local_evidence": local_evidence,
                }

                if "reject" in final_action:
                    return LoopResult(
                        best=self.selector.select(tuple(round_winners)),
                        history=tuple(history),
                        stop_reason="rejected",
                        resolved_plan=resolved_plan,
                    )

                if "local_editing" in final_action and self.executor_registry is not None:
                    try:
                        from physgenloop.learning_repair.contracts import RepairAction, RepairDecision
                        from physgenloop.learning_repair.executors import ExecutionRequest

                        local_decision = decision
                        if str(getattr(decision, "action", "")).lower() != "local_editing":
                            local_decision = RepairDecision(
                                action=RepairAction.LOCAL_EDITING,
                                confidence=decision.confidence,
                                instruction=decision.instruction,
                                action_probabilities=decision.action_probabilities,
                                per_action_values=decision.per_action_values,
                                parameters=dict(getattr(decision, "parameters", {})),
                                local_target=decision.local_target,
                                source=decision.source,
                                abstained=decision.abstained,
                                fallback_reason=decision.fallback_reason,
                                compatibility_id=decision.compatibility_id,
                            )
                        exec_request = ExecutionRequest(
                            candidate=selected.candidate,
                            prompt=current_prompt,
                            physics_plan=resolved_plan,
                            critic_report=selected.report,
                            decision=local_decision,
                            seed=self.config.base_seed + round_index + 1000,
                        )
                        exec_result = self.executor_registry.execute(exec_request)
                        if exec_result.status == "succeeded" and exec_result.candidate is not None:
                            edit_report = self.critic.evaluate(
                                exec_result.candidate,
                                prompt=current_prompt,
                                physics_plan=resolved_plan,
                            )
                            round_winners.append(CandidateEvaluation(exec_result.candidate, edit_report))
                    except Exception as exc:
                        import sys
                        print(f"[LoopController] LOCAL_EDITING failed ({exc}), falling back to prompt repair", file=sys.stderr)
                elif "global_regeneration" in final_action:
                    current_prompt = original_prompt
                else:
                    current_prompt = next_prompt
            else:
                current_prompt = self.repairer.repair(prompt=current_prompt, report=selected.report)

        return LoopResult(
            best=self.selector.select(tuple(round_winners)),
            history=tuple(history),
            stop_reason="max_rounds",
            resolved_plan=resolved_plan,
        )
"""
        write_file(sftp, f"{ROOT}/src/physgenloop/controller.py", controller)

        run_loop = read_file(sftp, f"{ROOT}/agents/wanphysics/run_loop.py")
        run_loop = run_loop.replace(
            "    repairer = load_action_value_repairer(ckpt_root, max_attempts=max_rounds)\n",
            "    repairer = load_action_value_repairer(\n"
            "        ckpt_root,\n"
            "        max_attempts=max_rounds,\n"
            "        local_editor_available=not args.no_executor_registry,\n"
            "    )\n",
        )
        run_loop = run_loop.replace(
            '    config = LoopConfig(\n        max_rounds=max_rounds,\n        candidates_per_round=candidates_per_round,\n        acceptance_score=loop["acceptance_score"],\n        base_seed=loop["base_seed"],\n    )\n',
            '    config = LoopConfig(\n        max_rounds=max_rounds,\n        candidates_per_round=candidates_per_round,\n        acceptance_score=loop["acceptance_score"],\n        base_seed=loop["base_seed"],\n        error_scope_threshold=loop.get("error_scope_threshold", 0.4),\n        default_total_frames=gen["num_frames"],\n    )\n',
        )
        write_file(sftp, f"{ROOT}/agents/wanphysics/run_loop.py", run_loop)

        loop_yaml = """# PhysGenLoop 闭环运行配置
paths:
  root: /root/PhysGenLoop-
  models:
    wan: /root/PhysGenLoop-/models/wan2.2_ti2v_5b
    vllm: /root/PhysGenLoop-/models/Qwen3-VL-8B-Instruct
    sam2: /root/PhysGenLoop-/models/sam2.1_hiera_base_plus.pt
  checkpoints:
    repair_agent: /root/PhysGenLoop-/checkpoints/repair_agent/repair-agent-v3.1-proxy-20260717
  outputs: /root/PhysGenLoop-/outputs
  videophy2_manifest: /root/PhysGenLoop-/evaluation/manifests/videophy2_pilot300.json
  envs:
    main: /root/PhysGenLoop-/envs/main/bin/python
    vllm: /root/PhysGenLoop-/envs/vllm-cu128/bin/python

loop:
  prompt: "a red ball rolling on a flat table"
  max_rounds: 2
  candidates_per_round: 1
  acceptance_score: 0.8
  base_seed: 42
  error_scope_threshold: 0.4

generator:
  num_frames: 81
  height: 704
  width: 1280
  fps: 24
  negative_prompt: null

vllm:
  gpu_util: 0.85
  max_model_len: 16384
  served_name: qwen3-vl-8b-instruct
"""
        write_file(sftp, f"{ROOT}/configs/loop.yaml", loop_yaml)

        run_videophy2_loop = """\"\"\"videophy2 manifest/CSV 驱动的 PhysGenLoop 全链路入口。\"\"\"
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, "/root/PhysGenLoop-")
sys.path.insert(0, "/root/PhysGenLoop-/src")

from dotenv import load_dotenv

load_dotenv("/root/PhysGenLoop-/.env")

from physgenloop.controller import LoopController
from physgenloop.contracts import LoopConfig
from physgenloop.selector import EvidenceAwareSelector

from generators.wanphysics.adapter import WanSubprocessGenerator
from generators.wanphysics.executor_factory import build_executor_registry
from generators.wanphysics.repairer import load_action_value_repairer
from generators.wanphysics.sam2_vlm_critic import Sam2VlmSubprocessCritic

_DEFAULT_CONFIG = Path("/root/PhysGenLoop-/configs/loop.yaml")


def _load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class _VllmHandoffCritic:
    def __init__(self, inner: Sam2VlmSubprocessCritic, max_rounds: int) -> None:
        self._inner = inner
        self._max_rounds = max_rounds
        self._round_count = 0

    def evaluate(self, candidate, *, prompt, physics_plan):
        report = self._inner.evaluate(candidate, prompt=prompt, physics_plan=physics_plan)
        self._round_count += 1
        if self._round_count < self._max_rounds:
            self._inner.stop_vllm()
        return report

    def shutdown(self) -> None:
        self._inner.shutdown()


def _load_samples(manifest: str | None, csv_path: str | None) -> list[dict]:
    if manifest:
        with open(manifest, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload.get("samples", []))
    if csv_path:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("manifest or csv is required")


def _pick_prompt(sample: dict, prompt_field: str) -> str:
    for key in (prompt_field, "prompt", "caption", "upsampled_caption"):
        value = str(sample.get(key, "") or "").strip()
        if value:
            return value
    raise ValueError(f"missing prompt field: {prompt_field}")


def _error_scope_trace(result) -> list[dict]:
    traces: list[dict] = []
    for round_record in result.history:
        selected = next(
            (item for item in round_record.evaluations if item.candidate.candidate_id == round_record.selected_candidate_id),
            None,
        )
        if selected is None:
            continue
        trace = selected.report.diagnostics.get("error_scope")
        if trace:
            traces.append(trace)
    return traces


def _write_trials(path: Path, result) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, round_record in enumerate(result.history):
            before_eval = round_record.evaluations[0] if round_record.evaluations else None
            after_eval = result.history[index + 1].evaluations[0] if index + 1 < len(result.history) else None
            trial = {
                "round_index": round_record.round_index,
                "prompt": round_record.prompt,
                "before_candidate_id": before_eval.candidate.candidate_id if before_eval else None,
                "before_physics_score": before_eval.report.physics_score if before_eval else None,
                "before_decision": before_eval.report.decision if before_eval else None,
                "before_detector_backend": (before_eval.report.diagnostics.get("detector_backend") if before_eval else None),
                "after_candidate_id": after_eval.candidate.candidate_id if after_eval else None,
                "after_physics_score": after_eval.report.physics_score if after_eval else None,
                "stop_reason": result.stop_reason if index == len(result.history) - 1 else "continued",
            }
            handle.write(json.dumps(trial, ensure_ascii=False) + "\\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="videophy2 全链路闭环")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--prompt-field", choices=["caption", "upsampled_caption", "prompt"], default="caption")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id-field", default="sample_id")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--ckpt-root", default=None)
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    paths = cfg["paths"]
    loop = cfg["loop"]
    gen = cfg["generator"]
    vllm = cfg["vllm"]

    manifest = args.manifest or paths.get("videophy2_manifest")
    samples = _load_samples(manifest, args.csv_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        print("[videophy2] 没有可运行样本", file=sys.stderr)
        return 1

    max_rounds = args.max_rounds if args.max_rounds is not None else loop["max_rounds"]
    ckpt_root = args.ckpt_root or paths["checkpoints"]["repair_agent"]

    run_id = datetime.now().strftime("videophy2_run_%Y%m%d_%H%M%S")
    run_dir = Path(paths["outputs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    local_count = 0
    global_count = 0

    for index, sample in enumerate(samples):
        sample_id = str(sample.get(args.task_id_field) or f"sample-{index:04d}")
        prompt = _pick_prompt(sample, args.prompt_field)
        sample_dir = run_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        generator = WanSubprocessGenerator(
            python=paths["envs"]["main"],
            model_path=paths["models"]["wan"],
            output_root=str(sample_dir),
            num_frames=gen["num_frames"],
            height=gen["height"],
            width=gen["width"],
            fps=gen["fps"],
            negative_prompt=gen.get("negative_prompt"),
        )
        raw_critic = Sam2VlmSubprocessCritic(
            python=paths["envs"]["main"],
            vllm_python=paths["envs"]["vllm"],
            vllm_model=paths["models"]["vllm"],
            vllm_served_name=vllm["served_name"],
            vllm_log=str(run_dir / "vllm.log"),
            vllm_gpu_util=vllm["gpu_util"],
            vllm_max_model_len=vllm["max_model_len"],
        )
        critic = _VllmHandoffCritic(raw_critic, max_rounds=max_rounds * loop["candidates_per_round"])
        repairer = load_action_value_repairer(
            ckpt_root,
            max_attempts=max_rounds,
            local_editor_available=True,
        )
        selector = EvidenceAwareSelector()
        executor_registry = build_executor_registry(
            run_dir=str(sample_dir),
            ckpt_root=ckpt_root,
            python=paths["envs"]["main"],
            propainter_repo=paths.get("propainter_repo", "/root/ProPainter"),
            max_attempts=max_rounds,
        )
        config = LoopConfig(
            max_rounds=max_rounds,
            candidates_per_round=loop["candidates_per_round"],
            acceptance_score=loop["acceptance_score"],
            base_seed=loop["base_seed"] + index * 100,
            error_scope_threshold=loop.get("error_scope_threshold", 0.4),
            default_total_frames=gen["num_frames"],
        )
        controller = LoopController(
            generator=generator,
            critic=critic,
            repairer=repairer,
            selector=selector,
            config=config,
            executor_registry=executor_registry,
        )
        try:
            result = controller.run(prompt=prompt)
        finally:
            critic.shutdown()

        traces = _error_scope_trace(result)
        local_count += sum(1 for item in traces if item.get("final_action") == "local_editing")
        global_count += sum(1 for item in traces if item.get("scope") == "global")
        _write_trials(sample_dir / "trials.jsonl", result)

        summary = {
            "sample_id": sample_id,
            "prompt_field": args.prompt_field,
            "stop_reason": result.stop_reason,
            "best_candidate_id": result.best.candidate.candidate_id,
            "best_video_path": result.best.candidate.video_path,
            "best_physics_score": result.best.report.physics_score,
            "best_decision": result.best.report.decision,
            "rounds": len(result.history),
            "detector_backend": result.best.report.diagnostics.get("detector_backend", "unknown"),
            "error_scope_trace": traces,
        }
        (sample_dir / "loop_result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        summaries.append(summary)

    accepted = sum(1 for item in summaries if item["stop_reason"] == "accepted")
    summary = {
        "run_id": run_id,
        "samples": len(summaries),
        "accepted": accepted,
        "acceptance_rate": (accepted / len(summaries)) if summaries else 0.0,
        "average_rounds": (sum(item["rounds"] for item in summaries) / len(summaries)) if summaries else 0.0,
        "local_editing_count": local_count,
        "global_scope_count": global_count,
        "results": summaries,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
        write_file(sftp, f"{ROOT}/agents/wanphysics/run_videophy2_loop.py", run_videophy2_loop)
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
