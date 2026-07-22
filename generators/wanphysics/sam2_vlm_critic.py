"""SAM2+VLM 子进程式 CandidateCritic，实现 physgenloop.interfaces.CandidateCritic 协议。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from pavg_critic.schemas import CriticReport, PhysicsPlan

from physgenloop.contracts import GeneratedCandidate

_EVAL_STEP = Path(__file__).parent.parent.parent / "agents" / "wanphysics" / "eval_step.py"
_HEALTH_URL = "http://localhost:8000/health"


def _report_from_dict(d: dict) -> CriticReport:
    from pavg_critic import schemas as _s
    if hasattr(_s.CriticReport, "from_dict"):
        return _s.CriticReport.from_dict(d)
    decision = d.get("decision", "violation")
    is_physical = decision == "physical"
    violations: tuple = ()
    if hasattr(_s, "Violation"):
        raw_v = d.get("violations", ())
        try:
            violations = tuple(_s.Violation.from_dict(v) for v in raw_v if isinstance(v, dict))
        except Exception:
            violations = ()
    evidence_bundles: tuple = ()
    if hasattr(_s, "EvidenceBundle"):
        raw_e = d.get("evidence_bundles", ())
        try:
            evidence_bundles = tuple(_s.EvidenceBundle.from_dict(e) for e in raw_e if isinstance(e, dict))
        except Exception:
            evidence_bundles = ()
    return CriticReport(
        is_physical=is_physical,
        physics_score=float(d.get("physics_score", 0.0)),
        confidence=float(d.get("confidence", 0.0)),
        decision=decision,
        coverage=float(d.get("coverage", 1.0)),
        score_breakdown=d.get("score_breakdown", {}),
        diagnostics=d.get("diagnostics", {}),
        model_versions=d.get("model_versions", {}),
        violations=violations,
        evidence_bundles=evidence_bundles,
    )


class Sam2VlmSubprocessCritic:
    def __init__(
        self,
        python: str,
        vllm_python: str,
        vllm_model: str = "/root/PhysGenLoop-/models/Qwen3-VL-8B-Instruct",
        vllm_served_name: str = "qwen3-vl-8b-instruct",
        vllm_log: str = "/root/PhysGenLoop-/outputs/vllm_qwen3vl_serve.log",
        vllm_gpu_util: float = 0.85,
        vllm_max_model_len: int = 16384,
        vllm_gpu_id: str | int | None = None,
        dual_gpu: bool = False,
    ) -> None:
        self._python = python
        self._vllm_python = vllm_python
        self._vllm_model = vllm_model
        self._vllm_served_name = vllm_served_name
        self._vllm_log = vllm_log
        self._vllm_gpu_util = vllm_gpu_util
        self._vllm_max_model_len = vllm_max_model_len
        # 双卡角色分工：把 vLLM(Qwen3-VL) 固定到某张卡（如 GPU1），与 Wan(GPU0) 分离。
        self._vllm_gpu_id = vllm_gpu_id
        # dual_gpu=True 时，Wan 与 vLLM 各占一卡，无需在生成前杀掉/重启 vLLM，
        # 省掉每轮 vLLM 冷启动（历史 ~836s）。
        self._dual_gpu = dual_gpu

    def _vllm_healthy(self) -> bool:
        try:
            urllib.request.urlopen(_HEALTH_URL, timeout=2)
            return True
        except Exception:
            return False

    def _gpu_memory_used_mb(self) -> int | None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        first = (result.stdout.strip().splitlines() or [""])[0].strip()
        if not first:
            return None
        try:
            return int(float(first))
        except ValueError:
            return None

    def _wait_for_gpu_release(self, *, threshold_mb: int = 1024, timeout_s: int = 120) -> None:
        deadline = time.time() + timeout_s
        last_seen = None
        while time.time() < deadline:
            used = self._gpu_memory_used_mb()
            last_seen = used
            if used is None or used <= threshold_mb:
                return
            time.sleep(2)
        raise RuntimeError(
            f"GPU memory did not fall below {threshold_mb} MiB within {timeout_s}s; last_seen={last_seen} MiB"
        )

    def start_vllm(self) -> None:
        if self._vllm_healthy():
            return
        print("[Sam2VlmCritic] 启动 vLLM ...", file=sys.stderr)
        vllm_env = None
        if self._vllm_gpu_id is not None and str(self._vllm_gpu_id) != "":
            import os as _os

            vllm_env = dict(_os.environ)
            vllm_env["CUDA_VISIBLE_DEVICES"] = str(self._vllm_gpu_id)
            print(f"[Sam2VlmCritic] vLLM 绑定 GPU {self._vllm_gpu_id}", file=sys.stderr)
        with open(self._vllm_log, "a", encoding="utf-8") as log_file:
            subprocess.Popen(
                [
                    self._vllm_python,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    self._vllm_model,
                    "--served-model-name",
                    self._vllm_served_name,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--gpu-memory-utilization",
                    str(self._vllm_gpu_util),
                    "--max-model-len",
                    str(self._vllm_max_model_len),
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=vllm_env,
            )
        for _ in range(180):
            if self._vllm_healthy():
                print("[Sam2VlmCritic] vLLM 就绪", file=sys.stderr)
                return
            time.sleep(2)
        raise RuntimeError("vLLM 启动超时（360s）")

    def stop_vllm(self) -> None:
        for pattern in (
            "VLLM::EngineCore",
            "vllm.entrypoints.openai.api_server",
            "vllm",
        ):
            subprocess.run(["pkill", "-9", "-f", pattern], check=False)
        for _ in range(30):
            if not self._vllm_healthy():
                break
            time.sleep(1)
        self._wait_for_gpu_release()
        print("[Sam2VlmCritic] vLLM 已停止且显存已释放", file=sys.stderr)

    def prepare_for_generation(self) -> None:
        # 双卡模式：Wan 在 GPU0，vLLM 常驻 GPU1，生成前无需清理 vLLM/显存。
        if self._dual_gpu:
            return
        if self._vllm_healthy() or (self._gpu_memory_used_mb() or 0) > 1024:
            print("[Sam2VlmCritic] 生成前清理 vLLM/GPU 显存", file=sys.stderr)
            self.stop_vllm()
        else:
            self._wait_for_gpu_release()

    def shutdown(self) -> None:
        self.stop_vllm()

    def evaluate(
        self,
        candidate: GeneratedCandidate,
        *,
        prompt: str,
        physics_plan: PhysicsPlan,
    ) -> CriticReport:
        self.start_vllm()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as cf:
            candidate_json = cf.name
            try:
                plan_dict = physics_plan.to_dict() if hasattr(physics_plan, "to_dict") else {}
            except Exception:
                plan_dict = {}
            json.dump(
                {
                    "candidate_id": candidate.candidate_id,
                    "video_path": candidate.video_path,
                    "prompt": candidate.prompt,
                    "seed": candidate.seed,
                    "metadata": candidate.metadata,
                    "physics_plan": plan_dict,
                },
                cf,
                ensure_ascii=False,
            )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as ef:
            out_json = ef.name

        subprocess.run(
            [
                self._python,
                str(_EVAL_STEP),
                "--candidate-json",
                candidate_json,
                "--out-json",
                out_json,
            ],
            check=True,
        )

        with open(out_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        Path(candidate_json).unlink(missing_ok=True)
        Path(out_json).unlink(missing_ok=True)

        report = _report_from_dict(payload["report"])
        detector_backend = payload.get("detector_backend", "unknown")
        if detector_backend != report.diagnostics.get("detector_backend"):
            from dataclasses import replace as _replace

            report = _replace(report, diagnostics={**report.diagnostics, "detector_backend": detector_backend})

        candidate_dir = Path(candidate.video_path).parent
        critic_path = candidate_dir / "critic.json"
        if critic_path.exists():
            critic_result = {
                "video": Path(candidate.video_path).name,
                "status": "completed",
                "physics_violation": report.decision != "physical",
                "reason": getattr(report, "summary", None),
                "confidence": report.confidence,
                "detector_backend": detector_backend,
            }
            critic_path.write_text(json.dumps(critic_result, ensure_ascii=False, indent=2), encoding="utf-8")

        return report
