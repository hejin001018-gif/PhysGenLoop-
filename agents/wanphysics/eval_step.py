"""一次性评估步骤：SAM2+VLM 评分，写出 CriticReport JSON 后退出。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from pavg_critic import OpenAIChatModel, PhysicsCritic, SAM2ObjectDetector, CriticRequest
from pavg_critic.schemas import CriticReport, Violation
from physgenloop.contracts import GeneratedCandidate


def _build_critic(vlm, video_path: str, prompt: str):
    mask_output_dir = str(Path(video_path).resolve().parent / "sam2_masks")
    try:
        detector = SAM2ObjectDetector(
            vlm,
            video_path,
            model_ckpt=os.environ.get(
                "SAM2_CHECKPOINT", str(_ROOT / "models" / "sam2.1_hiera_base_plus.pt")
            ),
            prompt=prompt,
            mask_output_dir=mask_output_dir,
        )
        return PhysicsCritic(detector=detector, use_physics_plan=False), "sam2+vlm", detector
    except Exception as exc:
        raise RuntimeError(
            f"strict SAM2+VLM backend unavailable: {type(exc).__name__}: {exc}"
        ) from exc


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


def _attach_detector_diagnostics(
    report: CriticReport,
    detector: SAM2ObjectDetector | None,
    backend: str,
) -> CriticReport:
    diagnostics = dict(report.diagnostics or {})
    diagnostics["detector_backend"] = backend
    if detector is not None:
        diagnostics.update(dict(getattr(detector, "diagnostics", {}) or {}))
    elif backend == "rules_fallback":
        diagnostics.update(
            {
                "degraded": True,
                "fallback_used": True,
                "degraded_reasons": ["sam2_backend_unavailable"],
            }
        )
    return replace(report, diagnostics=diagnostics)


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

    vlm = OpenAIChatModel(
        api_key=os.environ["API_KEY"],
        model=os.environ["VLM_MODEL"],
        base_url=os.environ["BASE_URL"],
        strict_json_schema=os.environ.get("VLM_STRICT_SCHEMA", "").lower() == "true",
    )
    critic, backend, detector = _build_critic(vlm, candidate.video_path, candidate.prompt)
    report = critic.analyze(
        CriticRequest(
            video_path=candidate.video_path,
            prompt=candidate.prompt,
        )
    )
    report = _attach_detector_diagnostics(report, detector, backend)
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
