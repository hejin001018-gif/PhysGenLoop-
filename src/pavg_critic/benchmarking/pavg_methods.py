"""PAVG benchmark adapters backed by a shared SAM2 observation cache."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

from pavg_critic.evaluation import build_ablation_config
from pavg_critic.pipeline import PhysicsCritic
from pavg_critic.planner import ModelPhysicsPlanner
from pavg_critic.schemas import CriticRequest, FrameState
from pavg_critic.vlm_verifier import CategoryGroupedVLMVerifier

from .contracts import BenchmarkPrediction, BenchmarkSample
from .model_cache import AuditedCachedModel, ModelCallEvent
from .pavg_diagnostics import (
    build_pavg_diagnostics,
    build_pavg_failure_diagnostics,
)
from .prompt_diagnostics import OracleRulePhysicsPlanner


ObservationProducer = Callable[[BenchmarkSample], tuple[FrameState, ...]]


class CachedObservationProvider:
    def __init__(
        self,
        cache_dir: str | Path,
        producer: ObservationProducer,
    ):
        self.cache_dir = Path(cache_dir)
        self.producer = producer

    def _path(self, sample_id: str) -> Path:
        if (
            not sample_id
            or Path(sample_id).name != sample_id
            or any(mark in sample_id for mark in ("/", "\\"))
        ):
            raise ValueError(
                f"unsafe sample ID for observation cache: {sample_id!r}"
            )
        return self.cache_dir / f"{sample_id}.json"

    def _metadata_path(self, sample_id: str) -> Path:
        self._path(sample_id)
        return self.cache_dir / f"{sample_id}.meta.json"

    @staticmethod
    def _total_video_frames(video_path: str) -> int | None:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
            return total if total > 0 else None
        except Exception:
            return None

    def _write_metadata(
        self,
        sample: BenchmarkSample,
        *,
        states: tuple[FrameState, ...] = (),
        failure: Exception | None = None,
        production_latency_sec: float | None = None,
    ) -> None:
        represented_frames = sorted({state.frame for state in states})
        tracks = {
            state.track_id or f"object:{state.object}"
            for state in states
        }
        total_frames = self._total_video_frames(sample.video_path)
        coverage = (
            len(represented_frames) / total_frames
            if total_frames is not None
            else None
        )
        payload = {
            "schema_version": "1.0",
            "sample_id": sample.sample_id,
            "video_sha256": sample.sha256,
            "total_video_frames": total_frames,
            "represented_frames": represented_frames,
            "observed_frame_count": len(represented_frames),
            "frame_coverage": coverage,
            "track_count": len(tracks),
            "production_latency_sec": production_latency_sec,
            "propagation_failure": (
                None
                if failure is None
                else {
                    "type": type(failure).__name__,
                    "message": str(failure)[:500],
                }
            ),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._metadata_path(sample.sample_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def get(self, sample: BenchmarkSample) -> tuple[FrameState, ...]:
        path = self._path(sample.sample_id)
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError(f"invalid observation cache: {path}")
            states = tuple(FrameState.from_dict(item) for item in raw)
            if not states:
                raise ValueError(f"empty observation cache: {path}")
            if not self._metadata_path(sample.sample_id).is_file():
                self._write_metadata(
                    sample,
                    states=states,
                    production_latency_sec=None,
                )
            return states
        production_started = perf_counter()
        try:
            states = tuple(self.producer(sample))
            if not states:
                raise ValueError(
                    f"observation provider produced no states for {sample.sample_id}"
                )
        except Exception as exc:
            production_latency = perf_counter() - production_started
            try:
                self._write_metadata(
                    sample,
                    failure=exc,
                    production_latency_sec=production_latency,
                )
            except OSError:
                pass
            raise
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                [state.to_dict() for state in states],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self._write_metadata(
            sample,
            states=states,
            production_latency_sec=perf_counter() - production_started,
        )
        return states


class PAVGMethod:
    def __init__(
        self,
        mode: str,
        observations: CachedObservationProvider,
        *,
        model_id: str | None,
        planner_model=None,
        question_model=None,
        verifier_model=None,
        verifier_detector_weight: float = 0.7,
        model_stages: Mapping[str, AuditedCachedModel] | None = None,
        output_method_id: str | None = None,
        oracle_plan: bool = False,
    ):
        if mode not in {
            "B1_RULE",
            "M1_GRAPH",
            "M2_CHECKLIST",
            "M3_MECHANICS",
            "M4_VLM",
            "M5_FULL",
        }:
            raise ValueError(f"Stage A PAVG mode is not supported: {mode}")
        if mode == "M4_VLM" and verifier_model is None:
            raise ValueError("M4_VLM requires an explicit verifier model")
        if mode == "M5_FULL" and any(
            model is None
            for model in (planner_model, question_model, verifier_model)
        ):
            raise ValueError(
                "M5_FULL requires planner, question, and verifier models"
            )
        if not 0.0 <= verifier_detector_weight <= 1.0:
            raise ValueError("verifier_detector_weight must be in [0, 1]")
        if oracle_plan and mode != "M5_FULL":
            raise ValueError("oracle_plan is valid only for M5_FULL")
        self.mode = mode
        self.method_id = output_method_id or mode
        self.observations = observations
        self.model_id = model_id
        self.planner_model = planner_model
        self.question_model = question_model
        self.model_stages = dict(model_stages or {})
        self.oracle_plan = oracle_plan
        self.verifier_detector_weight = verifier_detector_weight
        self.verifier = (
            CategoryGroupedVLMVerifier(
                verifier_model,
                model_name=model_id or "benchmark-verifier",
            )
            if mode in {"M4_VLM", "M5_FULL"}
            else None
        )

    def evaluate(self, sample: BenchmarkSample) -> BenchmarkPrediction:
        return self.evaluate_audited(sample)[0]

    def _stage_cursors(self) -> dict[str, int]:
        return {
            stage: model.event_count
            for stage, model in sorted(self.model_stages.items())
        }

    def _stage_events(
        self,
        cursors: Mapping[str, int],
    ) -> dict[str, tuple[ModelCallEvent, ...]]:
        return {
            stage: self.model_stages[stage].events_since(cursor)
            for stage, cursor in sorted(cursors.items())
        }

    def evaluate_audited(
        self,
        sample: BenchmarkSample,
    ) -> tuple[BenchmarkPrediction, dict[str, object]]:
        for model in self.model_stages.values():
            model.bind_sample(sample.sample_id)
        total_started = perf_counter()
        visible_frame_count = 0
        analysis_started = total_started
        analysis_latency = 0.0
        artifacts = None
        config = None
        cursors = self._stage_cursors()
        try:
            states = self.observations.get(sample)
            analysis_started = perf_counter()
            visible_frame_count = len({state.frame for state in states})
            config = build_ablation_config(self.mode)
            if self.mode == "M4_VLM":
                config = replace(
                    config,
                    fusion=replace(
                        config.fusion,
                        detector_weight=self.verifier_detector_weight,
                        vlm_weight=1.0 - self.verifier_detector_weight,
                    ),
                )
            critic_kwargs = {"vlm_verifier": self.verifier}
            if self.mode == "M5_FULL":
                critic_kwargs["question_model"] = self.question_model
                if self.oracle_plan:
                    critic_kwargs["physics_planner"] = OracleRulePhysicsPlanner(
                        ModelPhysicsPlanner(self.planner_model),
                        sample.physical_rules,
                        model_id=self.model_id or "benchmark-planner",
                    )
                else:
                    critic_kwargs["planner_model"] = self.planner_model
            artifacts = PhysicsCritic(config, **critic_kwargs).analyze_detailed(
                CriticRequest(
                    video_path=sample.video_path,
                    prompt=sample.prompt,
                ),
                observations=states,
            )
            report = artifacts.report
            analysis_latency = perf_counter() - analysis_started
            provider_failures = report.diagnostics.get("provider_failures", ())
            if self.mode in {"M4_VLM", "M5_FULL"} and any(
                str(item.get("stage", "")).startswith("vlm_review")
                for item in provider_failures
            ):
                raise RuntimeError("grouped VLM verification failed")
            if report.decision not in {"physical", "violation", "unknown"}:
                raise ValueError(f"invalid PAVG decision: {report.decision}")
            categories = tuple(
                sorted({item.category for item in report.violations})
            )
            evidence = tuple(
                sorted(
                    {
                        frame
                        for item in report.violations
                        for frame in item.critical_frames
                    }
                )
            )
            repairs = [
                item.repair_instruction
                for item in report.violations
                if item.repair_instruction
            ]
            prediction = BenchmarkPrediction(
                sample_id=sample.sample_id,
                method_id=self.method_id,
                model_id=self.model_id,
                semantic_score=None,
                physics_score=report.physics_score * 4 + 1,
                semantic_label="unknown",
                physics_label=report.decision,
                confidence=report.confidence,
                coverage=report.coverage,
                latency_sec=analysis_latency,
                visible_frame_count=visible_frame_count,
                violation_categories=categories,
                evidence_frames=evidence,
                repair_instruction="; ".join(repairs) or None,
            )
            diagnostics = build_pavg_diagnostics(
                sample=sample,
                method_id=self.method_id,
                artifacts=artifacts,
                config=config,
                stage_events=self._stage_events(cursors),
                analysis_latency_sec=analysis_latency,
                total_latency_sec=perf_counter() - total_started,
                visible_frame_count=visible_frame_count,
            )
            return prediction, diagnostics
        except Exception as exc:
            total_latency = perf_counter() - total_started
            prediction = BenchmarkPrediction(
                sample_id=sample.sample_id,
                method_id=self.method_id,
                model_id=self.model_id,
                semantic_score=None,
                physics_score=None,
                semantic_label="unknown",
                physics_label="unknown",
                confidence=0.0,
                coverage=0.0,
                latency_sec=total_latency,
                visible_frame_count=visible_frame_count,
                failure={
                    "type": type(exc).__name__,
                },
            )
            events = self._stage_events(cursors)
            if artifacts is not None and config is not None:
                diagnostics = build_pavg_diagnostics(
                    sample=sample,
                    method_id=self.method_id,
                    artifacts=artifacts,
                    config=config,
                    stage_events=events,
                    analysis_latency_sec=(
                        analysis_latency
                        or perf_counter() - analysis_started
                    ),
                    total_latency_sec=total_latency,
                    visible_frame_count=visible_frame_count,
                )
                diagnostics["failure"] = {"error_type": type(exc).__name__}
            else:
                diagnostics = build_pavg_failure_diagnostics(
                    sample=sample,
                    method_id=self.method_id,
                    stage_events=events,
                    total_latency_sec=total_latency,
                    visible_frame_count=visible_frame_count,
                    error=exc,
                )
            return prediction, diagnostics


def make_sam2_observation_producer(
    model,
    *,
    model_config: str,
    checkpoint: str,
) -> ObservationProducer:
    """Create the required dense SAM2 frontend for Stage A PAVG methods."""

    from pavg_critic.config import CriticConfig
    from pavg_critic.sam2_detector import SAM2ObjectDetector

    def produce(sample: BenchmarkSample) -> tuple[FrameState, ...]:
        detector = SAM2ObjectDetector(
            model,
            sample.video_path,
            model_cfg=model_config,
            model_ckpt=checkpoint,
            prompt=sample.prompt,
        )
        states, _ = PhysicsCritic(
            CriticConfig(),
            detector=detector,
        ).observe_video(sample.video_path)
        if not states:
            raise ValueError(
                f"observation provider produced no states for {sample.sample_id}"
            )
        return states

    return produce
