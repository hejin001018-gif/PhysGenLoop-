"""videophy2 manifest/CSV 驱动的 PhysGenLoop 全链路入口。"""
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

    def prepare_for_generation(self) -> None:
        self._inner.prepare_for_generation()

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
            handle.write(json.dumps(trial, ensure_ascii=False) + "\n")


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
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
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
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
