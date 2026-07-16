"""PAVG Critic 的基线、消融运行器与无额外依赖评估指标。

B0 纯 PQSG 保持为独立预测输入，避免把 PAVG 的规则执行结果误称为论文基线；B1 和
M1–M5 使用同一批冻结样本与指标。外部 PQSG 仓库产生的 decision/score 可直接转换为
``EvaluationRecord`` 后调用 ``compute_metrics``，与 PAVG 结果并列表比较。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from .config import CriticConfig
from .pipeline import PhysicsCritic
from .schemas import CriticRequest, FrameState


ABLATION_MODES = (
    "B0_PQSG",
    "B1_RULE",
    "M1_GRAPH",
    "M2_CHECKLIST",
    "M3_MECHANICS",
    "M4_VLM",
    "M5_FULL",
)


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    label: str
    request: CriticRequest
    observations: tuple[FrameState, ...]
    floor_y: float | None


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    label: str
    prediction: str
    physics_score: float
    coverage: float


@dataclass(frozen=True)
class EvaluationMetrics:
    count: int
    accuracy: float
    violation_precision: float
    violation_recall: float
    violation_f1: float
    unknown_rate: float
    mean_coverage: float
    mean_physics_score: float


def build_ablation_config(mode: str) -> CriticConfig:
    """返回一个只改变模块开关/家族权重的可复现实验配置。"""

    normalized = mode.upper()
    if normalized not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode {mode!r}; choose from {ABLATION_MODES}")
    base = CriticConfig()
    if normalized == "B0_PQSG":
        # 仅定义评分边界；实际 B0 预测必须来自独立 PQSG QG/QA 运行结果。
        return replace(
            base,
            checklist=replace(base.checklist, enabled=False),
            mechanics=replace(base.mechanics, enabled=False),
            fusion=_family_weights(base, rules=0, pqsg=1, checklist=0, mechanics=0, vlm=0),
        )
    if normalized == "B1_RULE":
        return replace(
            base,
            question_graph=replace(base.question_graph, enabled=False),
            checklist=replace(base.checklist, enabled=False),
            mechanics=replace(base.mechanics, enabled=False),
            fusion=_family_weights(
                base, rules=1, pqsg=0, checklist=0, mechanics=0, vlm=0,
                minimum_coverage=0.3,
            ),
        )
    if normalized == "M1_GRAPH":
        return replace(
            base,
            checklist=replace(base.checklist, enabled=False),
            mechanics=replace(base.mechanics, enabled=False),
            fusion=_family_weights(base, rules=0.7, pqsg=0.3, checklist=0, mechanics=0, vlm=0),
        )
    if normalized == "M2_CHECKLIST":
        return replace(
            base,
            mechanics=replace(base.mechanics, enabled=False),
            fusion=_family_weights(
                base, rules=0.5, pqsg=0.25, checklist=0.25, mechanics=0, vlm=0
            ),
        )
    if normalized == "M3_MECHANICS":
        return replace(
            base,
            fusion=_family_weights(
                base, rules=0.4, pqsg=0.2, checklist=0.2, mechanics=0.2, vlm=0
            ),
        )
    if normalized == "M4_VLM":
        return replace(
            base,
            fusion=_family_weights(
                base, rules=0.35, pqsg=0.2, checklist=0.2, mechanics=0.15, vlm=0.1
            ),
        )
    return base


def _family_weights(
    base: CriticConfig,
    *,
    rules: float,
    pqsg: float,
    checklist: float,
    mechanics: float,
    vlm: float,
    minimum_coverage: float | None = None,
):
    return replace(
        base.fusion,
        rule_family_weight=rules,
        pqsg_family_weight=pqsg,
        checklist_family_weight=checklist,
        mechanics_family_weight=mechanics,
        vlm_family_weight=vlm,
        minimum_coverage=(
            base.fusion.minimum_coverage
            if minimum_coverage is None
            else minimum_coverage
        ),
    )


def load_evaluation_samples(path: str | Path) -> tuple[EvaluationSample, ...]:
    """读取冻结 JSON 轨迹夹具，不修改源文件。"""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        raise ValueError("Evaluation fixture must contain a samples array")
    samples = []
    for item in raw["samples"]:
        label = str(item["label"])
        if label not in {"physical", "violation"}:
            raise ValueError(f"Unsupported ground-truth label: {label}")
        samples.append(
            EvaluationSample(
                sample_id=str(item["sample_id"]),
                label=label,
                request=CriticRequest.from_dict(item["request"]),
                observations=tuple(
                    FrameState.from_dict(state) for state in item["observations"]
                ),
                floor_y=(None if item.get("floor_y") is None else float(item["floor_y"])),
            )
        )
    return tuple(samples)


def run_rule_evaluation(
    samples: Iterable[EvaluationSample], *, mode: str
) -> tuple[tuple[EvaluationRecord, ...], EvaluationMetrics]:
    """运行无需外部 API 的 B1/M1–M3；其他模式必须由调用方显式注入模型。"""

    if mode.upper() in {"B0_PQSG", "M4_VLM", "M5_FULL"}:
        raise ValueError(
            f"{mode.upper()} requires independent predictions or explicitly injected models"
        )
    critic = PhysicsCritic(build_ablation_config(mode))
    records = []
    for sample in samples:
        report = critic.analyze(
            sample.request,
            observations=sample.observations,
            floor_y=sample.floor_y,
        )
        records.append(
            EvaluationRecord(
                sample_id=sample.sample_id,
                label=sample.label,
                prediction=report.decision or "unknown",
                physics_score=report.physics_score,
                coverage=report.coverage,
            )
        )
    result = tuple(records)
    return result, compute_metrics(result)


def load_pqsg_evaluation_records(
    path: str | Path,
    *,
    threshold: float = 0.5,
    labels: Mapping[str, str] | None = None,
) -> tuple[EvaluationRecord, ...]:
    """把官方 ``scripts/run.py`` 的原生输出转换成独立 B0 记录。

    官方 ``score`` 是树传播后的物理可信度；条目若带 ``error`` 或缺少 score，则预测
    为 unknown。标签既可随条目保存，也可通过 sample-id 映射提供。
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("PQSG decision threshold must be in [0, 1]")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Official PQSG output root must be an array")
    result = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each PQSG output entry must be an object")
        sample_id = str(entry.get("id", entry.get("sample_id", "")))
        if not sample_id:
            raise ValueError("PQSG output entry is missing id")
        label = _ground_truth_label(entry, sample_id, labels)
        score_raw = entry.get("score")
        failed = bool(entry.get("error")) or score_raw is None
        if failed:
            prediction = "unknown"
            score = 0.5
        else:
            score = float(score_raw)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"PQSG score for {sample_id} must be in [0, 1]")
            prediction = "physical" if score >= threshold else "violation"
        total_nodes = _nested_count(entry.get("psg", {}).get("nodes", {}))
        answered_nodes = _nested_count(entry.get("answers", {}))
        coverage = (
            min(1.0, answered_nodes / total_nodes)
            if total_nodes
            else (0.0 if failed else 1.0)
        )
        result.append(
            EvaluationRecord(
                sample_id=sample_id,
                label=label,
                prediction=prediction,
                physics_score=score,
                coverage=coverage,
            )
        )
    return tuple(result)


def _ground_truth_label(
    entry: dict,
    sample_id: str,
    labels: Mapping[str, str] | None,
) -> str:
    raw = entry.get("label")
    if raw is None and "is_physical" in entry:
        raw = "physical" if bool(entry["is_physical"]) else "violation"
    if raw is None and labels is not None:
        raw = labels.get(sample_id)
    label = str(raw) if raw is not None else ""
    if label not in {"physical", "violation"}:
        raise ValueError(f"Missing/invalid ground-truth label for PQSG sample {sample_id}")
    return label


def _nested_count(value) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(len(items) for items in value.values() if isinstance(items, dict))


def compute_metrics(records: Iterable[EvaluationRecord]) -> EvaluationMetrics:
    items = tuple(records)
    if not items:
        raise ValueError("At least one evaluation record is required")
    correct = sum(item.prediction == item.label for item in items)
    true_positive = sum(
        item.label == "violation" and item.prediction == "violation" for item in items
    )
    false_positive = sum(
        item.label != "violation" and item.prediction == "violation" for item in items
    )
    false_negative = sum(
        item.label == "violation" and item.prediction != "violation" for item in items
    )
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvaluationMetrics(
        count=len(items),
        accuracy=correct / len(items),
        violation_precision=precision,
        violation_recall=recall,
        violation_f1=f1,
        unknown_rate=sum(item.prediction == "unknown" for item in items) / len(items),
        mean_coverage=sum(item.coverage for item in items) / len(items),
        mean_physics_score=sum(item.physics_score for item in items) / len(items),
    )
