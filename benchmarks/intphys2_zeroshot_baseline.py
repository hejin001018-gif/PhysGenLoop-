"""IntPhys 2 零样本 VLM 基线。

对齐 框架骨干§十：再斩第 2 步。
在 IntPhys 2 dev 分片上跑 Qwen2.5-VL / InternVL 零样本二分类，
产出 AUC / F1 / 每类正确率。

真实模型调用留 stub，先跑通数据加载 + 输出 schema 校验。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Protocol


PHYSICS_CATEGORIES = ["gravity", "collision", "solidity", "object_permanence"]


@dataclass
class PairSample:
    pair_id: str
    normal_video: Path
    violation_video: Path
    category: str  # 见 PHYSICS_CATEGORIES


@dataclass
class Prediction:
    pair_id: str
    category: str
    score_normal: float     # P(is_physical) for normal video
    score_violation: float  # P(is_physical) for violation video
    correct: bool           # score_normal > score_violation


class VLMScorer(Protocol):
    """异源 VLM 打分器接口。真实实现见 models/vlm/qwen25vl.py（TODO）。"""

    name: str

    def score_is_physical(self, video_path: Path) -> float: ...


def load_intphys2_dev(dev_root: Path) -> Iterable[PairSample]:
    """加载 IntPhys 2 dev 分片。目录约定：
    dev_root/
      <category>/
        <pair_id>/
          possible.mp4       # normal
          impossible.mp4     # violation
    """
    for cat_dir in sorted(dev_root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name not in PHYSICS_CATEGORIES:
            continue
        for pair_dir in sorted(cat_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            normal = pair_dir / "possible.mp4"
            viol = pair_dir / "impossible.mp4"
            if not normal.exists() or not viol.exists():
                continue
            yield PairSample(
                pair_id=f"{cat_dir.name}/{pair_dir.name}",
                normal_video=normal,
                violation_video=viol,
                category=cat_dir.name,
            )


def evaluate(scorer: VLMScorer, samples: Iterable[PairSample]) -> dict:
    preds: list[Prediction] = []
    for s in samples:
        sn = scorer.score_is_physical(s.normal_video)
        sv = scorer.score_is_physical(s.violation_video)
        preds.append(Prediction(
            pair_id=s.pair_id,
            category=s.category,
            score_normal=sn,
            score_violation=sv,
            correct=sn > sv,
        ))

    per_cat: dict[str, dict] = {c: {"total": 0, "correct": 0} for c in PHYSICS_CATEGORIES}
    for p in preds:
        per_cat[p.category]["total"] += 1
        per_cat[p.category]["correct"] += int(p.correct)

    overall_total = sum(v["total"] for v in per_cat.values())
    overall_correct = sum(v["correct"] for v in per_cat.values())

    return {
        "scorer": scorer.name,
        "overall_accuracy": overall_correct / overall_total if overall_total else 0.0,
        "per_category": {
            c: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": (v["correct"] / v["total"]) if v["total"] else 0.0,
            }
            for c, v in per_cat.items()
        },
        "predictions": [asdict(p) for p in preds],
    }


class _StubScorer:
    """占位 scorer。真实模型接入后替换。"""
    name = "stub-random"

    def score_is_physical(self, video_path: Path) -> float:
        import random
        random.seed(hash(str(video_path)) & 0xFFFFFFFF)
        return random.random()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-root", type=Path, required=True, help="IntPhys 2 dev 分片根目录")
    ap.add_argument("--out", type=Path, default=Path("outputs/intphys2_zeroshot.json"))
    ap.add_argument("--scorer", choices=["stub", "qwen25vl", "internvl"], default="stub")
    args = ap.parse_args()

    if args.scorer == "stub":
        scorer: VLMScorer = _StubScorer()
    else:
        raise NotImplementedError(
            f"scorer={args.scorer} 尚未接入。见 models/vlm/ TODO。"
        )

    samples = list(load_intphys2_dev(args.dev_root))
    if not samples:
        print(f"[warn] no pairs found under {args.dev_root}")
        return 1

    report = evaluate(scorer, samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"overall_accuracy = {report['overall_accuracy']:.4f}  n={len(samples)}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
