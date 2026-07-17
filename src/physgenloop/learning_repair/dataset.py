"""Repair 监督清单的读取、审计和防泄漏切分。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
from typing import Iterable

from .contracts import RepairExample


@dataclass(frozen=True)
class DatasetAudit:
    sample_count: int
    group_count: int
    action_counts: dict[str, int]
    split_counts: dict[str, int]
    successful_count: int
    group_leakage: dict[str, tuple[str, ...]]
    missing_artifacts: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.group_leakage and not self.missing_artifacts

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "sample_count": self.sample_count,
            "group_count": self.group_count,
            "action_counts": self.action_counts,
            "split_counts": self.split_counts,
            "successful_count": self.successful_count,
            "group_leakage": {
                key: list(value) for key, value in self.group_leakage.items()
            },
            "missing_artifacts": list(self.missing_artifacts),
        }


def load_repair_manifest(path: str | Path) -> tuple[RepairExample, ...]:
    """读取 JSONL，或读取 JSON 数组/带 ``samples`` 的 JSON 对象。"""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    records: list[object]
    if source.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    else:
        raw = json.loads(text)
        if isinstance(raw, dict):
            raw = raw.get("samples")
        if not isinstance(raw, list):
            raise ValueError("repair manifest must be JSONL, a JSON array, or contain samples")
        records = raw

    examples: list[RepairExample] = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"repair sample {index} must be an object")
        try:
            example = RepairExample.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid repair sample {index}: {exc}") from exc
        if example.sample_id in seen:
            raise ValueError(f"duplicate sample_id: {example.sample_id}")
        seen.add(example.sample_id)
        examples.append(example)
    if not examples:
        raise ValueError("repair manifest contains no samples")
    return tuple(examples)


def write_repair_manifest(examples: Iterable[RepairExample], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = tuple(examples)
    if destination.suffix.lower() == ".jsonl":
        text = "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        )
    else:
        text = json.dumps(
            {"schema_version": "1.0", "samples": [item.to_dict() for item in records]},
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    destination.write_text(text, encoding="utf-8")


def collect_repair_samples(
    root: str | Path,
    *,
    record_name: str = "repair_sample.json",
) -> tuple[RepairExample, ...]:
    """递归收集 Blender 作业写出的逐样本记录并拒绝重复 ID。"""

    source = Path(root)
    paths = sorted(source.rglob(record_name))
    if not paths:
        raise ValueError(f"no {record_name!r} records found below {source}")
    records = []
    seen = set()
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"sample record must be an object: {path}")
        try:
            item = RepairExample.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid sample record {path}: {exc}") from exc
        if item.sample_id in seen:
            raise ValueError(f"duplicate sample_id {item.sample_id!r} in {path}")
        seen.add(item.sample_id)
        records.append(item)
    return tuple(records)


def audit_dataset(
    examples: Iterable[RepairExample],
    *,
    check_artifacts: bool = False,
    base_dir: str | Path | None = None,
) -> DatasetAudit:
    records = tuple(examples)
    by_group: dict[str, set[str]] = defaultdict(set)
    missing: list[str] = []
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    for example in records:
        if example.split is not None:
            by_group[example.group_id].add(example.split)
        if check_artifacts:
            for name, raw_path in example.artifacts.items():
                path = Path(raw_path)
                path = path if path.is_absolute() else root / path
                if not path.exists():
                    missing.append(f"{example.sample_id}:{name}:{raw_path}")
    leakage = {
        group: tuple(sorted(splits))
        for group, splits in by_group.items()
        if len(splits) > 1
    }
    return DatasetAudit(
        sample_count=len(records),
        group_count=len({item.group_id for item in records}),
        action_counts=dict(Counter(item.target_action.value for item in records)),
        split_counts=dict(Counter(item.split or "unassigned" for item in records)),
        successful_count=sum(item.successful for item in records),
        group_leakage=leakage,
        missing_artifacts=tuple(sorted(missing)),
    )


def grouped_split(
    examples: Iterable[RepairExample],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[RepairExample, ...]:
    """按 group 随机切分，同一基础场景的所有变体保持在同一集合。"""

    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    records = tuple(examples)
    groups = sorted({item.group_id for item in records})
    random.Random(seed).shuffle(groups)
    group_count = len(groups)
    test_count = round(group_count * test_fraction)
    validation_count = round(group_count * validation_fraction)
    if group_count >= 3 and test_fraction > 0:
        test_count = max(1, test_count)
    if group_count - test_count >= 2 and validation_fraction > 0:
        validation_count = max(1, validation_count)
    while test_count + validation_count >= group_count and validation_count > 0:
        validation_count -= 1
    while test_count + validation_count >= group_count and test_count > 0:
        test_count -= 1
    test_groups = set(groups[:test_count])
    validation_groups = set(groups[test_count : test_count + validation_count])
    result = []
    for item in records:
        split = (
            "test"
            if item.group_id in test_groups
            else "validation"
            if item.group_id in validation_groups
            else "train"
        )
        result.append(replace(item, split=split))
    return tuple(result)


def select_split(
    examples: Iterable[RepairExample], split: str
) -> tuple[RepairExample, ...]:
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    return tuple(item for item in examples if item.split == split)
