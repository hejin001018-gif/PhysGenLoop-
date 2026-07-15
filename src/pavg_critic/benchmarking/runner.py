"""Append-only, resumable benchmark execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample


class BenchmarkMethod(Protocol):
    method_id: str

    def evaluate(self, sample: BenchmarkSample) -> BenchmarkPrediction: ...


class BenchmarkRunner:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def _completed(self) -> set[tuple[str, str]]:
        if not self.output_path.exists():
            return set()
        completed: set[tuple[str, str]] = set()
        for line_number, line in enumerate(
            self.output_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                raw = json.loads(line)
                completed.add((str(raw["sample_id"]), str(raw["method_id"])))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid prediction JSONL line {line_number}"
                ) from exc
        return completed

    def run(
        self,
        samples: Sequence[BenchmarkSample],
        methods: Sequence[BenchmarkMethod],
    ) -> tuple[BenchmarkPrediction, ...]:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        completed = self._completed()
        new_records: list[BenchmarkPrediction] = []
        with self.output_path.open("a", encoding="utf-8") as handle:
            for sample in samples:
                for method in methods:
                    key = (sample.sample_id, method.method_id)
                    if key in completed:
                        continue
                    prediction = method.evaluate(sample)
                    if (prediction.sample_id, prediction.method_id) != key:
                        raise ValueError(
                            f"method returned mismatched prediction key: {key}"
                        )
                    handle.write(
                        json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed.add(key)
                    new_records.append(prediction)
        return tuple(new_records)


def load_predictions(path: str | Path) -> tuple[BenchmarkPrediction, ...]:
    source = Path(path)
    if not source.is_file():
        return ()
    result: list[BenchmarkPrediction] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            result.append(BenchmarkPrediction.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction JSONL line {line_number}") from exc
    return tuple(result)
